"""Provenance of the code a gate run measures (G0.provenance, official rule).

- ``overlay``: the connector-under-test arm sees another tree only through a
  dir holding a single ``pegaflow`` symlink, so that tree cannot shadow the
  probe, its worker extension, vLLM, torch or ``sitecustomize``.
- ``module_problems``: the module files a probe recorded in its spec dumps
  (``rsi_probe.module_files``) must resolve under the gate tree (probe,
  worker extension), the shared venv (vLLM, torch, interpreter prefix) and
  the arm's expected ``pegaflow`` dir.
- ``Trees.tree``: a tree's rev, *verified*: a clean git toplevel (HEAD), or
  ``.rsi-rev`` whose ``git ls-tree -r <rev> <subpath>`` blob ids equal the
  files on disk (trees fetched blob-less from ``remote``); untracked native
  extensions are allowed and their sha256 recorded (a ``.so`` cannot be tied
  to a rev). Anything else is ``verified: false``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

REMOTE = "https://github.com/gentlecold/pegaflow.git"
_IGNORED_DIRS = {"__pycache__"}
_NATIVE = (".so", ".pyd", ".dylib")


def overlay(under_test: Path, dest: Path) -> Path:
    """``dest`` with only ``pegaflow -> <under_test>/pegaflow``."""
    pkg = (under_test / "pegaflow").resolve()
    if not (pkg / "__init__.py").is_file():
        raise ValueError(f"{under_test}: no pegaflow package")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    (dest / "pegaflow").symlink_to(pkg, target_is_directory=True)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _under(path: str | None, root: Path) -> bool:
    if not path:
        return False
    p = Path(path).resolve()
    return p == root or root in p.parents


def venv_purelib() -> Path:
    return Path(sysconfig.get_path("purelib")).resolve()


def gate_pegaflow_dir() -> Path | None:
    """Where ``pegaflow`` resolves for the gate's interpreter (venv .pth)."""
    import importlib.util

    spec = importlib.util.find_spec("pegaflow")
    return Path(spec.origin).resolve().parent if spec and spec.origin else None


def module_problems(
    files: dict[str, Any], repo: Path, pegaflow_dir: Path | None, where: str
) -> list[str]:
    out = []
    for k in ("rsi.gates.rsi_probe", "worker_extension"):
        if k in files and not _under(files[k], repo.resolve()):
            out.append(f"{where}: {k} = {files[k]} not under the gate tree {repo}")
    if "rsi.gates.rsi_probe" not in files:
        out.append(f"{where}: probe module file not recorded")
    purelib = venv_purelib()
    for k in ("vllm", "torch"):
        if not _under(files.get(k), purelib):
            out.append(f"{where}: {k} = {files.get(k)} not in the shared venv {purelib}")
    if Path(files.get("prefix") or "").resolve() != Path(sys.prefix).resolve():
        out.append(f"{where}: interpreter prefix {files.get('prefix')} != {sys.prefix}")
    for k in ("sitecustomize", "usercustomize"):
        f = files.get(k)
        if f and not (_under(f, purelib) or _under(f, Path(sys.base_prefix).resolve())):
            out.append(f"{where}: {k} = {f} outside the venv / base interpreter")
    for k in ("pegaflow", "pegaflow.pegaflow"):
        f = files.get(k)
        if f and (pegaflow_dir is None or not _under(f, pegaflow_dir)):
            out.append(f"{where}: {k} = {f} not under the arm's pegaflow {pegaflow_dir}")
    return out


def _git(*a: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *a], cwd=cwd, text=True, stderr=subprocess.DEVNULL, timeout=300
    )


def _blob_id(p: Path) -> str:
    data = os.readlink(p).encode() if p.is_symlink() else p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _files(root: Path, sub: str) -> list[str]:
    base = root / sub
    if base.is_file() or base.is_symlink():
        return [sub]
    out = []
    for dirpath, dirs, names in os.walk(base):
        dirs[:] = [d for d in dirs if d not in _IGNORED_DIRS]
        for n in names:
            out.append(str((Path(dirpath) / n).relative_to(root)))
    return sorted(out)


class Trees:
    """Verified revs, one blob-less fetch per rev (git dir under ``work``)."""

    def __init__(self, work: Path, remote: str = REMOTE):
        self.work, self.remote = work, remote
        self._cache: dict[tuple[Path, tuple[str, ...]], dict[str, Any]] = {}

    def _listing(self, rev: str) -> dict[str, str] | str:
        gd = self.work / "git"
        try:
            if not gd.exists():
                _git("init", "-q", "--bare", str(gd))
            with_tree = subprocess.run(
                ["git", "-C", str(gd), "cat-file", "-e", f"{rev}^{{tree}}"],
                capture_output=True,
                timeout=60,
            )
            if with_tree.returncode:
                _git("-C", str(gd), "fetch", "-q", "--depth=1", "--filter=blob:none", self.remote, rev)  # fmt: skip
            out = _git("-C", str(gd), "ls-tree", "-r", rev)
        except (subprocess.SubprocessError, OSError) as e:
            return f"cannot fetch {rev} from {self.remote}: {type(e).__name__}"
        listing = {}
        for line in out.splitlines():
            meta, path = line.split("\t", 1)
            mode, kind, oid = meta.split()
            if kind == "blob":
                listing[path] = oid
        return listing

    def tree(self, root: Path, subpaths: tuple[str, ...]) -> dict[str, Any]:
        root = root.resolve()
        key = (root, subpaths)
        if key not in self._cache:
            self._cache[key] = self._tree(root, subpaths)
        return self._cache[key]

    def _tree(self, root: Path, subpaths: tuple[str, ...]) -> dict[str, Any]:
        rec: dict[str, Any] = {"root": str(root), "subpaths": list(subpaths), "verified": False}
        native = {
            f: sha256(root / f)
            for sub in subpaths
            if (root / sub).exists()
            for f in _files(root, sub)
            if f.endswith(_NATIVE)
        }
        rec["native_sha256"] = native
        try:
            top = Path(_git("rev-parse", "--show-toplevel", cwd=root).strip()).resolve()
        except (subprocess.SubprocessError, OSError):
            top = None
        if top == root:
            rec["how"] = "git"
            rec["rev"] = _git("rev-parse", "HEAD", cwd=root).strip()
            dirty = _git("status", "--porcelain", "--", *subpaths, cwd=root).strip()
            rec["verified"] = not dirty
            if dirty:
                rec["reason"] = f"uncommitted changes: {dirty.splitlines()[:5]}"
            return rec
        rev_file = root / ".rsi-rev"
        if not rev_file.is_file():
            rec |= {"how": "unknown", "rev": "unknown", "reason": "not a git toplevel, no .rsi-rev"}
            return rec
        rev = rev_file.read_text().strip()
        rec |= {"how": "rsi-rev", "rev": rev}
        listing = self._listing(rev)
        if isinstance(listing, str):
            rec["reason"] = listing
            return rec
        want = {
            p: o
            for p, o in listing.items()
            if any(p == s or p.startswith(s.rstrip("/") + "/") for s in subpaths)
        }
        have = [f for sub in subpaths if (root / sub).exists() for f in _files(root, sub)]
        missing = sorted(set(want) - set(have))
        extra = sorted(f for f in set(have) - set(want) if not f.endswith(_NATIVE))
        differ = sorted(f for f in set(have) & set(want) if _blob_id(root / f) != want[f])
        if not want:
            rec["reason"] = f"{rev} tracks nothing under {list(subpaths)}"
        elif missing or extra or differ:
            rec["reason"] = "tree differs from rev"
            rec |= {"missing": missing[:20], "extra": extra[:20], "differ": differ[:20]}
        else:
            rec["verified"] = True
            rec["files"] = len(want)
        return rec

    def cleanup(self) -> None:
        if (self.work / "git").exists():
            shutil.rmtree(self.work / "git")
        if self.work.is_dir() and not any(self.work.iterdir()):
            self.work.rmdir()


def tree_root(pegaflow_dir: Path) -> Path:
    """``<tree>/python/pegaflow`` -> ``<tree>``."""
    return pegaflow_dir.resolve().parents[1]
