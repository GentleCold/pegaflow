//! DMA copy-engine backend: coalesce contiguous copies, then submit the merged
//! ranges either as one `cuMemcpyBatchAsync` call or as one directional
//! `cuMemcpyAsync` per range.
//!
//! Batch submission is the default when the driver supports it. A layer-first
//! load with one 64 KiB page per (block, layer) is thousands of copies that
//! rarely merge, and per-call submission cost then bounds the transfer at
//! ~13 GB/s.
//!
//! `cuMemcpyBatchAsync*` was removed once (b96e80c) after "driver instability
//! on some CUDA stacks"; no stack or failure was recorded. The per-range path is
//! kept as the fallback:
//! - `PEGAFLOW_DISABLE_BATCH_MEMCPY` set to anything but empty, `0`, `false`,
//!   `no` or `off` selects it for the whole process;
//! - drivers older than the batch API select it;
//! - a batch call that returns an error is retried per range. If the retry
//!   succeeds, the batch API is at fault on this stack and that backend submits
//!   per range from then on. If the retry fails too, the ranges are at fault,
//!   the transfer errors, and batching stays on.
//!
//! Both paths enqueue stream-ordered copies, so callers see the same semantics.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cudarc::driver::{CudaStream, sys};
use log::{debug, info, warn};

use super::{CopyDesc, TransferBackend};
use crate::metrics::{MEMCPY_FALLBACK_RETRY_FAILED, MEMCPY_FALLBACK_RETRY_OK, core_metrics};

/// Oldest driver that exports the batch entry point this build calls.
#[cfg(feature = "cuda-13")]
const MIN_BATCH_DRIVER: i32 = 13000;
#[cfg(not(feature = "cuda-13"))]
const MIN_BATCH_DRIVER: i32 = 12080;

/// The batch entry point this build calls. cudarc resolves it lazily and
/// panics if libcuda does not export it, so it is probed before first use.
#[cfg(feature = "cuda-13")]
const BATCH_SYMBOL: &str = "cuMemcpyBatchAsync_v2";
#[cfg(not(feature = "cuda-13"))]
const BATCH_SYMBOL: &str = "cuMemcpyBatchAsync";

const DISABLE_BATCH_ENV: &str = "PEGAFLOW_DISABLE_BATCH_MEMCPY";

/// A run of input copies that are contiguous on both the device and the host
/// side, submitted as a single copy.
struct Merged {
    device: u64,
    host: *mut u8,
    size: usize,
}

/// Coalesce copies that are adjacent on both sides into larger ranges. Inputs
/// were already bounds-checked when their offsets were built, and merging only
/// joins adjacent validated ranges, so no further bounds check is needed.
fn merge(copies: &[CopyDesc]) -> Vec<Merged> {
    let mut out = Vec::with_capacity(copies.len());
    let mut i = 0;
    while i < copies.len() {
        let start = copies[i];
        let mut size = start.size;
        let mut j = i + 1;
        while j < copies.len() {
            let next = copies[j];
            let same_allocations = start.device_allocation == next.device_allocation
                && start.host_allocation == next.host_allocation;
            let device_contiguous = start.device + size as u64 == next.device;
            // SAFETY: pointer arithmetic used only for an address-equality check;
            // the result is never dereferenced.
            let host_contiguous = unsafe { start.host.add(size) } == next.host;
            if same_allocations && device_contiguous && host_contiguous {
                size += next.size;
                j += 1;
            } else {
                break;
            }
        }
        out.push(Merged {
            device: start.device,
            host: start.host,
            size,
        });
        i = j;
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    H2d,
    D2h,
}

impl Direction {
    fn name(self) -> &'static str {
        match self {
            Self::H2d => "h2d",
            Self::D2h => "d2h",
        }
    }
}

/// DMA copy-engine transfer backend.
///
/// Every GPU worker builds its own backend, so the batch-fault latch is scoped
/// to one worker on one device.
pub struct MemcpyBackend {
    /// Batch submission was selected at construction.
    batch: bool,
    /// A batch call failed and the per-range retry of the same ranges worked,
    /// so this backend submits per range from now on.
    batch_faulted: AtomicBool,
    /// Make every batch call fail before it enqueues anything.
    #[cfg(test)]
    inject_batch_error: bool,
}

/// Why [`MemcpyBackend::new`] picked its submission path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Choice {
    Batch,
    /// `PEGAFLOW_DISABLE_BATCH_MEMCPY` is set to a value that is not false.
    DisabledByEnv,
    /// The driver version is unknown or older than the batch API.
    OldDriver,
    /// The driver is new enough but libcuda does not export [`BATCH_SYMBOL`].
    MissingSymbol,
}

impl Choice {
    fn describe(self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::DisabledByEnv => "per-range (disabled by env)",
            Self::OldDriver => "per-range (driver older than the batch API)",
            Self::MissingSymbol => "per-range (libcuda does not export the batch symbol)",
        }
    }
}

/// Whether the loaded libcuda exports [`BATCH_SYMBOL`]. Looks it up in the
/// same library cudarc's lazy loader uses.
fn batch_symbol_present() -> bool {
    // SAFETY: `culib` returns the process-wide libcuda handle cudarc already
    // opened for `cuDriverGetVersion`; `get` only resolves the address and the
    // pointer is never called here.
    unsafe {
        sys::culib()
            .get::<*const std::ffi::c_void>(BATCH_SYMBOL.as_bytes())
            .is_ok()
    }
}

/// `PEGAFLOW_DISABLE_BATCH_MEMCPY` turns batching off unless its value is
/// empty, `0`, `false`, `no` or `off` (any case): a kill switch errs towards off.
fn disables(value: &std::ffi::OsStr) -> bool {
    let v = value.to_string_lossy().trim().to_ascii_lowercase();
    !matches!(v.as_str(), "" | "0" | "false" | "no" | "off")
}

/// `symbol_present` is only asked once the env and the driver version allow
/// batching.
fn choose(
    env: Option<&std::ffi::OsStr>,
    driver_version: Option<i32>,
    symbol_present: impl FnOnce() -> bool,
) -> Choice {
    if env.is_some_and(disables) {
        return Choice::DisabledByEnv;
    }
    match driver_version {
        Some(v) if v >= MIN_BATCH_DRIVER => {
            if symbol_present() {
                Choice::Batch
            } else {
                Choice::MissingSymbol
            }
        }
        _ => Choice::OldDriver,
    }
}

impl MemcpyBackend {
    /// Batch submission when the driver supports it and it is not disabled
    /// through `PEGAFLOW_DISABLE_BATCH_MEMCPY`; per-range submission otherwise.
    pub fn new() -> Self {
        let env = std::env::var_os(DISABLE_BATCH_ENV);
        let mut version = 0;
        // SAFETY: writes one c_int; a CUDA context is not required.
        let result = unsafe { sys::cuDriverGetVersion(&mut version) };
        let driver = (result == sys::cudaError_enum::CUDA_SUCCESS).then_some(version);
        let choice = choose(env.as_deref(), driver, batch_symbol_present);
        info!(
            "memcpy transfer submission: {} (driver={version}, min_batch_driver={MIN_BATCH_DRIVER}, {DISABLE_BATCH_ENV}={env:?})",
            choice.describe()
        );
        Self::with_batch(choice == Choice::Batch)
    }

    /// One `cuMemcpyAsync` per merged range, regardless of driver support.
    pub fn per_range() -> Self {
        Self::with_batch(false)
    }

    fn with_batch(batch: bool) -> Self {
        Self {
            batch,
            batch_faulted: AtomicBool::new(false),
            #[cfg(test)]
            inject_batch_error: false,
        }
    }

    fn batch_call(
        &self,
        direction: Direction,
        merged: &[Merged],
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        #[cfg(test)]
        if self.inject_batch_error {
            return Err("injected batch error".into());
        }
        submit_batch(direction, merged, stream)
    }

    fn uses_batch(&self) -> bool {
        self.batch && !self.batch_faulted.load(Ordering::Relaxed)
    }

    /// The submission path the next transfer takes: "batch" or "per-range".
    pub fn submission(&self) -> &'static str {
        if self.uses_batch() {
            "batch"
        } else {
            "per-range"
        }
    }

    fn submit(
        &self,
        direction: Direction,
        copies: &[CopyDesc],
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        let merged = merge(copies);
        if merged.is_empty() {
            return Ok(());
        }
        let batch = self.uses_batch();
        if log::log_enabled!(log::Level::Debug) {
            let bytes: usize = merged.iter().map(|m| m.size).sum();
            debug!(
                "memcpy {}: copies={} ranges={} submits={} bytes={bytes}",
                direction.name(),
                copies.len(),
                merged.len(),
                if batch { 1 } else { merged.len() }
            );
        }
        if !batch {
            return submit_per_range(direction, &merged, stream);
        }
        let Err(batch_err) = self.batch_call(direction, &merged, stream) else {
            return Ok(());
        };
        // Re-issuing every range is safe: each copy is a plain stream-ordered
        // byte move, so a range the failed batch already enqueued rewrites the
        // same bytes.
        let retry = submit_per_range(direction, &merged, stream);
        let metrics = core_metrics();
        match retry {
            Ok(()) => {
                metrics
                    .memcpy_batch_fallbacks
                    .add(1, &*MEMCPY_FALLBACK_RETRY_OK);
                // The same ranges went through per range, so the batch API is
                // at fault here: stop using it on this backend.
                if !self.batch_faulted.swap(true, Ordering::Relaxed) {
                    warn!(
                        "{batch_err}; the per-range retry worked, so this worker submits per range from now on"
                    );
                }
                Ok(())
            }
            Err(e) => {
                metrics
                    .memcpy_batch_fallbacks
                    .add(1, &*MEMCPY_FALLBACK_RETRY_FAILED);
                // The ranges themselves are bad; batching stays on.
                Err(format!("{batch_err}; per-range retry: {e}"))
            }
        }
    }
}

impl Default for MemcpyBackend {
    fn default() -> Self {
        Self::new()
    }
}

fn submit_per_range(
    direction: Direction,
    merged: &[Merged],
    stream: &Arc<CudaStream>,
) -> Result<(), String> {
    for m in merged {
        // SAFETY: `m.device` is a valid device address and `m.host` is pinned
        // host memory kept alive by the caller until the stream is
        // synchronized. The merged range stays within the validated bounds.
        let result = unsafe {
            match direction {
                Direction::H2d => sys::cuMemcpyHtoDAsync_v2(
                    m.device,
                    m.host as *const std::ffi::c_void,
                    m.size,
                    stream.cu_stream(),
                ),
                Direction::D2h => sys::cuMemcpyDtoHAsync_v2(
                    m.host as *mut std::ffi::c_void,
                    m.device,
                    m.size,
                    stream.cu_stream(),
                ),
            }
        };
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(format!(
                "cuMemcpy{}Async failed: {result:?}",
                match direction {
                    Direction::H2d => "HtoD",
                    Direction::D2h => "DtoH",
                }
            ));
        }
    }
    Ok(())
}

/// Enqueue every merged range with one batch call. Source access is stream
/// ordered, the same guarantee as the per-range path.
fn submit_batch(
    direction: Direction,
    merged: &[Merged],
    stream: &Arc<CudaStream>,
) -> Result<(), String> {
    let mut dsts = Vec::with_capacity(merged.len());
    let mut srcs = Vec::with_capacity(merged.len());
    let mut sizes = Vec::with_capacity(merged.len());
    for m in merged {
        // Host pointers are valid CUdeviceptr values under unified addressing.
        let host = m.host as sys::CUdeviceptr;
        let (dst, src) = match direction {
            Direction::H2d => (m.device, host),
            Direction::D2h => (host, m.device),
        };
        dsts.push(dst);
        srcs.push(src);
        sizes.push(m.size);
    }
    let unspecified = sys::CUmemLocation {
        type_: sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_INVALID,
        id: 0,
    };
    let mut attrs = [sys::CUmemcpyAttributes {
        srcAccessOrder: sys::CUmemcpySrcAccessOrder_enum::CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        srcLocHint: unspecified,
        dstLocHint: unspecified,
        flags: 0,
    }];
    // One attribute entry, applying to every copy from index 0.
    let mut attr_idxs = [0usize];

    // SAFETY: the three arrays hold `merged.len()` entries each and outlive the
    // call (the driver reads them before returning). Every range is a valid
    // device range plus pinned host memory kept alive by the caller until the
    // stream is synchronized. Merged ranges never overlap.
    let result = unsafe {
        batch_memcpy(
            dsts.as_mut_ptr(),
            srcs.as_mut_ptr(),
            sizes.as_mut_ptr(),
            merged.len(),
            attrs.as_mut_ptr(),
            attr_idxs.as_mut_ptr(),
            attrs.len(),
            stream.cu_stream(),
        )
    };
    if result != sys::cudaError_enum::CUDA_SUCCESS {
        return Err(format!(
            "cuMemcpyBatchAsync {} of {} ranges failed: {result:?}",
            direction.name(),
            merged.len()
        ));
    }
    Ok(())
}

#[cfg(feature = "cuda-13")]
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the CUDA driver API signature"
)]
unsafe fn batch_memcpy(
    dsts: *mut sys::CUdeviceptr,
    srcs: *mut sys::CUdeviceptr,
    sizes: *mut usize,
    count: usize,
    attrs: *mut sys::CUmemcpyAttributes,
    attr_idxs: *mut usize,
    num_attrs: usize,
    stream: sys::CUstream,
) -> sys::CUresult {
    // SAFETY: forwarded from `submit_batch`.
    unsafe {
        sys::cuMemcpyBatchAsync_v2(
            dsts, srcs, sizes, count, attrs, attr_idxs, num_attrs, stream,
        )
    }
}

#[cfg(not(feature = "cuda-13"))]
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the CUDA driver API signature"
)]
unsafe fn batch_memcpy(
    dsts: *mut sys::CUdeviceptr,
    srcs: *mut sys::CUdeviceptr,
    sizes: *mut usize,
    count: usize,
    attrs: *mut sys::CUmemcpyAttributes,
    attr_idxs: *mut usize,
    num_attrs: usize,
    stream: sys::CUstream,
) -> sys::CUresult {
    let mut fail_idx = 0usize;
    // SAFETY: forwarded from `submit_batch`; `fail_idx` is a valid out slot.
    unsafe {
        sys::cuMemcpyBatchAsync(
            dsts,
            srcs,
            sizes,
            count,
            attrs,
            attr_idxs,
            num_attrs,
            &mut fail_idx,
            stream,
        )
    }
}

impl TransferBackend for MemcpyBackend {
    fn h2d(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String> {
        self.submit(Direction::H2d, copies, stream)
    }

    fn d2h(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String> {
        self.submit(Direction::D2h, copies, stream)
    }

    fn name(&self) -> &'static str {
        "direct"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy(
        device: u64,
        host: *mut u8,
        device_allocation: usize,
        host_allocation: usize,
    ) -> CopyDesc {
        CopyDesc {
            device,
            host,
            host_device: 0,
            size: 4,
            device_allocation,
            host_allocation,
        }
    }

    #[test]
    fn merge_requires_contiguous_ranges_from_the_same_allocations() {
        let mut host = [0_u8; 8];
        let first_host = host.as_mut_ptr();
        // SAFETY: `host` contains eight bytes, so this points to its second half.
        let second_host = unsafe { first_host.add(4) };

        let merged = merge(&[copy(100, first_host, 1, 2), copy(104, second_host, 1, 2)]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].size, 8);

        for second in [copy(104, second_host, 3, 2), copy(104, second_host, 1, 4)] {
            assert_eq!(merge(&[copy(100, first_host, 1, 2), second]).len(), 2);
        }
    }

    #[test]
    fn kill_switch_and_driver_version_pick_the_submission_path() {
        use std::ffi::OsStr;
        let new_driver = Some(MIN_BATCH_DRIVER);
        let cases: [(&str, Option<&str>, Option<i32>, Choice); 9] = [
            ("unset", None, new_driver, Choice::Batch),
            ("empty", Some(""), new_driver, Choice::Batch),
            ("zero", Some("0"), new_driver, Choice::Batch),
            ("false", Some("FALSE"), new_driver, Choice::Batch),
            ("one", Some("1"), new_driver, Choice::DisabledByEnv),
            ("true", Some("true"), new_driver, Choice::DisabledByEnv),
            (
                "env wins over driver",
                Some("1"),
                None,
                Choice::DisabledByEnv,
            ),
            (
                "old driver",
                None,
                Some(MIN_BATCH_DRIVER - 10),
                Choice::OldDriver,
            ),
            ("unknown driver", None, None, Choice::OldDriver),
        ];
        for (id, env, driver, want) in cases {
            assert_eq!(
                choose(env.map(OsStr::new), driver, || true),
                want,
                "case {id}"
            );
        }
        // A new enough driver without the symbol falls back; the probe is not
        // consulted once the env or the version already decided.
        assert_eq!(choose(None, new_driver, || false), Choice::MissingSymbol);
        let probe = || -> bool { panic!("probe must not run") };
        assert_eq!(
            choose(Some(OsStr::new("1")), new_driver, probe),
            Choice::DisabledByEnv
        );
        assert_eq!(choose(None, None, probe), Choice::OldDriver);
    }

    /// The driver on a GPU host exports the symbol this build calls.
    #[test]
    #[ignore = "requires a CUDA driver"]
    fn installed_driver_exports_the_batch_symbol() {
        assert!(batch_symbol_present(), "{BATCH_SYMBOL} not exported");
    }

    /// A batch call that fails is retried per range: every range lands exactly
    /// where it should, the backend switches to per range, and a per-range
    /// failure leaves batching on. Covers an injected failure and the real one
    /// the driver gives on the legacy default stream.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn failed_batch_retries_per_range_and_lands_every_range() {
        use cudarc::driver::CudaContext;
        const N: usize = 64;
        const SEG: usize = 64 << 10;
        let ctx = CudaContext::new(0).expect("ctx");
        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe { sys::cuMemHostAlloc(&mut host_ptr, N * SEG, 0) };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemHostAlloc");
        let host = host_ptr as *mut u8;
        let mut device: sys::CUdeviceptr = 0;
        let r = unsafe { sys::cuMemAlloc_v2(&mut device, N * SEG) };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemAlloc");

        // Reverse order with a per-range allocation id: nothing merges, so
        // every range is a separate copy and a lost or misplaced one shows.
        let copies: Vec<CopyDesc> = (0..N)
            .map(|k| CopyDesc {
                device: device + ((N - 1 - k) * SEG) as u64,
                host: unsafe { host.add(k * SEG) },
                host_device: 0,
                size: SEG,
                device_allocation: k,
                host_allocation: k,
            })
            .collect();
        let pattern: Vec<u8> = (0..N * SEG).map(|i| (i * 131 + 17) as u8).collect();
        let host_slice = unsafe { std::slice::from_raw_parts_mut(host, N * SEG) };

        let mut check_h2d = |backend: &MemcpyBackend, stream: &Arc<CudaStream>, id: &str| {
            host_slice.copy_from_slice(&pattern);
            let r = unsafe { sys::cuMemsetD8_v2(device, 0, N * SEG) };
            assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "memset");
            backend.h2d(&copies, stream).expect(id);
            stream.synchronize().expect("sync");
            let mut out = vec![0u8; N * SEG];
            let r = unsafe { sys::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut _, device, N * SEG) };
            assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "DtoH");
            for k in 0..N {
                let want = &pattern[k * SEG..(k + 1) * SEG];
                let got = &out[(N - 1 - k) * SEG..(N - k) * SEG];
                assert!(got == want, "{id}: range {k} did not land");
            }
        };

        let created = ctx.new_stream().expect("stream");
        let mut injected = MemcpyBackend::with_batch(true);
        injected.inject_batch_error = true;
        assert_eq!(injected.submission(), "batch");
        check_h2d(&injected, &created, "injected");
        assert_eq!(injected.submission(), "per-range", "a good retry latches");

        // The batch API rejects the legacy default stream; per range accepts it.
        let legacy = ctx.default_stream();
        let real = MemcpyBackend::with_batch(true);
        check_h2d(&real, &legacy, "legacy stream");
        assert_eq!(
            real.submission(),
            "per-range",
            "the real batch failure latches"
        );

        // Bad ranges: batch and retry both fail, so batching stays on.
        let mut bad_ranges = MemcpyBackend::with_batch(true);
        bad_ranges.inject_batch_error = true;
        let bad = [CopyDesc {
            device,
            host: std::ptr::null_mut(),
            host_device: 0,
            size: SEG,
            device_allocation: 0,
            host_allocation: 0,
        }];
        let err = bad_ranges
            .d2h(&bad, &created)
            .expect_err("null host must fail");
        assert!(
            err.contains("injected batch error") && err.contains("per-range retry"),
            "{err}"
        );
        assert_eq!(
            bad_ranges.submission(),
            "batch",
            "bad ranges must not latch"
        );
        created.synchronize().expect("sync");

        unsafe {
            sys::cuMemFree_v2(device);
            sys::cuMemFreeHost(host_ptr);
        }
    }
}
