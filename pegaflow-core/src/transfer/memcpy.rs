//! DMA copy-engine backend: coalesce contiguous copies, then submit the merged
//! ranges either as one `cuMemcpyBatchAsync` call or as one directional
//! `cuMemcpyAsync` per range.
//!
//! Batch submission is the default when the driver supports it. A layer-first
//! load of Qwen3-4B is ~7.7k copies of 64 KiB that rarely merge, and per-call
//! submission cost then bounds the transfer at ~13 GB/s.
//!
//! `cuMemcpyBatchAsync*` was removed once (b96e80c) after "driver instability
//! on some CUDA stacks"; no stack or failure was recorded. The per-range path is
//! kept as the fallback:
//! - `PEGAFLOW_DISABLE_BATCH_MEMCPY=1` selects it for the whole process;
//! - drivers older than the batch API select it;
//! - a batch call that returns an error is retried per range, and batching is
//!   turned off for the rest of the process.
//!
//! Both paths enqueue stream-ordered copies, so callers see the same semantics.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cudarc::driver::{CudaStream, sys};
use log::{debug, info, warn};

use super::{CopyDesc, TransferBackend};

/// Oldest driver that exports the batch entry point this build calls.
#[cfg(feature = "cuda-13")]
const MIN_BATCH_DRIVER: i32 = 13000;
#[cfg(not(feature = "cuda-13"))]
const MIN_BATCH_DRIVER: i32 = 12080;

const DISABLE_BATCH_ENV: &str = "PEGAFLOW_DISABLE_BATCH_MEMCPY";

/// Set after a batch call fails; every backend then submits per range.
static BATCH_FAILED: AtomicBool = AtomicBool::new(false);

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
pub struct MemcpyBackend {
    batch: bool,
}

impl MemcpyBackend {
    /// Batch submission when the driver supports it and it is not disabled
    /// through `PEGAFLOW_DISABLE_BATCH_MEMCPY`; per-range submission otherwise.
    pub fn new() -> Self {
        let disabled = std::env::var_os(DISABLE_BATCH_ENV).is_some_and(|v| v != "0");
        let mut version = 0;
        // SAFETY: writes one c_int; a CUDA context is not required.
        let result = unsafe { sys::cuDriverGetVersion(&mut version) };
        let supported = result == sys::cudaError_enum::CUDA_SUCCESS && version >= MIN_BATCH_DRIVER;
        let batch = supported && !disabled;
        info!(
            "memcpy transfer submission: {} (driver={version}, min_batch_driver={MIN_BATCH_DRIVER}, {DISABLE_BATCH_ENV}={disabled})",
            if batch { "batch" } else { "per-range" }
        );
        Self { batch }
    }

    /// One `cuMemcpyAsync` per merged range, regardless of driver support.
    pub fn per_range() -> Self {
        Self { batch: false }
    }

    /// The submission path the next transfer takes: "batch" or "per-range".
    pub fn submission(&self) -> &'static str {
        if self.batch && !BATCH_FAILED.load(Ordering::Relaxed) {
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
        let batch = self.batch && !BATCH_FAILED.load(Ordering::Relaxed);
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
        if batch {
            match submit_batch(direction, &merged, stream) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    // Re-issuing every range is safe: each copy is a plain
                    // stream-ordered byte move, so a repeated one rewrites the
                    // same bytes.
                    if !BATCH_FAILED.swap(true, Ordering::Relaxed) {
                        warn!("{e}; falling back to per-range memcpy for this process");
                    }
                }
            }
        }
        submit_per_range(direction, &merged, stream)
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
}
