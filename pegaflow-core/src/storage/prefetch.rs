// Per-request prefetch state machine. A single Mutex is sufficient because
// prefetch operations are per-query (low frequency, never a bottleneck).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use log::{info, warn};
use parking_lot::Mutex;
use tokio::task::JoinHandle;

#[cfg(feature = "rdma")]
use crate::backing::RdmaFetchStore;
use crate::backing::{PrefetchChunkConsumer, PrefetchResult, RdmaPrefetchResult, SsdBackingStore};
use crate::block::{BlockKey, PrefetchStatus, SealedBlock};
use crate::internode::MetaServerClient;
use crate::metrics::core_metrics;

use super::read_cache::ReadCache;
use super::tier_attribution::{
    AttributionSource, TierAttribution, record_cache_tier_block_requests,
};

#[cfg(feature = "rdma")]
#[derive(Clone)]
pub(super) struct RdmaFetch(Arc<RdmaFetchStore>);
#[cfg(not(feature = "rdma"))]
#[derive(Clone)]
pub(super) struct RdmaFetch;

#[cfg(feature = "rdma")]
impl RdmaFetch {
    pub(super) fn new(store: Arc<RdmaFetchStore>) -> Self {
        Self(store)
    }

    async fn query_prefix(
        &self,
        namespace: &str,
        remaining_hashes: &[Vec<u8>],
    ) -> Option<(String, usize)> {
        self.0.query_prefix(namespace, remaining_hashes).await
    }

    async fn fetch_blocks(
        &self,
        remote_addr: &str,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        chunk_consumer: Option<PrefetchChunkConsumer>,
    ) -> RdmaPrefetchResult {
        self.0
            .fetch_blocks(remote_addr, req_id, namespace, hashes, chunk_consumer)
            .await
    }
}

#[cfg(not(feature = "rdma"))]
impl RdmaFetch {
    async fn query_prefix(
        &self,
        _namespace: &str,
        _remaining_hashes: &[Vec<u8>],
    ) -> Option<(String, usize)> {
        None
    }

    async fn fetch_blocks(
        &self,
        _remote_addr: &str,
        _req_id: &str,
        _namespace: &str,
        _hashes: &[Vec<u8>],
        _chunk_consumer: Option<PrefetchChunkConsumer>,
    ) -> RdmaPrefetchResult {
        RdmaPrefetchResult::default()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PrefetchSource {
    Ssd,
    Rdma,
}

impl PrefetchSource {
    const fn as_attribution(self) -> AttributionSource {
        match self {
            Self::Ssd => AttributionSource::Ssd,
            Self::Rdma => AttributionSource::Rdma,
        }
    }

    fn record_backpressure(self, blocks: usize) {
        if blocks == 0 {
            return;
        }
        let metrics = core_metrics();
        match self {
            Self::Ssd => metrics
                .ssd_prefetch_backpressure_blocks
                .add(blocks as u64, &[]),
            Self::Rdma => metrics
                .rdma_prefetch_backpressure_blocks
                .add(blocks as u64, &[]),
        }
    }

    fn record_active_delta(self, blocks: i64) {
        if blocks == 0 {
            return;
        }
        if self == Self::Rdma {
            core_metrics().rdma_prefetch_active_blocks.add(blocks, &[]);
        }
    }
}

struct PrefetchEntry {
    handle: JoinHandle<PrefetchTaskResult>,
    started_at: Instant,
}

struct PrefetchTaskResult {
    source: Option<PrefetchSource>,
    found: usize,
    fetched: usize,
    cache_inserts: PrefetchResult,
    cache_inserts_handled: bool,
    ready_blocks: Vec<Arc<SealedBlock>>,
    missing: usize,
}

struct PrefixScan<'a> {
    req_id: &'a str,
    namespace: &'a str,
    hashes: &'a [Vec<u8>],
    emit_tier_metrics: bool,
}

struct PrefetchStart<'a> {
    req_id: &'a str,
    namespace: &'a str,
    remaining: &'a [BlockKey],
    prefix_blocks: Vec<Arc<SealedBlock>>,
    total: usize,
    hit: usize,
    emit_tier_metrics: bool,
}

struct PrefetchTaskDeps {
    rdma_fetch: Option<RdmaFetch>,
    ssd_store: Option<Arc<SsdBackingStore>>,
    read_cache: Arc<ReadCache>,
    metaserver_client: Option<Arc<MetaServerClient>>,
    prefetch_state: Arc<Mutex<PrefetchState>>,
    max_prefetch_blocks: usize,
}

struct PrefetchTaskInput {
    req_id: String,
    namespace: String,
    remaining_keys: Vec<BlockKey>,
    prefix_blocks: Vec<Arc<SealedBlock>>,
    total: usize,
    hit: usize,
    emit_tier_metrics: bool,
}

struct PrefetchState {
    active: HashMap<String, PrefetchEntry>,
    /// Reserved prefetch block budget for active background tasks.
    reserved_prefetch_blocks: usize,
    /// req_ids where RDMA remote fetch returned zero blocks (remote evicted).
    /// Prevents re-triggering RDMA on every subsequent poll for the same request.
    failed_remote: HashMap<String, Instant>,
}

impl PrefetchState {
    fn remove_entry(&mut self, req_id: &str) -> Option<PrefetchEntry> {
        self.active.remove(req_id)
    }
}

struct PrefetchReservation {
    state: Arc<Mutex<PrefetchState>>,
    source: PrefetchSource,
    blocks: usize,
}

impl Drop for PrefetchReservation {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        state.reserved_prefetch_blocks = state.reserved_prefetch_blocks.saturating_sub(self.blocks);
        self.source.record_active_delta(-(self.blocks as i64));
    }
}

pub(super) struct PrefetchScheduler {
    state: Arc<Mutex<PrefetchState>>,
    ssd_store: Option<Arc<SsdBackingStore>>,
    rdma_fetch: Option<RdmaFetch>,
    metaserver_client: Option<Arc<MetaServerClient>>,
    max_prefetch_blocks: usize,
}

impl PrefetchScheduler {
    pub(super) fn new(
        ssd_store: Option<Arc<SsdBackingStore>>,
        rdma_fetch: Option<RdmaFetch>,
        metaserver_client: Option<Arc<MetaServerClient>>,
        max_prefetch_blocks: usize,
    ) -> Self {
        Self {
            state: Arc::new(Mutex::new(PrefetchState {
                active: HashMap::new(),
                reserved_prefetch_blocks: 0,
                failed_remote: HashMap::new(),
            })),
            ssd_store,
            rdma_fetch,
            metaserver_client,
            max_prefetch_blocks,
        }
    }

    pub(super) async fn check_and_prefetch(
        &self,
        read_cache: Arc<ReadCache>,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> PrefetchStatus {
        // Default: this call may be the first decision and should attribute.
        match self.poll_existing(&read_cache, req_id).await {
            PollResult::NoActivePrefetch => {}
            PollResult::StillLoading => {
                return PrefetchStatus::Loading;
            }
            PollResult::Ready(status) => return status,
        }

        self.full_prefix_scan(
            &read_cache,
            PrefixScan {
                req_id,
                namespace,
                hashes,
                emit_tier_metrics: true,
            },
            Arc::clone(&read_cache),
        )
        .await
    }

    async fn poll_existing(&self, read_cache: &ReadCache, req_id: &str) -> PollResult {
        let entry = {
            let mut state = self.state.lock();
            let Some(entry) = state.active.get(req_id) else {
                return PollResult::NoActivePrefetch;
            };
            if !entry.handle.is_finished() {
                return PollResult::StillLoading;
            }
            state
                .remove_entry(req_id)
                .expect("active entry must exist after readiness check")
        };

        let result = match entry.handle.await {
            Ok(result) => result,
            Err(err) => {
                warn!("Prefetch task failed for req_id={}: {}", req_id, err);
                PrefetchTaskResult {
                    source: None,
                    found: 0,
                    fetched: 0,
                    cache_inserts: Vec::new(),
                    cache_inserts_handled: false,
                    ready_blocks: Vec::new(),
                    missing: 0,
                }
            }
        };

        // RDMA remote node can return fewer blocks than MetaServer promised
        // (likely evicted). Don't re-trigger RDMA on subsequent scans.
        if result.source == Some(PrefetchSource::Rdma)
            && result.fetched < result.found
            && result.found > 0
        {
            self.state
                .lock()
                .failed_remote
                .insert(req_id.to_string(), Instant::now());
            info!(
                "RDMA prefetch returned fewer blocks than expected: req_id={} returned={} expected={}",
                req_id, result.fetched, result.found
            );
        }

        // RDMA-fetched blocks that survive cache admission are now resident on
        // this node. Re-advertise only those resident blocks to the MetaServer
        // so peers can discover and fetch from here too. SSD prefetch is
        // skipped: those blocks were already registered by this node's own save
        // path, and eviction explicitly unregisters them.
        let rdma_registration =
            if result.source == Some(PrefetchSource::Rdma) && !result.cache_inserts_handled {
                let (resident_keys, _rejected_blocks) =
                    read_cache.batch_insert_split_residency(result.cache_inserts);
                rdma_registration_from_resident_keys(result.source, &resident_keys)
            } else {
                if !result.cache_inserts_handled {
                    read_cache.batch_insert(result.cache_inserts);
                }
                None
            };

        if let Some(client) = &self.metaserver_client
            && let Some((namespace, hashes)) = rdma_registration
        {
            client.try_register_namespace(namespace, hashes);
        }

        PollResult::Ready(PrefetchStatus::Ready {
            blocks: result.ready_blocks,
            missing: result.missing,
        })
    }

    async fn full_prefix_scan(
        &self,
        read_cache: &ReadCache,
        scan: PrefixScan<'_>,
        read_cache_arc: Arc<ReadCache>,
    ) -> PrefetchStatus {
        let total_start = Instant::now();

        let key_build_start = Instant::now();
        let keys: Vec<BlockKey> = scan
            .hashes
            .iter()
            .map(|hash| BlockKey::new(scan.namespace.to_string(), hash.clone()))
            .collect();
        let key_build = key_build_start.elapsed();

        let cache_scan_start = Instant::now();
        let (hit, prefix_blocks) = read_cache.get_prefix_blocks(&keys);
        let cache_scan = cache_scan_start.elapsed();
        let remaining = &keys[hit..];

        let task_start = Instant::now();
        let task_started = !remaining.is_empty()
            && self.start_prefetch_task(
                PrefetchStart {
                    req_id: scan.req_id,
                    namespace: scan.namespace,
                    remaining,
                    prefix_blocks: prefix_blocks.clone(),
                    total: keys.len(),
                    hit,
                    emit_tier_metrics: scan.emit_tier_metrics,
                },
                read_cache_arc,
            );
        let task_schedule = task_start.elapsed();

        if task_started {
            info!(
                "Prefetch scheduling timing: req_id={} total_keys={} hit={} remaining={} key_build={:?} cache_scan={:?} task_schedule={:?} total={:?}",
                scan.req_id,
                keys.len(),
                hit,
                remaining.len(),
                key_build,
                cache_scan,
                task_schedule,
                total_start.elapsed()
            );
            PrefetchStatus::Loading
        } else {
            let missing = keys.len() - hit;
            record_tier_attribution(
                keys.len(),
                hit,
                /* loading = */ 0,
                /* loading_source = */ None,
                scan.emit_tier_metrics,
            );

            info!(
                "Prefetch local-hit timing: req_id={} total_keys={} hit={} missing={} key_build={:?} cache_scan={:?} task_schedule={:?} total={:?}",
                scan.req_id,
                keys.len(),
                hit,
                missing,
                key_build,
                cache_scan,
                task_schedule,
                total_start.elapsed()
            );
            PrefetchStatus::Ready {
                blocks: prefix_blocks,
                missing,
            }
        }
    }

    fn start_prefetch_task(&self, start: PrefetchStart<'_>, read_cache: Arc<ReadCache>) -> bool {
        if start.remaining.is_empty() {
            return false;
        }

        let mut state = self.state.lock();
        if state.active.contains_key(start.req_id) {
            return true;
        }

        let rdma_fetch = self
            .rdma_fetch
            .as_ref()
            .filter(|_| !state.failed_remote.contains_key(start.req_id))
            .cloned();

        if rdma_fetch.is_none() && self.ssd_store.is_none() {
            return false;
        }

        let deps = PrefetchTaskDeps {
            rdma_fetch,
            ssd_store: self.ssd_store.clone(),
            read_cache,
            metaserver_client: self.metaserver_client.clone(),
            prefetch_state: Arc::clone(&self.state),
            max_prefetch_blocks: self.max_prefetch_blocks,
        };
        let input = PrefetchTaskInput {
            req_id: start.req_id.to_string(),
            namespace: start.namespace.to_string(),
            remaining_keys: start.remaining.to_vec(),
            prefix_blocks: start.prefix_blocks,
            total: start.total,
            hit: start.hit,
            emit_tier_metrics: start.emit_tier_metrics,
        };

        let handle = tokio::spawn(async move { run_prefetch_task(deps, input).await });

        state.active.insert(
            start.req_id.to_string(),
            PrefetchEntry {
                handle,
                started_at: Instant::now(),
            },
        );
        true
    }

    /// Drop stale active entries and sweep old `failed_remote` entries.
    ///
    /// Dropping a `JoinHandle` detaches the task; it keeps running so RDMA
    /// transfer locks can still be released by the normal completion path.
    pub(super) fn gc_stale_entries(
        &self,
        active_max_age: std::time::Duration,
        failed_remote_max_age: std::time::Duration,
    ) -> (usize, usize) {
        let mut state = self.state.lock();
        let active_before = state.active.len();
        state
            .active
            .retain(|_, entry| entry.started_at.elapsed() < active_max_age);
        let active_removed = active_before - state.active.len();

        let failed_before = state.failed_remote.len();
        state
            .failed_remote
            .retain(|_, ts| ts.elapsed() < failed_remote_max_age);
        (active_removed, failed_before - state.failed_remote.len())
    }
}

/// Attribute this `query_prefetch` decision. Skips attribution when:
/// * `emit_tier_metrics == false` (e.g. post-completion fall-through);
/// * `total` is zero (no decision to attribute).
fn record_tier_attribution(
    total: usize,
    hit: usize,
    loading: usize,
    loading_source: Option<AttributionSource>,
    emit_tier_metrics: bool,
) {
    if !emit_tier_metrics || total == 0 {
        return;
    }
    let attribution = TierAttribution::classify(total, hit, loading, loading_source);
    record_cache_tier_block_requests(total, attribution);
}

fn reserve_prefetch_slots(
    state: Arc<Mutex<PrefetchState>>,
    max_prefetch_blocks: usize,
    requested: usize,
    source: PrefetchSource,
) -> Option<(usize, PrefetchReservation)> {
    let mut guard = state.lock();
    let available = max_prefetch_blocks.saturating_sub(guard.reserved_prefetch_blocks);

    if available == 0 {
        source.record_backpressure(requested);
        return None;
    }

    let reserved = requested.min(available);
    let skipped = requested - reserved;
    source.record_backpressure(skipped);
    guard.reserved_prefetch_blocks += reserved;
    drop(guard);
    source.record_active_delta(reserved as i64);

    Some((
        reserved,
        PrefetchReservation {
            state,
            source,
            blocks: reserved,
        },
    ))
}

fn build_ready_result(
    prefix_blocks: Vec<Arc<SealedBlock>>,
    total: usize,
    source: Option<PrefetchSource>,
    found: usize,
    requested_keys: &[BlockKey],
    cache_inserts: PrefetchResult,
) -> PrefetchTaskResult {
    let mut ready_blocks = prefix_blocks;
    let inserts_by_key: HashMap<_, _> = cache_inserts
        .iter()
        .map(|(key, block)| (key, block))
        .collect();
    ready_blocks.extend(
        requested_keys
            .iter()
            .map_while(|key| inserts_by_key.get(key).map(|block| Arc::clone(*block))),
    );
    let missing = total.saturating_sub(ready_blocks.len());
    PrefetchTaskResult {
        source,
        found,
        fetched: cache_inserts.len(),
        cache_inserts,
        cache_inserts_handled: false,
        ready_blocks,
        missing,
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "ready reconstruction carries the completed prefetch context without allocating a temporary struct"
)]
fn build_ready_result_from_cache(
    read_cache: &ReadCache,
    prefix_blocks: Vec<Arc<SealedBlock>>,
    total: usize,
    source: Option<PrefetchSource>,
    found: usize,
    fetched: usize,
    requested_keys: &[BlockKey],
    retained_blocks: PrefetchResult,
) -> PrefetchTaskResult {
    let mut ready_blocks = prefix_blocks;
    let retained_by_key: HashMap<_, _> = retained_blocks
        .iter()
        .map(|(key, block)| (key, block))
        .collect();

    for key in requested_keys {
        if let Some(block) = read_cache.get_block(key) {
            ready_blocks.push(block);
        } else if let Some(block) = retained_by_key.get(key) {
            ready_blocks.push(Arc::clone(*block));
        } else {
            break;
        }
    }

    let missing = total.saturating_sub(ready_blocks.len());
    PrefetchTaskResult {
        source,
        found,
        fetched,
        cache_inserts: Vec::new(),
        cache_inserts_handled: true,
        ready_blocks,
        missing,
    }
}

fn insert_and_register_rdma_blocks(
    read_cache: &ReadCache,
    metaserver_client: Option<&Arc<MetaServerClient>>,
    blocks: PrefetchResult,
) -> PrefetchResult {
    let (resident_keys, rejected_blocks) = read_cache.batch_insert_split_residency(blocks);
    if let Some(client) = metaserver_client
        && let Some((namespace, hashes)) =
            rdma_registration_from_resident_keys(Some(PrefetchSource::Rdma), &resident_keys)
    {
        client.try_register_namespace(namespace, hashes);
    }
    rejected_blocks
}

fn rdma_registration_from_resident_keys(
    source: Option<PrefetchSource>,
    resident_keys: &[BlockKey],
) -> Option<(String, Vec<Vec<u8>>)> {
    if source != Some(PrefetchSource::Rdma) || resident_keys.is_empty() {
        return None;
    }

    let namespace = resident_keys[0].namespace.clone();
    let hashes = resident_keys.iter().map(|key| key.hash.clone()).collect();
    Some((namespace, hashes))
}

async fn run_prefetch_task(deps: PrefetchTaskDeps, input: PrefetchTaskInput) -> PrefetchTaskResult {
    let PrefetchTaskDeps {
        rdma_fetch,
        ssd_store,
        read_cache,
        metaserver_client,
        prefetch_state,
        max_prefetch_blocks,
    } = deps;
    let PrefetchTaskInput {
        req_id,
        namespace,
        remaining_keys,
        prefix_blocks,
        total,
        hit,
        emit_tier_metrics,
    } = input;
    let remaining_hashes: Vec<Vec<u8>> = remaining_keys.iter().map(|k| k.hash.clone()).collect();

    if let Some(rdma) = rdma_fetch
        && let Some((node, found)) = rdma.query_prefix(&namespace, &remaining_hashes).await
    {
        if let Some((reserved, _reservation)) = reserve_prefetch_slots(
            Arc::clone(&prefetch_state),
            max_prefetch_blocks,
            found,
            PrefetchSource::Rdma,
        ) {
            let retained_blocks = Arc::new(Mutex::new(Vec::new()));
            let chunk_consumer: PrefetchChunkConsumer = {
                let read_cache = Arc::clone(&read_cache);
                let metaserver_client = metaserver_client.clone();
                let retained_blocks = Arc::clone(&retained_blocks);
                Arc::new(move |blocks| {
                    let rejected = insert_and_register_rdma_blocks(
                        &read_cache,
                        metaserver_client.as_ref(),
                        blocks,
                    );
                    retained_blocks.lock().extend(rejected);
                })
            };
            let rdma_result = rdma
                .fetch_blocks(
                    &node,
                    &req_id,
                    &namespace,
                    &remaining_hashes[..reserved],
                    Some(chunk_consumer),
                )
                .await;
            record_tier_attribution(
                total,
                hit,
                reserved,
                Some(PrefetchSource::Rdma.as_attribution()),
                emit_tier_metrics,
            );
            return if rdma_result.inserted_by_consumer {
                let fetched = rdma_result.fetched_blocks.min(reserved);
                let retained = std::mem::take(&mut *retained_blocks.lock());
                build_ready_result_from_cache(
                    &read_cache,
                    prefix_blocks,
                    total,
                    Some(PrefetchSource::Rdma),
                    reserved,
                    fetched,
                    &remaining_keys[..fetched],
                    retained,
                )
            } else {
                build_ready_result(
                    prefix_blocks,
                    total,
                    Some(PrefetchSource::Rdma),
                    reserved,
                    &remaining_keys[..reserved],
                    rdma_result.cache_inserts,
                )
            };
        }

        warn!(
            "RDMA prefetch skipped by active block backpressure: req_id={} found={}",
            req_id, found
        );
    }

    if let Some(ssd) = ssd_store {
        let found = ssd.prefix_len(&remaining_keys);
        if found == 0 {
            record_tier_attribution(total, hit, 0, None, emit_tier_metrics);
            return build_ready_result(prefix_blocks, total, None, 0, &[], Vec::new());
        }
        let Some((reserved, _reservation)) = reserve_prefetch_slots(
            prefetch_state,
            max_prefetch_blocks,
            found,
            PrefetchSource::Ssd,
        ) else {
            record_tier_attribution(total, hit, 0, None, emit_tier_metrics);
            return build_ready_result(prefix_blocks, total, None, 0, &[], Vec::new());
        };
        let keys = remaining_keys[..reserved].to_vec();
        let (found, blocks) = ssd.prefetch_prefix(keys).await;
        if found > 0 {
            record_tier_attribution(
                total,
                hit,
                found,
                Some(PrefetchSource::Ssd.as_attribution()),
                emit_tier_metrics,
            );
            return build_ready_result(
                prefix_blocks,
                total,
                Some(PrefetchSource::Ssd),
                found,
                &remaining_keys[..found],
                blocks,
            );
        }
    }

    record_tier_attribution(total, hit, 0, None, emit_tier_metrics);
    build_ready_result(prefix_blocks, total, None, 0, &[], Vec::new())
}

enum PollResult {
    NoActivePrefetch,
    StillLoading,
    Ready(PrefetchStatus),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefetch_state() -> Arc<Mutex<PrefetchState>> {
        Arc::new(Mutex::new(PrefetchState {
            active: HashMap::new(),
            reserved_prefetch_blocks: 0,
            failed_remote: HashMap::new(),
        }))
    }

    fn key(n: u8) -> BlockKey {
        BlockKey::new("ns".to_string(), vec![n])
    }

    fn block() -> Arc<SealedBlock> {
        Arc::new(SealedBlock::from_slots(Vec::new()))
    }

    #[test]
    fn ready_result_rebuilds_prefix_in_requested_key_order() {
        let local = block();
        let k1 = key(1);
        let k2 = key(2);
        let k3 = key(3);
        let b1 = block();
        let b2 = block();
        let b3 = block();

        let result = build_ready_result(
            vec![Arc::clone(&local)],
            4,
            Some(PrefetchSource::Ssd),
            3,
            &[k1.clone(), k2.clone(), k3.clone()],
            vec![
                (k2, Arc::clone(&b2)),
                (k1, Arc::clone(&b1)),
                (k3, Arc::clone(&b3)),
            ],
        );

        assert_eq!(result.ready_blocks.len(), 4);
        assert!(Arc::ptr_eq(&result.ready_blocks[0], &local));
        assert!(Arc::ptr_eq(&result.ready_blocks[1], &b1));
        assert!(Arc::ptr_eq(&result.ready_blocks[2], &b2));
        assert!(Arc::ptr_eq(&result.ready_blocks[3], &b3));
        assert_eq!(result.missing, 0);
        assert_eq!(result.cache_inserts.len(), 3);
    }

    #[test]
    fn ready_result_stops_at_first_missing_prefetch_key() {
        let k1 = key(1);
        let k2 = key(2);
        let k3 = key(3);
        let b1 = block();
        let b3 = block();

        let result = build_ready_result(
            Vec::new(),
            3,
            Some(PrefetchSource::Ssd),
            3,
            &[k1.clone(), k2, k3.clone()],
            vec![(k3, b3), (k1, Arc::clone(&b1))],
        );

        assert_eq!(result.ready_blocks.len(), 1);
        assert!(Arc::ptr_eq(&result.ready_blocks[0], &b1));
        assert_eq!(result.missing, 2);
        assert_eq!(result.cache_inserts.len(), 2);
    }

    #[test]
    fn ready_result_from_cache_uses_retained_non_resident_blocks() {
        let cache = ReadCache::new(1 << 20, false, None);
        let k1 = key(1);
        let k2 = key(2);
        let k3 = key(3);
        let b1 = block();
        let b2 = block();

        cache.batch_insert(vec![(k1.clone(), Arc::clone(&b1))]);

        let result = build_ready_result_from_cache(
            &cache,
            Vec::new(),
            3,
            Some(PrefetchSource::Rdma),
            3,
            3,
            &[k1.clone(), k2.clone(), k3],
            vec![(k2, Arc::clone(&b2))],
        );

        assert!(result.cache_inserts_handled);
        assert_eq!(result.fetched, 3);
        assert_eq!(result.ready_blocks.len(), 2);
        assert!(Arc::ptr_eq(&result.ready_blocks[0], &b1));
        assert!(Arc::ptr_eq(&result.ready_blocks[1], &b2));
        assert_eq!(result.missing, 1);
    }

    #[test]
    fn rdma_registration_uses_only_resident_keys() {
        let k1 = key(1);
        let k3 = key(3);

        let (namespace, hashes) =
            rdma_registration_from_resident_keys(Some(PrefetchSource::Rdma), &[k1, k3])
                .expect("RDMA resident keys should register");

        assert_eq!(namespace, "ns");
        assert_eq!(hashes, vec![vec![1], vec![3]]);
    }

    #[test]
    fn rdma_registration_skips_ssd_and_empty_resident_keys() {
        let k1 = key(1);

        assert!(rdma_registration_from_resident_keys(Some(PrefetchSource::Ssd), &[k1]).is_none());
        assert!(rdma_registration_from_resident_keys(Some(PrefetchSource::Rdma), &[]).is_none());
        assert!(rdma_registration_from_resident_keys(None, &[]).is_none());
    }

    #[test]
    fn prefetch_reservation_uses_shared_budget_across_sources() {
        let state = prefetch_state();

        let (rdma_reserved, rdma_reservation) =
            reserve_prefetch_slots(Arc::clone(&state), 4, 3, PrefetchSource::Rdma)
                .expect("RDMA reserve");
        assert_eq!(rdma_reserved, 3);

        let (ssd_reserved, ssd_reservation) =
            reserve_prefetch_slots(Arc::clone(&state), 4, 3, PrefetchSource::Ssd)
                .expect("SSD reserve");
        assert_eq!(ssd_reserved, 1);
        assert_eq!(state.lock().reserved_prefetch_blocks, 4);

        drop(rdma_reservation);
        assert_eq!(state.lock().reserved_prefetch_blocks, 1);
        drop(ssd_reservation);
        assert_eq!(state.lock().reserved_prefetch_blocks, 0);
    }

    #[test]
    fn prefetch_reservation_rejects_when_budget_exhausted() {
        let state = prefetch_state();
        let (_reserved, _reservation) =
            reserve_prefetch_slots(Arc::clone(&state), 2, 2, PrefetchSource::Rdma)
                .expect("initial reserve");

        assert!(reserve_prefetch_slots(state, 2, 1, PrefetchSource::Ssd).is_none());
    }
}
