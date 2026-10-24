use std::{cmp::Reverse, collections::HashMap, sync::Arc, time::Instant};

use hashlink::LruCache;
use parking_lot::Mutex;

use crate::block::{BlockKey, SealedBlock};
use crate::cache::{CacheInsertOutcome, TinyLfuCache};
use crate::metrics::{
    CACHE_CLASS_RECLAIMABLE, CACHE_CLASS_RETAINED, CACHE_RESIDENCE_REASON_CLEANUP,
    CACHE_RESIDENCE_REASON_PRESSURE, DEMOTION_BACKUP_HINT, DEMOTION_RECLAIM_HINT,
    DEMOTION_TRANSFER, core_metrics,
};

/// Why resident blocks moved from retained to reclaimable.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum DemotionReason {
    /// A peer finished copying the blocks from this node.
    TransferRelease,
    /// The MetaServer reported enough other owners at registration.
    ReclaimHint,
    /// The MetaServer reported a live decode backup for reported candidates.
    BackupHint,
}

impl DemotionReason {
    fn attributes(self) -> &'static [opentelemetry::KeyValue] {
        match self {
            Self::TransferRelease => &*DEMOTION_TRANSFER,
            Self::ReclaimHint => &*DEMOTION_RECLAIM_HINT,
            Self::BackupHint => &*DEMOTION_BACKUP_HINT,
        }
    }
}

/// Resident bytes per replacement class.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct ClassBytes {
    reclaimable: u64,
    retained: u64,
}

pub(crate) struct ReadCache {
    inner: Mutex<ReadCacheInner>,
    capacity_bytes: u64,
    /// Reclaimable share of capacity that decode backups keep ready.
    backup_reclaimable_percent: u64,
    /// Share of a class, oldest first, that tail-first eviction picks from.
    /// `0` is plain LRU.
    tail_window_percent: u64,
}

/// Backup reporting starts once resident blocks fill this share of capacity.
const BACKUP_REPORT_USAGE_PERCENT: u64 = 90;
/// Default reclaimable share of capacity kept ready by decode backups.
pub(crate) const DEFAULT_BACKUP_RECLAIMABLE_PERCENT: u64 = 10;
/// Retained entries a candidate scan may look at per candidate wanted.
const BACKUP_SCAN_FACTOR: usize = 4;
/// Default tail-first window: the oldest quarter of a class.
pub(crate) const DEFAULT_TAIL_WINDOW_PERCENT: u64 = 25;
/// Tail-first candidates cloned per block a batch still wants.
const TAIL_CANDIDATE_FACTOR: usize = 4;
/// Sort rank of a block whose chain depth was never observed (decode tails,
/// blocks of other cache groups): evicted before any block of known depth.
const UNKNOWN_DEPTH: u32 = u32::MAX;

struct ReadCacheInner {
    cache: TinyLfuCache<BlockKey, Arc<SealedBlock>>,
    reclaimable: LruCache<BlockKey, ResidentMetadata>,
    retained: LruCache<BlockKey, ResidentMetadata>,
    next_generation: u64,
    class_bytes: ClassBytes,
    /// Pressure eviction has run since the last full cleanup. Allocator
    /// fragmentation starts eviction well below full occupancy, so this, not
    /// resident bytes alone, marks a cache that is effectively full.
    pressured: bool,
    links: ChainLinks,
}

/// Dependencies between resident blocks, learned from queries: a block's
/// parent is the block it extends (the previous block of a prefix query) or,
/// for a cache group > 0, the group 0 block at the same position. A resume
/// point needs a block and every ancestor, so tail-first eviction only
/// evicts blocks without resident children.
///
/// The server knows group ids, not group kinds. A group block is therefore
/// a leaf under its position: it may leave alone (a window group's block of
/// an early position is never needed again), but never outlives its
/// position, which would leave it as weight no resume point can use.
///
/// Links exist only while both ends are resident; evicting either end drops
/// them.
#[derive(Default)]
struct ChainLinks {
    parent: HashMap<BlockKey, BlockKey>,
    children: HashMap<BlockKey, Vec<BlockKey>>,
}

impl ChainLinks {
    fn link(&mut self, parent: &BlockKey, child: &BlockKey) {
        if parent == child || self.parent.get(child) == Some(parent) {
            return;
        }
        if let Some(old) = self.parent.insert(child.clone(), parent.clone()) {
            self.drop_child(&old, child);
        }
        self.children
            .entry(parent.clone())
            .or_default()
            .push(child.clone());
    }

    fn drop_child(&mut self, parent: &BlockKey, child: &BlockKey) {
        if let Some(children) = self.children.get_mut(parent) {
            children.retain(|c| c != child);
            if children.is_empty() {
                self.children.remove(parent);
            }
        }
    }

    /// Forget every link of a block leaving the cache. Returns whether it
    /// still had resident children.
    fn unlink(&mut self, key: &BlockKey) -> bool {
        if let Some(parent) = self.parent.remove(key) {
            self.drop_child(&parent, key);
        }
        let Some(children) = self.children.remove(key) else {
            return false;
        };
        for child in &children {
            self.parent.remove(child);
        }
        !children.is_empty()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ResidentMetadata {
    inserted_at: Instant,
    generation: u64,
    /// Footprint at insertion, used for per-class byte accounting.
    bytes: u64,
    /// Index in the latest prefix query whose hit prefix covered the block.
    /// Relative to that query's first block, so it orders blocks within a
    /// chain; [`UNKNOWN_DEPTH`] until a query covers it.
    depth: u32,
}

struct RemovedResident {
    key: BlockKey,
    block: Arc<SealedBlock>,
    /// Insertion time taken from the block's replacement-class metadata.
    inserted_at: Instant,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ResidentClass {
    Reclaimable,
    Retained,
}

impl ReadCache {
    pub(crate) fn new(
        capacity_bytes: usize,
        enable_lfu_admission: bool,
        value_size_hint: Option<usize>,
    ) -> Self {
        let cache =
            TinyLfuCache::new_unbounded(capacity_bytes, enable_lfu_admission, value_size_hint);
        Self {
            inner: Mutex::new(ReadCacheInner {
                cache,
                reclaimable: LruCache::new_unbounded(),
                retained: LruCache::new_unbounded(),
                next_generation: 0,
                class_bytes: ClassBytes::default(),
                pressured: false,
                links: ChainLinks::default(),
            }),
            capacity_bytes: capacity_bytes as u64,
            backup_reclaimable_percent: DEFAULT_BACKUP_RECLAIMABLE_PERCENT,
            tail_window_percent: DEFAULT_TAIL_WINDOW_PERCENT,
        }
    }

    pub(crate) fn with_backup_reclaimable_percent(mut self, percent: u64) -> Self {
        self.backup_reclaimable_percent = percent.min(100);
        self
    }

    pub(crate) fn with_tail_window_percent(mut self, percent: u64) -> Self {
        self.tail_window_percent = percent.min(100);
        self
    }

    pub(super) fn resident_blocks(&self) -> usize {
        let inner = self.inner.lock();
        inner.reclaimable.len() + inner.retained.len()
    }

    pub(super) fn contains_keys(&self, keys: &[BlockKey]) -> Vec<bool> {
        let inner = self.inner.lock();
        keys.iter().map(|k| inner.cache.contains_key(k)).collect()
    }

    /// Position-aligned resident blocks without touching recency. A held
    /// `Arc` keeps the block out of pressure eviction until it is dropped.
    pub(super) fn peek_blocks(&self, keys: &[BlockKey]) -> Vec<Option<Arc<SealedBlock>>> {
        let inner = self.inner.lock();
        keys.iter().map(|k| inner.cache.peek(k)).collect()
    }

    /// Scan cache for a prefix of `keys`, stopping at the first miss.
    ///
    /// `chain` marks `keys` as one prefix chain in order (an attention prefix
    /// query): the hit prefix then records each block's depth and its parent
    /// link for tail-first eviction. Set lists (membership fetches) pass false.
    pub(super) fn get_prefix_blocks(
        &self,
        keys: &[BlockKey],
        chain: bool,
    ) -> (usize, Vec<Arc<SealedBlock>>) {
        let mut hit = 0usize;
        let mut blocks = Vec::with_capacity(keys.len());
        {
            let mut inner = self.inner.lock();
            for (i, key) in keys.iter().enumerate() {
                if let Some(block) = inner.cache.get(key) {
                    refresh_recency(&mut inner, key);
                    if chain {
                        set_depth(&mut inner, key, i);
                        if i > 0 {
                            inner.links.link(&keys[i - 1], key);
                        }
                    }
                    hit += 1;
                    blocks.push(block);
                } else {
                    break;
                }
            }
        }
        (hit, blocks)
    }

    /// Record each resident `(position, member)` pair: `member` is a cache
    /// group > 0 block at the chain position of the group 0 block `position`.
    /// The member takes the position's depth and is evicted no later than it.
    /// Pairs with a non-resident end are ignored.
    pub(super) fn link_group_members(&self, pairs: &[(BlockKey, BlockKey)]) {
        let mut inner = self.inner.lock();
        for (position, member) in pairs {
            if !inner.cache.contains_key(position) || !inner.cache.contains_key(member) {
                continue;
            }
            inner.links.link(position, member);
            let depth = depth_of(&inner, position);
            if let Some(metadata) = resident_metadata_mut(&mut inner, member) {
                metadata.depth = depth;
            }
        }
    }

    pub(super) fn batch_insert(&self, blocks: Vec<(BlockKey, Arc<SealedBlock>)>) {
        let mut inner = self.inner.lock();
        for (key, block) in blocks {
            insert_block(&mut inner, key, block, ResidentClass::Retained);
        }
    }

    #[cfg(test)]
    fn batch_insert_resident_keys(
        &self,
        blocks: Vec<(BlockKey, Arc<SealedBlock>)>,
    ) -> Vec<BlockKey> {
        let mut inner = self.inner.lock();
        let mut resident_keys = Vec::new();
        for (key, block) in blocks {
            match insert_block(&mut inner, key.clone(), block, ResidentClass::Reclaimable) {
                CacheInsertOutcome::InsertedNew | CacheInsertOutcome::AlreadyExists => {
                    resident_keys.push(key);
                }
                CacheInsertOutcome::Rejected => {}
            }
        }
        resident_keys
    }

    pub(super) fn batch_insert_refs(
        &self,
        blocks: &[(BlockKey, Arc<SealedBlock>)],
    ) -> Vec<BlockKey> {
        let mut inner = self.inner.lock();
        let mut resident_keys = Vec::new();
        for (key, block) in blocks {
            let outcome = insert_block(
                &mut inner,
                key.clone(),
                Arc::clone(block),
                ResidentClass::Retained,
            );
            if matches!(
                outcome,
                CacheInsertOutcome::InsertedNew | CacheInsertOutcome::AlreadyExists
            ) {
                resident_keys.push(key.clone());
            }
        }
        resident_keys
    }

    pub(super) fn resident_generations(&self, keys: &[BlockKey]) -> Vec<(BlockKey, u64)> {
        let inner = self.inner.lock();
        keys.iter()
            .filter_map(|key| {
                let metadata = inner
                    .reclaimable
                    .peek(key)
                    .or_else(|| inner.retained.peek(key))?;
                Some((key.clone(), metadata.generation))
            })
            .collect()
    }

    /// Look up specific blocks by key without prefix-scan semantics (does not
    /// stop at first miss). Used by the serving side of cross-node transfer.
    pub(super) fn get_blocks(&self, keys: &[BlockKey]) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let mut inner = self.inner.lock();
        let mut found = Vec::new();
        for key in keys {
            if let Some(block) = inner.cache.get(key) {
                refresh_recency(&mut inner, key);
                found.push((key.clone(), block));
            }
        }
        found
    }

    /// Position-aligned membership: entry `i` is the block for `keys[i]`, or
    /// `None` on miss. Unlike [`Self::get_prefix_blocks`] this never stops at
    /// the first gap — hybrid-cache checkpoint groups (recurrent state) have
    /// sparse hit patterns by design, where the caller picks the rightmost
    /// hit instead of a prefix.
    pub(super) fn get_blocks_aligned(&self, keys: &[BlockKey]) -> Vec<Option<Arc<SealedBlock>>> {
        let mut inner = self.inner.lock();
        keys.iter()
            .map(|key| {
                inner.cache.get(key).inspect(|_| {
                    refresh_recency(&mut inner, key);
                })
            })
            .collect()
    }

    /// Evict up to `batch_size` cache-owned blocks, reclaimable class first.
    ///
    /// With a tail window, each class evicts tail-first: among its oldest
    /// `tail_window_percent` blocks, the deepest chain position goes first,
    /// and a block goes only after its resident descendants. Otherwise plain
    /// LRU.
    pub(super) fn remove_lru_batch(&self, batch_size: usize) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let removed = {
            let mut inner = self.inner.lock();
            let mut removed = Vec::with_capacity(batch_size);
            for class in [ResidentClass::Reclaimable, ResidentClass::Retained] {
                if removed.len() == batch_size {
                    break;
                }
                if self.tail_window_percent == 0 {
                    remove_lru_batch_from_class(&mut inner, class, batch_size, &mut removed);
                } else {
                    remove_tail_first_from_class(
                        &mut inner,
                        class,
                        batch_size,
                        self.tail_window_percent,
                        &mut removed,
                    );
                }
            }
            inner.pressured |= !removed.is_empty();
            removed
        };
        record_residence_durations(removed, &*CACHE_RESIDENCE_REASON_PRESSURE)
    }

    pub(super) fn remove_all(&self) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let removed = {
            let mut inner = self.inner.lock();
            let reclaimable_blocks = inner.reclaimable.len() as i64;
            let retained_blocks = inner.retained.len() as i64;
            let mut metadata = HashMap::with_capacity(
                inner.reclaimable.len().saturating_add(inner.retained.len()),
            );
            metadata.extend(inner.reclaimable.drain());
            metadata.extend(inner.retained.drain());
            inner.links = ChainLinks::default();
            let removed = inner
                .cache
                .remove_all()
                .into_iter()
                .map(|(key, block)| {
                    let inserted_at = metadata.remove(&key).map(|entry| entry.inserted_at);
                    debug_assert!(
                        inserted_at.is_some(),
                        "resident block is missing its replacement metadata"
                    );
                    RemovedResident {
                        inserted_at: inserted_at.unwrap_or_else(Instant::now),
                        key,
                        block,
                    }
                })
                .collect::<Vec<_>>();
            debug_assert_eq!(
                removed.len() as i64,
                reclaimable_blocks + retained_blocks,
                "resident cache and replacement classes diverged"
            );
            debug_assert!(
                metadata.is_empty(),
                "replacement metadata outlives its resident block"
            );
            let metrics = core_metrics();
            metrics
                .cache_resident_blocks
                .add(-reclaimable_blocks, &*CACHE_CLASS_RECLAIMABLE);
            metrics
                .cache_resident_blocks
                .add(-retained_blocks, &*CACHE_CLASS_RETAINED);
            let bytes = std::mem::take(&mut inner.class_bytes);
            inner.pressured = false;
            metrics
                .cache_resident_bytes_by_class
                .add(-(bytes.reclaimable as i64), &*CACHE_CLASS_RECLAIMABLE);
            metrics
                .cache_resident_bytes_by_class
                .add(-(bytes.retained as i64), &*CACHE_CLASS_RETAINED);
            removed
        };
        record_residence_durations(removed, &*CACHE_RESIDENCE_REASON_CLEANUP)
    }

    /// Returns the number of blocks moved.
    pub(crate) fn mark_reclaimable_hashes(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
        reason: DemotionReason,
    ) -> usize {
        if hashes.is_empty() {
            return 0;
        }

        let keys: Vec<BlockKey> = hashes
            .iter()
            .map(|hash| BlockKey::new(namespace.to_string(), hash.clone()))
            .collect();
        self.mark_reclaimable_keys(&keys, reason)
    }

    pub(crate) fn mark_reclaimable_hashes_if_generation(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
        generations: &HashMap<Vec<u8>, u64>,
    ) {
        if hashes.is_empty() {
            return;
        }

        let mut inner = self.inner.lock();
        let mut moved = Demoted::default();
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            if let Some(&generation) = generations.get(hash) {
                moved.add(mark_reclaimable_with_generation(
                    &mut inner,
                    &key,
                    Some(generation),
                ));
            }
        }
        moved.record(DemotionReason::ReclaimHint);
    }

    /// Move resident blocks to the reclaimable replacement class.
    ///
    /// The caller supplies complete keys so serving-side paths can classify
    /// the exact blocks they exposed without reconstructing namespace/hash
    /// pairs. Missing blocks and blocks already in the reclaimable class are
    /// ignored. Returns the number of blocks moved.
    pub(crate) fn mark_reclaimable_keys(&self, keys: &[BlockKey], reason: DemotionReason) -> usize {
        if keys.is_empty() {
            return 0;
        }

        let mut inner = self.inner.lock();
        let mut moved = Demoted::default();
        for key in keys {
            moved.add(mark_reclaimable_with_generation(&mut inner, key, None));
        }
        moved.record(reason)
    }

    /// Bytes of retained blocks to offer for decode backup right now.
    ///
    /// Zero until the cache is full: pressure eviction has run, or resident
    /// blocks reach 90% of capacity. Then the gap between the reclaimable
    /// class and its watermark, so a node only asks for as much backup as
    /// it needs to keep cheap eviction victims ready.
    pub(crate) fn backup_report_budget(&self) -> u64 {
        let inner = self.inner.lock();
        backup_report_budget(
            self.capacity_bytes,
            self.backup_reclaimable_percent,
            inner.class_bytes,
            inner.pressured,
        )
    }

    /// Oldest retained blocks, up to `max_bytes` / `max_blocks`, that only
    /// the cache holds.
    ///
    /// These are the next retained blocks pressure eviction would take, so a
    /// decode node copying them lets this node drop them first. Blocks pinned
    /// by an in-flight load or transfer are skipped. Does not touch recency.
    /// Looks at no more than `BACKUP_SCAN_FACTOR * max_blocks` entries, so a
    /// tail full of pinned blocks cannot hold the lock for a scan of millions.
    pub(crate) fn backup_candidates(
        &self,
        max_bytes: u64,
        max_blocks: usize,
    ) -> Vec<(BlockKey, u64)> {
        let inner = self.inner.lock();
        let mut candidates = Vec::new();
        let mut total = 0u64;
        let scan = max_blocks.saturating_mul(BACKUP_SCAN_FACTOR);
        for (key, metadata) in inner.retained.iter().take(scan) {
            if total >= max_bytes || candidates.len() >= max_blocks {
                break;
            }
            if !inner.cache.is_cache_owned_only(key) {
                continue;
            }
            total = total.saturating_add(metadata.bytes);
            candidates.push((key.clone(), metadata.bytes));
        }
        candidates
    }

    #[cfg(test)]
    pub(crate) fn insert_retained_for_test(&self, key: BlockKey, block: Arc<SealedBlock>) {
        let mut inner = self.inner.lock();
        insert_block(&mut inner, key, block, ResidentClass::Retained);
    }

    #[cfg(test)]
    pub(crate) fn resident_generation_for_test(&self, key: &BlockKey) -> u64 {
        self.resident_generations(std::slice::from_ref(key))[0].1
    }

    #[cfg(test)]
    pub(crate) fn remove_lru_batch_for_test(&self, batch_size: usize) {
        drop(self.remove_lru_batch(batch_size));
    }

    #[cfg(test)]
    fn class_bytes(&self) -> ClassBytes {
        self.inner.lock().class_bytes
    }

    #[cfg(test)]
    pub(crate) fn is_reclaimable_for_test(&self, key: &BlockKey) -> bool {
        self.inner.lock().reclaimable.contains_key(key)
    }
}

impl ResidentClass {
    fn attributes(self) -> &'static [opentelemetry::KeyValue] {
        match self {
            Self::Reclaimable => &*CACHE_CLASS_RECLAIMABLE,
            Self::Retained => &*CACHE_CLASS_RETAINED,
        }
    }
}

fn insert_block(
    inner: &mut ReadCacheInner,
    key: BlockKey,
    block: Arc<SealedBlock>,
    class: ResidentClass,
) -> CacheInsertOutcome {
    let footprint_bytes = block.memory_footprint();
    let outcome = inner.cache.insert(key.clone(), block);
    match outcome {
        CacheInsertOutcome::InsertedNew => {
            inner.next_generation = inner.next_generation.wrapping_add(1);
            let generation = inner.next_generation;
            class_lru(inner, class).insert(
                key,
                ResidentMetadata {
                    inserted_at: Instant::now(),
                    generation,
                    bytes: footprint_bytes,
                    depth: UNKNOWN_DEPTH,
                },
            );
            *class_bytes(inner, class) += footprint_bytes;
            let m = core_metrics();
            m.cache_block_insertions.add(1, &[]);
            m.cache_resident_bytes.add(footprint_bytes as i64, &[]);
            m.cache_resident_blocks.add(1, class.attributes());
            m.cache_resident_bytes_by_class
                .add(footprint_bytes as i64, class.attributes());
        }
        CacheInsertOutcome::AlreadyExists => refresh_recency(inner, &key),
        CacheInsertOutcome::Rejected => {
            core_metrics().cache_block_admission_rejections.add(1, &[]);
        }
    }
    outcome
}

fn class_lru(
    inner: &mut ReadCacheInner,
    class: ResidentClass,
) -> &mut LruCache<BlockKey, ResidentMetadata> {
    match class {
        ResidentClass::Reclaimable => &mut inner.reclaimable,
        ResidentClass::Retained => &mut inner.retained,
    }
}

fn class_lru_ref(
    inner: &ReadCacheInner,
    class: ResidentClass,
) -> &LruCache<BlockKey, ResidentMetadata> {
    match class {
        ResidentClass::Reclaimable => &inner.reclaimable,
        ResidentClass::Retained => &inner.retained,
    }
}

fn backup_report_budget(
    capacity_bytes: u64,
    reclaimable_percent: u64,
    bytes: ClassBytes,
    pressured: bool,
) -> u64 {
    let resident = bytes.reclaimable.saturating_add(bytes.retained);
    if !pressured
        && resident.saturating_mul(100) < capacity_bytes.saturating_mul(BACKUP_REPORT_USAGE_PERCENT)
    {
        return 0;
    }
    (capacity_bytes / 100 * reclaimable_percent).saturating_sub(bytes.reclaimable)
}

fn class_bytes(inner: &mut ReadCacheInner, class: ResidentClass) -> &mut u64 {
    match class {
        ResidentClass::Reclaimable => &mut inner.class_bytes.reclaimable,
        ResidentClass::Retained => &mut inner.class_bytes.retained,
    }
}

fn resident_metadata_mut<'a>(
    inner: &'a mut ReadCacheInner,
    key: &BlockKey,
) -> Option<&'a mut ResidentMetadata> {
    if inner.reclaimable.contains_key(key) {
        inner.reclaimable.peek_mut(key)
    } else {
        inner.retained.peek_mut(key)
    }
}

fn depth_of(inner: &ReadCacheInner, key: &BlockKey) -> u32 {
    inner
        .reclaimable
        .peek(key)
        .or_else(|| inner.retained.peek(key))
        .map_or(UNKNOWN_DEPTH, |metadata| metadata.depth)
}

fn set_depth(inner: &mut ReadCacheInner, key: &BlockKey, index: usize) {
    // UNKNOWN_DEPTH is reserved; a prefix of 4 billion blocks does not occur.
    let depth = u32::try_from(index).unwrap_or(UNKNOWN_DEPTH - 1);
    if let Some(metadata) = resident_metadata_mut(inner, key) {
        metadata.depth = depth.min(UNKNOWN_DEPTH - 1);
    }
}

fn refresh_recency(inner: &mut ReadCacheInner, key: &BlockKey) {
    let classified = inner.reclaimable.get(key).is_some() || inner.retained.get(key).is_some();
    debug_assert!(
        classified || !inner.cache.contains_key(key),
        "resident block is missing its replacement class"
    );
}

/// Blocks and bytes moved from retained to reclaimable in one batch.
#[derive(Default)]
struct Demoted {
    blocks: i64,
    bytes: i64,
}

impl Demoted {
    fn add(&mut self, bytes: Option<u64>) {
        if let Some(bytes) = bytes {
            self.blocks += 1;
            self.bytes += bytes as i64;
        }
    }

    /// Update the class metrics once per batch; returns the block count.
    fn record(self, reason: DemotionReason) -> usize {
        if self.blocks > 0 {
            let m = core_metrics();
            m.cache_class_demotions
                .add(self.blocks as u64, reason.attributes());
            m.cache_resident_blocks
                .add(-self.blocks, &*CACHE_CLASS_RETAINED);
            m.cache_resident_blocks
                .add(self.blocks, &*CACHE_CLASS_RECLAIMABLE);
            m.cache_resident_bytes_by_class
                .add(-self.bytes, &*CACHE_CLASS_RETAINED);
            m.cache_resident_bytes_by_class
                .add(self.bytes, &*CACHE_CLASS_RECLAIMABLE);
        }
        self.blocks as usize
    }
}

/// Returns the footprint moved, or `None` if the block stayed where it was.
fn mark_reclaimable_with_generation(
    inner: &mut ReadCacheInner,
    key: &BlockKey,
    expected_generation: Option<u64>,
) -> Option<u64> {
    if !inner.cache.contains_key(key) {
        return None;
    }
    if let Some(expected_generation) = expected_generation {
        match inner.retained.peek(key) {
            Some(metadata) if metadata.generation == expected_generation => {}
            _ => return None,
        }
    }
    if let Some(metadata) = inner.retained.remove(key) {
        inner.class_bytes.retained -= metadata.bytes;
        inner.class_bytes.reclaimable += metadata.bytes;
        inner.reclaimable.insert(key.clone(), metadata);
        Some(metadata.bytes)
    } else {
        debug_assert!(
            inner.reclaimable.contains_key(key),
            "resident block is missing its replacement class"
        );
        None
    }
}

fn remove_lru(inner: &mut ReadCacheInner, class: ResidentClass) -> Option<RemovedResident> {
    while let Some((key, metadata)) = class_lru(inner, class).remove_lru() {
        if let Some(removed) = finish_eviction(inner, class, key, metadata) {
            return Some(removed);
        }
    }
    None
}

/// Evict one resident block from whichever class holds it.
fn evict_resident(inner: &mut ReadCacheInner, key: &BlockKey) -> Option<RemovedResident> {
    let (class, metadata) = if let Some(metadata) = inner.reclaimable.remove(key) {
        (ResidentClass::Reclaimable, metadata)
    } else {
        (ResidentClass::Retained, inner.retained.remove(key)?)
    };
    finish_eviction(inner, class, key.clone(), metadata)
}

/// Drop a block whose class entry was just removed: the resident block, its
/// chain links, and the class accounting.
fn finish_eviction(
    inner: &mut ReadCacheInner,
    class: ResidentClass,
    key: BlockKey,
    metadata: ResidentMetadata,
) -> Option<RemovedResident> {
    let block = inner.cache.remove(&key);
    debug_assert!(
        block.is_some(),
        "replacement class contains a non-resident block"
    );
    let block = block?;
    let had_children = inner.links.unlink(&key);
    *class_bytes(inner, class) -= metadata.bytes;
    let metrics = core_metrics();
    metrics.cache_resident_blocks.add(-1, class.attributes());
    metrics
        .cache_resident_bytes_by_class
        .add(-(metadata.bytes as i64), class.attributes());
    metrics
        .cache_block_evictions_by_class
        .add(1, class.attributes());
    if had_children {
        metrics.cache_evictions_with_resident_children.add(1, &[]);
    }
    Some(RemovedResident {
        key,
        block,
        inserted_at: metadata.inserted_at,
    })
}

/// The resident child of `key` to evict before it: the deepest, unknown depth
/// first.
fn deepest_child(inner: &ReadCacheInner, key: &BlockKey) -> Option<BlockKey> {
    inner
        .links
        .children
        .get(key)?
        .iter()
        .max_by_key(|child| depth_of(inner, child))
        .cloned()
}

/// Tail-first eviction within one class.
///
/// Candidates are the oldest `window_percent` of the class (at least the
/// blocks still wanted), deepest chain position first, oldest first among
/// equals. Taking a candidate evicts its resident subtree leaves first, so no
/// block is evicted while a block that depends on it stays resident; such
/// descendants may sit in the other class. A subtree whose next leaf is
/// pinned by a load or transfer is skipped. If pins exhaust the candidates,
/// the rest of the class is tried in LRU order under the same rule.
fn remove_tail_first_from_class(
    inner: &mut ReadCacheInner,
    class: ResidentClass,
    batch_size: usize,
    window_percent: u64,
    removed: &mut Vec<RemovedResident>,
) {
    let len = class_lru_ref(inner, class).len();
    let wanted = batch_size.saturating_sub(removed.len());
    if len == 0 || wanted == 0 {
        return;
    }
    let share = (len as u128 * u128::from(window_percent) / 100) as usize;
    let window = share.max(wanted).min(len);
    // Clone only the keys the batch can use: a subtree yields at least one
    // victim unless pinned, so a few per wanted block is plenty.
    let take = wanted.saturating_mul(TAIL_CANDIDATE_FACTOR);
    let candidates: Vec<BlockKey> = {
        // Deepest first; the LRU index breaks ties oldest first.
        let mut window_keys: Vec<(Reverse<u32>, usize, &BlockKey)> = class_lru_ref(inner, class)
            .iter()
            .take(window)
            .enumerate()
            .map(|(index, (key, metadata))| (Reverse(metadata.depth), index, key))
            .collect();
        if take < window_keys.len() {
            window_keys.select_nth_unstable_by_key(take, |(depth, index, _)| (*depth, *index));
            window_keys.truncate(take);
        }
        window_keys.sort_unstable_by_key(|(depth, index, _)| (*depth, *index));
        window_keys
            .into_iter()
            .map(|(_, _, key)| key.clone())
            .collect()
    };
    evict_candidates(inner, candidates, batch_size, removed);
    if removed.len() < batch_size {
        // Pins cover the window: fall back to the class's leaves in LRU
        // order. Leaves need no walk, so a fully pinned class costs one scan.
        let leaves: Vec<BlockKey> = class_lru_ref(inner, class)
            .iter()
            .filter(|(key, _)| !inner.links.children.contains_key(*key))
            .map(|(key, _)| key.clone())
            .collect();
        evict_candidates(inner, leaves, batch_size, removed);
    }
}

fn evict_candidates(
    inner: &mut ReadCacheInner,
    candidates: Vec<BlockKey>,
    batch_size: usize,
    removed: &mut Vec<RemovedResident>,
) {
    for candidate in candidates {
        if removed.len() == batch_size {
            return;
        }
        if !inner.cache.contains_key(&candidate) {
            continue;
        }
        evict_subtree(inner, &candidate, batch_size, removed);
    }
}

/// Evict `root`'s resident descendants, leaves first, then `root`, until the
/// batch is full.
fn evict_subtree(
    inner: &mut ReadCacheInner,
    root: &BlockKey,
    batch_size: usize,
    removed: &mut Vec<RemovedResident>,
) {
    let mut current = root.clone();
    while removed.len() < batch_size && inner.cache.contains_key(root) {
        // Links come from client queries; a malformed client could link a
        // cycle. Bound the walk and evict where it stops, which drops the
        // block's links and so breaks the cycle.
        let mut steps = inner.links.parent.len();
        while steps > 0
            && let Some(child) = deepest_child(inner, &current)
        {
            steps -= 1;
            current = child;
        }
        if !inner.cache.is_cache_owned_only(&current) {
            return;
        }
        let parent = inner.links.parent.get(&current).cloned();
        let reached_root = current == *root;
        let Some(evicted) = evict_resident(inner, &current) else {
            return;
        };
        removed.push(evicted);
        if reached_root {
            return;
        }
        current = parent.unwrap_or_else(|| root.clone());
    }
}

fn remove_lru_batch_from_class(
    inner: &mut ReadCacheInner,
    class: ResidentClass,
    batch_size: usize,
    removed: &mut Vec<RemovedResident>,
) {
    let candidates = class_lru(inner, class).len();
    for _ in 0..candidates {
        if removed.len() == batch_size {
            break;
        }

        let Some(key) = class_lru(inner, class)
            .iter()
            .next()
            .map(|(key, _)| key.clone())
        else {
            break;
        };
        if inner.cache.is_cache_owned_only(&key) {
            let block = remove_lru(inner, class)
                .expect("cache-owned LRU candidate must remain resident while locked");
            removed.push(block);
        } else {
            class_lru(inner, class).get(&key);
        }
    }
}

fn record_residence_durations(
    removed: Vec<RemovedResident>,
    attributes: &[opentelemetry::KeyValue],
) -> Vec<(BlockKey, Arc<SealedBlock>)> {
    let removed_at = Instant::now();
    let metrics = core_metrics();
    removed
        .into_iter()
        .map(|entry| {
            metrics.cache_residence_duration.record(
                residence_duration_seconds(entry.inserted_at, removed_at),
                attributes,
            );
            (entry.key, entry.block)
        })
        .collect()
}

fn residence_duration_seconds(inserted_at: Instant, removed_at: Instant) -> f64 {
    removed_at
        .saturating_duration_since(inserted_at)
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn make_cache() -> ReadCache {
        ReadCache::new(1 << 20, false, None)
    }

    fn make_block() -> Arc<SealedBlock> {
        Arc::new(SealedBlock::from_slots(Vec::new()))
    }

    fn assert_class(cache: &ReadCache, key: &BlockKey, expected: ResidentClass) {
        let inner = cache.inner.lock();
        assert!(inner.cache.contains_key(key));
        assert_eq!(
            inner.reclaimable.contains_key(key),
            expected == ResidentClass::Reclaimable
        );
        assert_eq!(
            inner.retained.contains_key(key),
            expected == ResidentClass::Retained
        );
    }

    fn resident_metadata(cache: &ReadCache, key: &BlockKey) -> Option<ResidentMetadata> {
        let inner = cache.inner.lock();
        inner
            .reclaimable
            .peek(key)
            .or_else(|| inner.retained.peek(key))
            .copied()
    }

    fn backdate_resident(cache: &ReadCache, key: &BlockKey, age: Duration) -> Instant {
        let inserted_at = Instant::now() - age;
        let mut inner = cache.inner.lock();
        let metadata = if let Some(metadata) = inner.reclaimable.peek_mut(key) {
            metadata
        } else {
            inner
                .retained
                .peek_mut(key)
                .expect("test resident must have replacement metadata")
        };
        metadata.inserted_at = inserted_at;
        inserted_at
    }

    #[tokio::test]
    async fn backup_candidates_follow_retained_tail_and_watermark() {
        use crate::block::{RawBlock, Segment};
        use crate::storage::{StorageConfig, StorageEngine};
        use std::num::NonZeroU64;

        const BLOCK: u64 = 100;
        let engine =
            StorageEngine::new_with_config(1 << 20, false, StorageConfig::default(), &[]).unwrap();
        let sized_block = || {
            let alloc = engine
                .allocate(NonZeroU64::new(BLOCK).unwrap(), None)
                .expect("test pool should have space");
            let ptr = alloc.as_non_null();
            Arc::new(SealedBlock::from_slots(vec![(
                RawBlock::new(vec![Segment::new(ptr, BLOCK as usize, alloc)]),
                pegaflow_common::NumaNode::UNKNOWN,
            )]))
        };
        // Ten 100-byte blocks fill a 1000-byte cache.
        let cache = ReadCache::new(1000, false, None);
        let keys: Vec<BlockKey> = (0..10u8)
            .map(|i| BlockKey::new("ns".into(), vec![i]))
            .collect();
        let pinned = sized_block();
        for (i, key) in keys.iter().enumerate() {
            let block = if i == 1 {
                Arc::clone(&pinned)
            } else {
                sized_block()
            };
            cache.insert_retained_for_test(key.clone(), block);
        }
        assert_eq!(
            cache.class_bytes(),
            ClassBytes {
                reclaimable: 0,
                retained: 1000
            }
        );

        // Full and nothing reclaimable: ask for the 10% watermark.
        assert_eq!(cache.backup_report_budget(), 100);
        // Oldest first; the block pinned by a load or transfer is skipped.
        let hashes = |candidates: Vec<(BlockKey, u64)>| -> Vec<Vec<u8>> {
            candidates.into_iter().map(|(key, _)| key.hash).collect()
        };
        assert_eq!(
            hashes(cache.backup_candidates(250, usize::MAX)),
            [vec![0], vec![2], vec![3]]
        );
        assert_eq!(
            hashes(cache.backup_candidates(u64::MAX, 2)),
            [vec![0], vec![2]]
        );

        cache.mark_reclaimable_keys(&keys[..1], DemotionReason::TransferRelease);
        assert_eq!(
            cache.class_bytes(),
            ClassBytes {
                reclaimable: 100,
                retained: 900
            }
        );
        // Watermark met: nothing to report.
        assert_eq!(cache.backup_report_budget(), 0);

        drop(pinned);
        cache.remove_lru_batch_for_test(2);
        assert_eq!(
            cache.class_bytes(),
            ClassBytes {
                reclaimable: 0,
                retained: 900 - BLOCK
            }
        );
        // Pressure eviction ran, so the cache counts as full below 90%.
        assert_eq!(cache.backup_report_budget(), 100);
        drop(cache.remove_all());
        assert_eq!(cache.class_bytes(), ClassBytes::default());
        assert_eq!(cache.backup_report_budget(), 0);
    }

    #[test]
    fn new_blocks_are_classified_by_source() {
        let cache = make_cache();
        let local = BlockKey::new("ns".into(), vec![1]);
        let ssd = BlockKey::new("ns".into(), vec![2]);
        let remote = BlockKey::new("ns".into(), vec![3]);
        let local_block = make_block();

        cache.batch_insert_refs(&[(local.clone(), local_block)]);
        cache.batch_insert(vec![(ssd.clone(), make_block())]);
        cache.batch_insert_refs(&[(remote.clone(), make_block())]);

        assert_class(&cache, &local, ResidentClass::Retained);
        assert_class(&cache, &ssd, ResidentClass::Retained);
        assert_class(&cache, &remote, ResidentClass::Retained);
    }

    #[test]
    fn reclaimable_blocks_are_evicted_before_retained_blocks() {
        let cache = make_cache();
        let retained = BlockKey::new("ns".into(), vec![1]);
        let reclaimable = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert(vec![(retained.clone(), make_block())]);
        cache.batch_insert_resident_keys(vec![(reclaimable.clone(), make_block())]);

        let evicted = cache.remove_lru_batch(2);
        assert_eq!(
            evicted.into_iter().map(|(key, _)| key).collect::<Vec<_>>(),
            vec![reclaimable, retained]
        );
    }

    #[test]
    fn peeked_blocks_stay_out_of_pressure_eviction_while_held() {
        // A backup target re-registers blocks it already holds; holding the
        // peeked Arc until that insert is queued keeps eviction from queuing a
        // removal ahead of it.
        let cache = make_cache();
        let held = BlockKey::new("ns".into(), vec![1]);
        let other = BlockKey::new("ns".into(), vec![2]);
        let missing = BlockKey::new("ns".into(), vec![3]);
        cache.batch_insert(vec![
            (held.clone(), make_block()),
            (other.clone(), make_block()),
        ]);

        let peeked = cache.peek_blocks(&[held.clone(), missing]);
        assert!(peeked[0].is_some() && peeked[1].is_none());
        // `held` is the LRU victim, but the reference keeps it resident.
        let evicted = cache.remove_lru_batch(2);
        assert_eq!(
            evicted.into_iter().map(|(key, _)| key).collect::<Vec<_>>(),
            vec![other]
        );
        drop(peeked);
        assert_eq!(cache.remove_lru_batch(1)[0].0, held);

        // Peeking does not refresh recency.
        let first = BlockKey::new("ns".into(), vec![4]);
        let second = BlockKey::new("ns".into(), vec![5]);
        cache.batch_insert(vec![
            (first.clone(), make_block()),
            (second.clone(), make_block()),
        ]);
        drop(cache.peek_blocks(std::slice::from_ref(&first)));
        assert_eq!(cache.remove_lru_batch(1)[0].0, first);
    }

    #[test]
    fn stale_reclaimable_hint_does_not_demote_reinserted_generation() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.insert_retained_for_test(key.clone(), make_block());
        let old_generation = cache.resident_generations(std::slice::from_ref(&key))[0].1;

        drop(cache.remove_lru_batch(1));
        cache.insert_retained_for_test(key.clone(), make_block());
        let new_generation = cache.resident_generations(std::slice::from_ref(&key))[0].1;
        assert_ne!(old_generation, new_generation);

        let generations = HashMap::from([(key.hash.clone(), old_generation)]);
        cache.mark_reclaimable_hashes_if_generation(
            "ns",
            std::slice::from_ref(&key.hash),
            &generations,
        );

        assert_class(&cache, &key, ResidentClass::Retained);
    }

    #[test]
    fn pressure_reclaim_ignores_weak_references() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = make_block();
        let weak = Arc::downgrade(&block);
        cache.batch_insert(vec![(key.clone(), block)]);

        assert_eq!(cache.remove_lru_batch(1)[0].0, key);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn pressure_reclaim_waits_for_external_strong_reference() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = make_block();
        let external = Arc::clone(&block);
        let weak = Arc::downgrade(&block);
        cache.batch_insert(vec![(key.clone(), block)]);

        assert!(cache.remove_lru_batch(1).is_empty());
        assert_class(&cache, &key, ResidentClass::Retained);

        drop(external);
        assert_eq!(cache.remove_lru_batch(1)[0].0, key);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn local_hit_refreshes_recency_without_changing_class() {
        let cache = make_cache();
        let hit = BlockKey::new("ns".into(), vec![1]);
        let oldest = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert_resident_keys(vec![
            (hit.clone(), make_block()),
            (oldest.clone(), make_block()),
        ]);
        let inserted_at = backdate_resident(&cache, &hit, Duration::from_secs(60));
        let (count, _) = cache.get_prefix_blocks(std::slice::from_ref(&hit), true);

        assert_eq!(count, 1);
        assert_eq!(
            resident_metadata(&cache, &hit).unwrap().inserted_at,
            inserted_at
        );
        assert_eq!(cache.remove_lru_batch(1)[0].0, oldest);
        assert_class(&cache, &hit, ResidentClass::Reclaimable);
    }

    fn keys(prefix: u8, n: u8) -> Vec<BlockKey> {
        (0..n)
            .map(|i| BlockKey::new("ns".into(), vec![prefix, i]))
            .collect()
    }

    fn insert_all(cache: &ReadCache, keys: &[BlockKey]) {
        cache.batch_insert(keys.iter().map(|k| (k.clone(), make_block())).collect());
    }

    fn evict_order(cache: &ReadCache, n: usize) -> Vec<BlockKey> {
        (0..n)
            .flat_map(|_| cache.remove_lru_batch(1))
            .map(|(key, _)| key)
            .collect()
    }

    #[test]
    fn tail_first_evicts_deepest_block_of_oldest_window_first() {
        // (window percent, expected order). Chain a (4 blocks, oldest) then
        // chain b (2 blocks); both learned from prefix queries.
        let a = keys(1, 4);
        let b = keys(2, 2);
        let cases: [(&str, u64, Vec<&BlockKey>); 3] = [
            // Only a0 is in the window; its subtree goes leaf first.
            ("narrow", 10, vec![&a[3], &a[2], &a[1], &a[0], &b[1], &b[0]]),
            // Deepest across both chains, oldest first among equal depths.
            ("whole", 100, vec![&a[3], &a[2], &a[1], &b[1], &a[0], &b[0]]),
            ("lru", 0, vec![&a[0], &a[1], &a[2], &a[3], &b[0], &b[1]]),
        ];
        for (id, percent, expected) in cases {
            let cache = make_cache().with_tail_window_percent(percent);
            insert_all(&cache, &a);
            insert_all(&cache, &b);
            assert_eq!(cache.get_prefix_blocks(&a, true).0, 4, "{id}");
            assert_eq!(cache.get_prefix_blocks(&b, true).0, 2, "{id}");
            let expected: Vec<BlockKey> = expected.into_iter().cloned().collect();
            assert_eq!(evict_order(&cache, 6), expected, "{id}");
        }
    }

    #[test]
    fn queried_never_loaded_head_outlives_its_resident_descendants() {
        // The old connector queries from vLLM's local hit, so the chain head
        // is queried once, never loaded, and never refreshed again: it is the
        // LRU front while its descendants stay hot.
        let cache = make_cache();
        let chain = keys(1, 4);
        let newer = keys(2, 4);
        insert_all(&cache, &chain);
        assert_eq!(cache.get_prefix_blocks(&chain, true).0, 4);
        insert_all(&cache, &newer);
        assert_eq!(cache.get_prefix_blocks(&newer, true).0, 4);
        // Later turns hit chain[1..] only.
        assert_eq!(cache.get_prefix_blocks(&chain[1..], true).0, 3);

        let mut victims = Vec::new();
        for _ in 0..3 {
            victims.extend(evict_order(&cache, 1));
            assert!(
                cache.inner.lock().cache.contains_key(&chain[0]),
                "head evicted while {:?} is resident",
                chain[1..]
                    .iter()
                    .filter(|k| !victims.contains(k))
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(
            victims,
            vec![chain[3].clone(), chain[2].clone(), chain[1].clone()]
        );
        assert_eq!(evict_order(&cache, 1), vec![chain[0].clone()]);
    }

    #[test]
    fn group_blocks_leave_before_their_position_and_pins_hold_the_chain() {
        let cache = make_cache().with_tail_window_percent(100);
        let positions = keys(1, 2);
        let members: Vec<BlockKey> = positions
            .iter()
            .map(|k| BlockKey::new("ns".into(), [k.hash.as_slice(), &[0, 0, 0, 1]].concat()))
            .collect();
        insert_all(&cache, &members);
        insert_all(&cache, &positions);
        assert_eq!(cache.get_prefix_blocks(&positions, true).0, 2);
        cache.link_group_members(
            &positions
                .iter()
                .cloned()
                .zip(members.iter().cloned())
                .collect::<Vec<_>>(),
        );
        // The deepest position goes first, its group block before it.
        assert_eq!(
            evict_order(&cache, 2),
            vec![members[1].clone(), positions[1].clone()]
        );

        // A held group block keeps its position resident.
        let held = cache.peek_blocks(std::slice::from_ref(&members[0]));
        assert!(cache.remove_lru_batch(1).is_empty());
        drop(held);
        assert_eq!(
            evict_order(&cache, 2),
            vec![members[0].clone(), positions[0].clone()]
        );
    }

    #[test]
    fn serving_hit_refreshes_recency_without_changing_class() {
        let cache = make_cache();
        let hit = BlockKey::new("ns".into(), vec![1]);
        let oldest = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert(vec![
            (hit.clone(), make_block()),
            (oldest.clone(), make_block()),
        ]);
        let inserted_at = backdate_resident(&cache, &hit, Duration::from_secs(60));
        assert_eq!(cache.get_blocks(std::slice::from_ref(&hit)).len(), 1);

        assert_eq!(
            resident_metadata(&cache, &hit).unwrap().inserted_at,
            inserted_at
        );
        assert_eq!(cache.remove_lru_batch(1)[0].0, oldest);
        assert_class(&cache, &hit, ResidentClass::Retained);
    }

    #[test]
    fn already_existing_insert_keeps_original_class() {
        let cache = make_cache();
        let remote_first = BlockKey::new("ns".into(), vec![1]);
        let remote_other = BlockKey::new("ns".into(), vec![2]);
        let local_first = BlockKey::new("ns".into(), vec![3]);
        let local_other = BlockKey::new("ns".into(), vec![4]);

        cache.batch_insert_resident_keys(vec![(remote_first.clone(), make_block())]);
        cache.batch_insert_resident_keys(vec![(remote_other.clone(), make_block())]);
        cache.batch_insert(vec![(remote_first.clone(), make_block())]);
        cache.batch_insert(vec![(local_first.clone(), make_block())]);
        cache.batch_insert(vec![(local_other.clone(), make_block())]);
        cache.batch_insert_resident_keys(vec![(local_first.clone(), make_block())]);

        assert_class(&cache, &remote_first, ResidentClass::Reclaimable);
        assert_class(&cache, &local_first, ResidentClass::Retained);
        assert_eq!(cache.remove_lru_batch(1)[0].0, remote_other);
        assert_eq!(cache.remove_lru_batch(1)[0].0, remote_first);
        assert_eq!(cache.remove_lru_batch(1)[0].0, local_other);
        assert_class(&cache, &local_first, ResidentClass::Retained);
    }

    #[test]
    fn already_existing_insert_preserves_residence_start() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);
        let inserted_at = backdate_resident(&cache, &key, Duration::from_secs(60));

        cache.batch_insert_resident_keys(vec![(key.clone(), make_block())]);

        assert_eq!(
            resident_metadata(&cache, &key).unwrap().inserted_at,
            inserted_at
        );
        assert_class(&cache, &key, ResidentClass::Retained);
    }

    #[test]
    fn class_migration_preserves_residence_start() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);
        let inserted_at = backdate_resident(&cache, &key, Duration::from_secs(60));

        cache.mark_reclaimable_hashes(
            "ns",
            std::slice::from_ref(&key.hash),
            DemotionReason::ReclaimHint,
        );

        assert_eq!(
            resident_metadata(&cache, &key).unwrap().inserted_at,
            inserted_at
        );
        assert_class(&cache, &key, ResidentClass::Reclaimable);
    }

    #[test]
    fn reinsert_after_eviction_starts_new_residence_episode() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);
        let first_inserted_at = backdate_resident(&cache, &key, Duration::from_secs(60));

        cache.remove_lru_batch(1);
        cache.batch_insert(vec![(key.clone(), make_block())]);

        let second_inserted_at = resident_metadata(&cache, &key).unwrap().inserted_at;
        assert!(second_inserted_at > first_inserted_at);
    }

    #[test]
    fn residence_duration_is_non_negative_and_finite() {
        let removed_at = Instant::now();
        let inserted_at = removed_at - Duration::from_secs(60);

        assert_eq!(residence_duration_seconds(inserted_at, removed_at), 60.0);
        assert_eq!(residence_duration_seconds(removed_at, inserted_at), 0.0);
        assert!(residence_duration_seconds(inserted_at, removed_at).is_finite());
    }

    #[test]
    fn local_save_reports_resident_keys() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);

        assert_eq!(
            cache.batch_insert_refs(&[(key.clone(), make_block())]),
            vec![key.clone()]
        );
        assert_eq!(
            cache.batch_insert_refs(&[(key.clone(), make_block())]),
            vec![key]
        );
    }

    #[test]
    fn reclaimable_hashes_move_only_matching_residents() {
        let cache = make_cache();
        let retained = BlockKey::new("ns".into(), vec![1]);
        let reclaimable = BlockKey::new("ns".into(), vec![2]);
        let other_namespace = BlockKey::new("other".into(), vec![1]);

        cache.batch_insert(vec![
            (retained.clone(), make_block()),
            (other_namespace.clone(), make_block()),
        ]);
        cache.batch_insert_resident_keys(vec![(reclaimable.clone(), make_block())]);
        cache.mark_reclaimable_hashes(
            "ns",
            &[vec![1], vec![2], vec![3]],
            DemotionReason::ReclaimHint,
        );

        assert_class(&cache, &retained, ResidentClass::Reclaimable);
        assert_class(&cache, &reclaimable, ResidentClass::Reclaimable);
        assert_class(&cache, &other_namespace, ResidentClass::Retained);
    }

    #[test]
    fn reclaimable_hash_for_evicted_block_is_noop() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);
        cache.remove_lru_batch(1);

        cache.mark_reclaimable_hashes("ns", &[key.hash], DemotionReason::ReclaimHint);

        assert!(cache.remove_lru_batch(1).is_empty());
    }

    #[test]
    fn get_blocks_returns_existing_skips_missing() {
        let cache = make_cache();
        let key1 = BlockKey::new("ns".into(), vec![1]);
        let key2 = BlockKey::new("ns".into(), vec![2]);
        let key3 = BlockKey::new("ns".into(), vec![3]);

        cache.batch_insert(vec![
            (key1.clone(), make_block()),
            (key3.clone(), make_block()),
        ]);

        // key2 is missing — get_blocks should skip it (unlike prefix scan, no break)
        let result = cache.get_blocks(&[key1.clone(), key2.clone(), key3.clone()]);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, key1);
        assert_eq!(result[1].0, key3);
    }

    #[test]
    fn get_blocks_empty_input_returns_empty() {
        let cache = make_cache();
        let result = cache.get_blocks(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn get_blocks_all_missing_returns_empty() {
        let cache = make_cache();
        let key1 = BlockKey::new("ns".into(), vec![10]);
        let key2 = BlockKey::new("ns".into(), vec![20]);

        let result = cache.get_blocks(&[key1, key2]);
        assert!(result.is_empty());
    }

    #[test]
    fn get_blocks_is_idempotent() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);

        // Call get_blocks twice; both should return the same result
        let result1 = cache.get_blocks(std::slice::from_ref(&key));
        let result2 = cache.get_blocks(std::slice::from_ref(&key));
        assert_eq!(result1.len(), 1);
        assert_eq!(result2.len(), 1);
        assert_eq!(result1[0].0, result2[0].0);
    }

    #[test]
    fn get_blocks_does_not_break_at_first_miss() {
        // Contrast with get_prefix_blocks which stops at first miss
        let cache = make_cache();
        let keys: Vec<BlockKey> = (0u8..5)
            .map(|i| BlockKey::new("ns".into(), vec![i]))
            .collect();

        // Insert only even-indexed keys: 0, 2, 4
        for key in keys.iter().step_by(2) {
            cache.batch_insert(vec![(key.clone(), make_block())]);
        }

        // get_blocks: returns keys 0, 2, 4 (skips 1, 3)
        let result = cache.get_blocks(&keys);
        assert_eq!(result.len(), 3);

        // get_prefix_blocks: stops at key 1 (first miss), returns only key 0
        let (prefix_hit, _) = cache.get_prefix_blocks(&keys, true);
        assert_eq!(prefix_hit, 1);
    }

    #[test]
    fn remove_all_evicts_resident_blocks() {
        let cache = make_cache();
        let key1 = BlockKey::new("ns".into(), vec![1]);
        let key2 = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert(vec![
            (key1.clone(), make_block()),
            (key2.clone(), make_block()),
        ]);

        let removed = cache.remove_all();
        assert_eq!(removed.len(), 2);
        assert_eq!(cache.get_blocks(&[key1, key2]).len(), 0);
        let inner = cache.inner.lock();
        assert!(inner.reclaimable.is_empty());
        assert!(inner.retained.is_empty());
        drop(inner);
        assert!(cache.remove_all().is_empty());
    }

    #[test]
    fn batch_insert_resident_keys_excludes_lfu_rejected_blocks() {
        let cache = ReadCache::new(1, true, Some(1));
        let hot_key = BlockKey::new("ns".into(), vec![1]);
        let cold_key = BlockKey::new("ns".into(), vec![2]);

        assert_eq!(
            cache.batch_insert_resident_keys(vec![(hot_key.clone(), make_block())]),
            vec![hot_key.clone()]
        );

        for _ in 0..2 {
            assert_eq!(cache.get_blocks(std::slice::from_ref(&hot_key)).len(), 1);
        }

        assert!(
            cache
                .batch_insert_resident_keys(vec![(cold_key.clone(), make_block())])
                .is_empty()
        );
        assert!(!cache.inner.lock().reclaimable.contains_key(&cold_key));
        assert_eq!(cache.get_blocks(&[hot_key]).len(), 1);
        assert_eq!(cache.get_blocks(&[cold_key]).len(), 0);
    }

    #[test]
    fn batch_insert_resident_keys_includes_already_existing_blocks() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);

        assert_eq!(
            cache.batch_insert_resident_keys(vec![(key.clone(), make_block())]),
            vec![key]
        );
    }
}
