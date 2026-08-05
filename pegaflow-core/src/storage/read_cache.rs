use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use hashlink::LruCache;
use parking_lot::Mutex;

use crate::block::{BlockKey, SealedBlock};
use crate::cache::{CacheInsertOutcome, TinyLfuCache};
use crate::metrics::{CACHE_CLASS_RECLAIMABLE, CACHE_CLASS_RETAINED, core_metrics};

pub(crate) struct ReadCache {
    inner: Mutex<ReadCacheInner>,
}

struct ReadCacheInner {
    cache: TinyLfuCache<BlockKey, Arc<SealedBlock>>,
    s3_fifo: S3FifoState,
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
                s3_fifo: S3FifoState::new(),
            }),
        }
    }

    pub(super) fn contains_keys(&self, keys: &[BlockKey]) -> Vec<bool> {
        let inner = self.inner.lock();
        keys.iter().map(|k| inner.cache.contains_key(k)).collect()
    }

    /// Scan cache for a prefix of `keys`, stopping at the first miss.
    pub(super) fn get_prefix_blocks(&self, keys: &[BlockKey]) -> (usize, Vec<Arc<SealedBlock>>) {
        let mut hit = 0usize;
        let mut blocks = Vec::with_capacity(keys.len());
        let mut inner = self.inner.lock();
        for key in keys {
            if let Some(block) = inner.cache.get(key) {
                hit += 1;
                blocks.push(block);
            } else {
                break;
            }
        }
        for key in keys[..hit].iter().rev() {
            inner.s3_fifo.hit(key);
        }
        (hit, blocks)
    }

    pub(super) fn batch_insert(&self, blocks: Vec<(BlockKey, Arc<SealedBlock>)>) {
        let mut inner = self.inner.lock();
        for (key, block) in blocks {
            insert_block(&mut inner, key, block, ResidentClass::Retained);
        }
    }

    pub(super) fn batch_insert_resident_keys(
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

    /// Look up specific blocks by key without prefix-scan semantics (does not
    /// stop at first miss). Used by the serving side of cross-node transfer.
    pub(super) fn get_blocks(&self, keys: &[BlockKey]) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let mut inner = self.inner.lock();
        let mut found = Vec::new();
        for key in keys {
            if let Some(block) = inner.cache.get(key) {
                found.push((key.clone(), block));
            }
        }
        for (key, _) in found.iter().rev() {
            inner.s3_fifo.hit(key);
        }
        found
    }

    pub(super) fn touch_keys_in_reverse(&self, keys: &[BlockKey]) {
        let mut inner = self.inner.lock();
        touch_keys_in_reverse(&mut inner, keys);
    }

    pub(super) fn remove_lru_batch(&self, batch_size: usize) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let mut inner = self.inner.lock();
        let mut evicted = Vec::with_capacity(batch_size);
        while evicted.len() < batch_size {
            let next = remove_s3_fifo(&mut inner, ResidentClass::Reclaimable)
                .or_else(|| remove_s3_fifo(&mut inner, ResidentClass::Retained));
            let Some(block) = next else {
                break;
            };
            evicted.push(block);
        }
        evicted
    }

    pub(super) fn remove_all(&self) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let mut inner = self.inner.lock();
        let reclaimable_blocks = inner.s3_fifo.class_len(ResidentClass::Reclaimable) as i64;
        let retained_blocks = inner.s3_fifo.class_len(ResidentClass::Retained) as i64;
        inner.s3_fifo.clear();
        let removed = inner.cache.remove_all();
        debug_assert_eq!(
            removed.len() as i64,
            reclaimable_blocks + retained_blocks,
            "resident cache and replacement classes diverged"
        );
        let metrics = core_metrics();
        metrics
            .cache_resident_blocks
            .add(-reclaimable_blocks, &*CACHE_CLASS_RECLAIMABLE);
        metrics
            .cache_resident_blocks
            .add(-retained_blocks, &*CACHE_CLASS_RETAINED);
        removed
    }

    pub(crate) fn mark_reclaimable_hashes(&self, namespace: &str, hashes: &[Vec<u8>]) {
        if hashes.is_empty() {
            return;
        }

        let mut inner = self.inner.lock();
        let mut moved = 0;
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            if mark_reclaimable(&mut inner, &key) {
                moved += 1;
            }
        }
        if moved > 0 {
            inner.s3_fifo.compact_queues(ResidentClass::Retained);
            let metrics = core_metrics();
            metrics
                .cache_resident_blocks
                .add(-moved, &*CACHE_CLASS_RETAINED);
            metrics
                .cache_resident_blocks
                .add(moved, &*CACHE_CLASS_RECLAIMABLE);
        }
    }

    #[cfg(test)]
    pub(crate) fn insert_retained_for_test(&self, key: BlockKey, block: Arc<SealedBlock>) {
        let mut inner = self.inner.lock();
        insert_block(&mut inner, key, block, ResidentClass::Retained);
    }

    #[cfg(test)]
    pub(crate) fn is_reclaimable_for_test(&self, key: &BlockKey) -> bool {
        self.inner
            .lock()
            .s3_fifo
            .contains(key, ResidentClass::Reclaimable)
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
            inner.s3_fifo.insert(class, key, footprint_bytes);
            let m = core_metrics();
            m.cache_block_insertions.add(1, &[]);
            m.cache_resident_bytes.add(footprint_bytes as i64, &[]);
            m.cache_resident_blocks.add(1, class.attributes());
        }
        CacheInsertOutcome::AlreadyExists => inner.s3_fifo.hit(&key),
        CacheInsertOutcome::Rejected => {
            core_metrics().cache_block_admission_rejections.add(1, &[]);
        }
    }
    outcome
}

fn touch_keys_in_reverse(inner: &mut ReadCacheInner, keys: &[BlockKey]) {
    for key in keys.iter().rev() {
        if inner.cache.contains_key(key) {
            inner.s3_fifo.hit(key);
        }
    }
}
fn mark_reclaimable(inner: &mut ReadCacheInner, key: &BlockKey) -> bool {
    if !inner.cache.contains_key(key) {
        return false;
    }
    if inner.s3_fifo.contains(key, ResidentClass::Retained) {
        let bytes = inner.s3_fifo.remove(ResidentClass::Retained, key);
        inner
            .s3_fifo
            .insert(ResidentClass::Reclaimable, key.clone(), bytes);
        true
    } else {
        debug_assert!(
            inner.s3_fifo.contains(key, ResidentClass::Reclaimable),
            "resident block is missing its replacement class"
        );
        false
    }
}

fn remove_s3_fifo(
    inner: &mut ReadCacheInner,
    class: ResidentClass,
) -> Option<(BlockKey, Arc<SealedBlock>)> {
    while let Some(key) = inner.s3_fifo.remove_next(class) {
        let block = inner.cache.remove(&key);
        debug_assert!(block.is_some(), "S3-FIFO contains a non-resident block");
        let Some(block) = block else {
            continue;
        };
        let metrics = core_metrics();
        metrics.cache_resident_blocks.add(-1, class.attributes());
        metrics
            .cache_block_evictions_by_class
            .add(1, class.attributes());
        return Some((key, block));
    }
    None
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum S3Queue {
    Small,
    Main,
}

#[derive(Copy, Clone, Debug)]
struct S3Entry {
    queue: S3Queue,
    frequency: u8,
    footprint_bytes: u64,
}

struct S3FifoClass {
    small: VecDeque<BlockKey>,
    main: VecDeque<BlockKey>,
    ghost: LruCache<BlockKey, ()>,
    entries: HashMap<BlockKey, S3Entry>,
    resident_bytes: u64,
    small_bytes: u64,
}

impl Default for S3FifoClass {
    fn default() -> Self {
        Self {
            small: VecDeque::new(),
            main: VecDeque::new(),
            ghost: LruCache::new_unbounded(),
            entries: HashMap::new(),
            resident_bytes: 0,
            small_bytes: 0,
        }
    }
}

struct S3FifoState {
    reclaimable: S3FifoClass,
    retained: S3FifoClass,
}

impl S3FifoState {
    fn new() -> Self {
        Self {
            reclaimable: S3FifoClass::default(),
            retained: S3FifoClass::default(),
        }
    }

    fn class(&self, class: ResidentClass) -> &S3FifoClass {
        match class {
            ResidentClass::Reclaimable => &self.reclaimable,
            ResidentClass::Retained => &self.retained,
        }
    }

    fn class_mut(&mut self, class: ResidentClass) -> &mut S3FifoClass {
        match class {
            ResidentClass::Reclaimable => &mut self.reclaimable,
            ResidentClass::Retained => &mut self.retained,
        }
    }

    fn insert(&mut self, class: ResidentClass, key: BlockKey, footprint_bytes: u64) {
        let state = self.class_mut(class);
        debug_assert!(!state.entries.contains_key(&key));
        if state.ghost.remove(&key).is_some() {
            state.main.push_back(key.clone());
            state.entries.insert(
                key,
                S3Entry {
                    queue: S3Queue::Main,
                    frequency: 0,
                    footprint_bytes,
                },
            );
        } else {
            state.small.push_back(key.clone());
            state.small_bytes = state.small_bytes.saturating_add(footprint_bytes);
            state.entries.insert(
                key,
                S3Entry {
                    queue: S3Queue::Small,
                    frequency: 0,
                    footprint_bytes,
                },
            );
        }
        state.resident_bytes = state.resident_bytes.saturating_add(footprint_bytes);
    }

    fn hit(&mut self, key: &BlockKey) {
        for class in [ResidentClass::Reclaimable, ResidentClass::Retained] {
            let state = self.class_mut(class);
            if let Some(entry) = state.entries.get_mut(key) {
                entry.frequency = entry.frequency.saturating_add(1).min(3);
                return;
            }
        }
        debug_assert!(false, "S3-FIFO hit for missing resident key");
    }

    fn contains(&self, key: &BlockKey, class: ResidentClass) -> bool {
        self.class(class).entries.contains_key(key)
    }

    fn remove(&mut self, class: ResidentClass, key: &BlockKey) -> u64 {
        let state = self.class_mut(class);
        let Some(entry) = state.entries.remove(key) else {
            return 0;
        };
        state.resident_bytes = state.resident_bytes.saturating_sub(entry.footprint_bytes);
        if entry.queue == S3Queue::Small {
            state.small_bytes = state.small_bytes.saturating_sub(entry.footprint_bytes);
        }
        state.trim_ghost();
        entry.footprint_bytes
    }

    fn compact_queues(&mut self, class: ResidentClass) {
        let state = self.class_mut(class);
        let entries = &state.entries;
        state.small.retain(|key| {
            entries
                .get(key)
                .is_some_and(|entry| entry.queue == S3Queue::Small)
        });
        state.main.retain(|key| {
            entries
                .get(key)
                .is_some_and(|entry| entry.queue == S3Queue::Main)
        });
        state.trim_ghost();
    }

    fn class_len(&self, class: ResidentClass) -> usize {
        self.class(class).entries.len()
    }

    fn clear(&mut self) {
        self.reclaimable = S3FifoClass::default();
        self.retained = S3FifoClass::default();
    }

    fn remove_next(&mut self, class: ResidentClass) -> Option<BlockKey> {
        let state = self.class_mut(class);
        loop {
            let small_target = state.resident_bytes / 10;
            let queue = if !state.small.is_empty()
                && (state.main.is_empty() || state.small_bytes >= small_target)
            {
                S3Queue::Small
            } else if !state.main.is_empty() {
                S3Queue::Main
            } else if !state.small.is_empty() {
                S3Queue::Small
            } else {
                return None;
            };
            let key = match queue {
                S3Queue::Small => state.small.pop_front(),
                S3Queue::Main => state.main.pop_front(),
            }?;
            let Some(entry) = state.entries.get_mut(&key) else {
                continue;
            };
            if entry.queue != queue {
                continue;
            }
            match queue {
                S3Queue::Small if entry.frequency > 1 => {
                    entry.queue = S3Queue::Main;
                    entry.frequency = 0;
                    state.small_bytes = state.small_bytes.saturating_sub(entry.footprint_bytes);
                    state.main.push_back(key);
                }
                S3Queue::Small => {
                    let entry = state.entries.remove(&key)?;
                    state.resident_bytes =
                        state.resident_bytes.saturating_sub(entry.footprint_bytes);
                    state.small_bytes = state.small_bytes.saturating_sub(entry.footprint_bytes);
                    state.ghost.insert(key.clone(), ());
                    state.trim_ghost();
                    return Some(key);
                }
                S3Queue::Main if entry.frequency > 0 => {
                    entry.frequency -= 1;
                    state.main.push_back(key);
                }
                S3Queue::Main => {
                    let entry = state.entries.remove(&key)?;
                    state.resident_bytes =
                        state.resident_bytes.saturating_sub(entry.footprint_bytes);
                    state.trim_ghost();
                    return Some(key);
                }
            }
        }
    }
}

impl S3FifoClass {
    fn trim_ghost(&mut self) {
        let limit = self.main.len().max(1);
        while self.ghost.len() > limit {
            self.ghost.remove_lru();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache() -> ReadCache {
        ReadCache::new(1 << 20, false, None)
    }

    fn make_block() -> Arc<SealedBlock> {
        Arc::new(SealedBlock::from_slots(Vec::new()))
    }

    fn make_key(id: u8) -> BlockKey {
        BlockKey::new("ns".into(), vec![id])
    }

    fn assert_class(cache: &ReadCache, key: &BlockKey, expected: ResidentClass) {
        let inner = cache.inner.lock();
        assert!(inner.cache.contains_key(key));
        assert_eq!(
            inner.s3_fifo.contains(key, ResidentClass::Reclaimable),
            expected == ResidentClass::Reclaimable
        );
        assert_eq!(
            inner.s3_fifo.contains(key, ResidentClass::Retained),
            expected == ResidentClass::Retained
        );
    }

    #[test]
    fn s3_fifo_ghost_hit_enters_main() {
        let mut state = S3FifoState::new();
        let key = make_key(1);

        state.insert(ResidentClass::Retained, key.clone(), 100);
        assert_eq!(
            state.class(ResidentClass::Retained).entries[&key].queue,
            S3Queue::Small
        );
        assert_eq!(
            state.remove_next(ResidentClass::Retained),
            Some(key.clone())
        );
        assert!(
            state
                .class(ResidentClass::Retained)
                .ghost
                .contains_key(&key)
        );

        state.insert(ResidentClass::Retained, key.clone(), 100);
        let retained = state.class(ResidentClass::Retained);
        assert_eq!(retained.entries[&key].queue, S3Queue::Main);
        assert!(!retained.ghost.contains_key(&key));
    }

    #[test]
    fn s3_fifo_caps_frequency_and_promotes_reused_small_entry() {
        let mut state = S3FifoState::new();
        let hot = make_key(1);
        let cold = make_key(2);
        state.insert(ResidentClass::Retained, hot.clone(), 100);
        state.insert(ResidentClass::Retained, cold.clone(), 100);

        for _ in 0..4 {
            state.hit(&hot);
        }
        assert_eq!(
            state.class(ResidentClass::Retained).entries[&hot].frequency,
            3
        );

        assert_eq!(state.remove_next(ResidentClass::Retained), Some(cold));
        let hot_entry = state
            .class(ResidentClass::Retained)
            .entries
            .get(&hot)
            .expect("hot entry should remain resident");
        assert_eq!(hot_entry.queue, S3Queue::Main);
        assert_eq!(hot_entry.frequency, 0);
    }

    #[test]
    fn s3_fifo_requeues_main_entry_with_frequency() {
        let mut state = S3FifoState::new();
        let first = make_key(1);
        let second = make_key(2);

        for key in [&first, &second] {
            state.insert(ResidentClass::Retained, key.clone(), 100);
            assert_eq!(
                state.remove_next(ResidentClass::Retained),
                Some(key.clone())
            );
            state.insert(ResidentClass::Retained, key.clone(), 100);
        }
        state.hit(&first);

        assert_eq!(state.remove_next(ResidentClass::Retained), Some(second));
        let first_entry = state
            .class(ResidentClass::Retained)
            .entries
            .get(&first)
            .expect("frequent main entry should remain resident");
        assert_eq!(first_entry.queue, S3Queue::Main);
        assert_eq!(first_entry.frequency, 0);
    }

    #[test]
    fn s3_fifo_uses_resident_bytes_for_small_target() {
        let mut state = S3FifoState::new();
        let main = make_key(1);
        let small = make_key(2);

        state.insert(ResidentClass::Retained, main.clone(), 990);
        assert_eq!(
            state.remove_next(ResidentClass::Retained),
            Some(main.clone())
        );
        state.insert(ResidentClass::Retained, main.clone(), 990);
        state.insert(ResidentClass::Retained, small.clone(), 10);

        assert_eq!(state.remove_next(ResidentClass::Retained), Some(main));
        assert!(state.contains(&small, ResidentClass::Retained));
    }

    #[test]
    fn s3_fifo_bounds_ghost_by_main_entries() {
        let mut state = S3FifoState::new();
        let main_keys = [make_key(1), make_key(2)];
        for key in &main_keys {
            state.insert(ResidentClass::Retained, key.clone(), 1_000);
            assert_eq!(
                state.remove_next(ResidentClass::Retained),
                Some(key.clone())
            );
            state.insert(ResidentClass::Retained, key.clone(), 1_000);
        }

        let ghost_candidates = [make_key(3), make_key(4), make_key(5)];
        for key in &ghost_candidates {
            state.insert(ResidentClass::Retained, key.clone(), 1_000);
            assert_eq!(
                state.remove_next(ResidentClass::Retained),
                Some(key.clone())
            );
        }

        let retained = state.class(ResidentClass::Retained);
        assert_eq!(retained.main.len(), 2);
        assert_eq!(retained.ghost.len(), 2);
        assert!(!retained.ghost.contains_key(&ghost_candidates[0]));
        assert!(retained.ghost.contains_key(&ghost_candidates[1]));
        assert!(retained.ghost.contains_key(&ghost_candidates[2]));
    }

    #[test]
    fn s3_fifo_classes_have_independent_frequency_and_bytes() {
        let mut state = S3FifoState::new();
        let reclaimable = make_key(1);
        let retained = make_key(2);
        state.insert(ResidentClass::Reclaimable, reclaimable.clone(), 10);
        state.insert(ResidentClass::Retained, retained.clone(), 20);
        state.hit(&retained);

        assert_eq!(state.class(ResidentClass::Reclaimable).resident_bytes, 10);
        assert_eq!(state.class(ResidentClass::Reclaimable).small_bytes, 10);
        assert_eq!(state.class(ResidentClass::Retained).resident_bytes, 20);
        assert_eq!(
            state.class(ResidentClass::Retained).entries[&retained].frequency,
            1
        );
        assert_eq!(
            state.class(ResidentClass::Reclaimable).entries[&reclaimable].frequency,
            0
        );
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
        cache.batch_insert_resident_keys(vec![(remote.clone(), make_block())]);

        assert_class(&cache, &local, ResidentClass::Retained);
        assert_class(&cache, &ssd, ResidentClass::Retained);
        assert_class(&cache, &remote, ResidentClass::Reclaimable);
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
    fn repeated_local_hit_promotes_entry_without_changing_class() {
        let cache = make_cache();
        let hit = BlockKey::new("ns".into(), vec![1]);
        let oldest = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert_resident_keys(vec![
            (hit.clone(), make_block()),
            (oldest.clone(), make_block()),
        ]);
        let (first_count, _) = cache.get_prefix_blocks(std::slice::from_ref(&hit));
        let (second_count, _) = cache.get_prefix_blocks(std::slice::from_ref(&hit));

        assert_eq!((first_count, second_count), (1, 1));
        assert_eq!(cache.remove_lru_batch(1)[0].0, oldest);
        assert_class(&cache, &hit, ResidentClass::Reclaimable);
    }

    #[test]
    fn repeated_serving_hit_promotes_entry_without_changing_class() {
        let cache = make_cache();
        let hit = BlockKey::new("ns".into(), vec![1]);
        let oldest = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert(vec![
            (hit.clone(), make_block()),
            (oldest.clone(), make_block()),
        ]);
        assert_eq!(cache.get_blocks(std::slice::from_ref(&hit)).len(), 1);
        assert_eq!(cache.get_blocks(std::slice::from_ref(&hit)).len(), 1);

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
        cache.batch_insert(vec![(remote_first.clone(), make_block())]);
        cache.batch_insert(vec![(local_first.clone(), make_block())]);
        cache.batch_insert(vec![(local_other.clone(), make_block())]);
        cache.batch_insert_resident_keys(vec![(local_first.clone(), make_block())]);
        cache.batch_insert_resident_keys(vec![(local_first.clone(), make_block())]);

        assert_class(&cache, &remote_first, ResidentClass::Reclaimable);
        assert_class(&cache, &local_first, ResidentClass::Retained);
        assert_eq!(cache.remove_lru_batch(1)[0].0, remote_other);
        assert_eq!(cache.remove_lru_batch(1)[0].0, remote_first);
        assert_eq!(cache.remove_lru_batch(1)[0].0, local_other);
        assert_class(&cache, &local_first, ResidentClass::Retained);
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
        cache.mark_reclaimable_hashes("ns", &[vec![1], vec![2], vec![3]]);

        assert_class(&cache, &retained, ResidentClass::Reclaimable);
        assert_class(&cache, &reclaimable, ResidentClass::Reclaimable);
        assert_class(&cache, &other_namespace, ResidentClass::Retained);
    }

    #[test]
    fn reclaimable_hint_preserves_registration_order_within_class() {
        let cache = make_cache();
        let keys: Vec<_> = (1..=3)
            .map(|value| BlockKey::new("ns".into(), vec![value]))
            .collect();
        cache.batch_insert(
            keys.iter()
                .cloned()
                .map(|key| (key, make_block()))
                .collect(),
        );

        cache.mark_reclaimable_hashes(
            "ns",
            &[
                keys[2].hash.clone(),
                keys[1].hash.clone(),
                keys[0].hash.clone(),
            ],
        );

        assert_eq!(cache.remove_lru_batch(3)[0].0, keys[2]);
    }

    #[test]
    fn reclaimable_hash_for_evicted_block_is_noop() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);
        cache.remove_lru_batch(1);

        cache.mark_reclaimable_hashes("ns", &[key.hash]);

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
        let (prefix_hit, _) = cache.get_prefix_blocks(&keys);
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
        assert_eq!(inner.s3_fifo.class_len(ResidentClass::Reclaimable), 0);
        assert_eq!(inner.s3_fifo.class_len(ResidentClass::Retained), 0);
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
        assert!(
            !cache
                .inner
                .lock()
                .s3_fifo
                .contains(&cold_key, ResidentClass::Reclaimable)
        );
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
