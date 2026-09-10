use dashmap::DashMap;
use log::{info, warn};
use pegaflow_common::BlockKey;
use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};
use uuid::Uuid;

const MIN_RECLAIMABLE_OWNER_COUNT: usize = 3;
const MUTATION_BATCH_SIZE: usize = 256;

pub const DEFAULT_NODE_STALE_SECS: u64 = 30;
pub const DEFAULT_TTL_MINUTES: u64 = 120;
pub const MANUAL_CLEANUP_AGE_SECS: u64 = 60 * 60;

/// A prefix query result: one block hash and all live nodes that own it.
#[derive(Debug, Clone)]
pub struct PrefixEntry {
    pub block_hash: Vec<u8>,
    pub nodes: Vec<Arc<str>>,
}

#[derive(Debug, Clone, Copy)]
pub struct StoreConfig {
    pub node_stale_after: Duration,
    /// Node inactivity grace period; never applies to owner registration age.
    pub ttl: Duration,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            node_stale_after: Duration::from_secs(DEFAULT_NODE_STALE_SECS),
            ttl: Duration::from_secs(DEFAULT_TTL_MINUTES * 60),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SweepStats {
    pub removed_owners: usize,
    pub removed_keys: usize,
    pub removed_nodes: usize,
}

impl SweepStats {
    pub fn is_empty(self) -> bool {
        self.removed_owners == 0 && self.removed_keys == 0 && self.removed_nodes == 0
    }
}

/// Incremental distribution of stored owner records, not query-visible owners.
/// Stale nodes remain counted until TTL cleanup, and superseded sessions until
/// reconciliation. Reads of separate atomic fields may straddle a mutation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RedundancySnapshot {
    pub keys_1: u64,
    pub keys_2: u64,
    pub keys_3: u64,
    pub keys_4plus: u64,
    pub copies: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreError {
    UnknownNode,
    StaleSession,
}

/// Snapshot of the session that wrote this block ownership. Query compares it
/// with the current `NodeRecord.node_id` to filter owners left by old sessions.
#[derive(Debug, Clone)]
struct OwnerRecord {
    node_id: Uuid,
    key_register_time: Instant,
}

/// Authoritative current session for a node URL. `last_seen` is bumped on
/// heartbeat/insert/remove and gates query visibility.
#[derive(Debug, Clone)]
struct NodeRecord {
    node_id: Uuid,
    last_seen: Instant,
}

#[derive(Default)]
struct RedundancyCounters {
    keys_1: AtomicU64,
    keys_2: AtomicU64,
    keys_3: AtomicU64,
    keys_4plus: AtomicU64,
    copies: AtomicU64,
}

impl RedundancyCounters {
    fn snapshot(&self) -> RedundancySnapshot {
        RedundancySnapshot {
            keys_1: self.keys_1.load(Ordering::Relaxed),
            keys_2: self.keys_2.load(Ordering::Relaxed),
            keys_3: self.keys_3.load(Ordering::Relaxed),
            keys_4plus: self.keys_4plus.load(Ordering::Relaxed),
            copies: self.copies.load(Ordering::Relaxed),
        }
    }

    fn adjust_bucket(&self, count: u64, delta: i64) {
        let counter = match count {
            1 => &self.keys_1,
            2 => &self.keys_2,
            3 => &self.keys_3,
            _ if count >= 4 => &self.keys_4plus,
            _ => return,
        };
        if delta > 0 {
            counter.fetch_add(delta as u64, Ordering::Relaxed);
        } else {
            counter.fetch_sub((-delta) as u64, Ordering::Relaxed);
        }
    }

    fn adjust(&self, before: u64, after: u64) {
        if before == after {
            return;
        }
        self.adjust_bucket(before, -1);
        self.adjust_bucket(after, 1);
        if after > before {
            self.copies.fetch_add(after - before, Ordering::Relaxed);
        } else {
            self.copies.fetch_sub(before - after, Ordering::Relaxed);
        }
    }

    fn reset(&self) {
        self.keys_1.store(0, Ordering::Relaxed);
        self.keys_2.store(0, Ordering::Relaxed);
        self.keys_3.store(0, Ordering::Relaxed);
        self.keys_4plus.store(0, Ordering::Relaxed);
        self.copies.store(0, Ordering::Relaxed);
    }
}

#[derive(Default)]
struct MutationState {
    reconcile_needed: bool,
}

/// Async thread-safe block hash storage using DashMap.
///
/// `blocks` maps each block key to node URL ownership records. `nodes` tracks
/// the current MetaServer session and liveness for each node URL.
pub struct BlockHashStore {
    blocks: DashMap<BlockKey, HashMap<Arc<str>, OwnerRecord>>,
    nodes: DashMap<Arc<str>, NodeRecord>,
    config: StoreConfig,
    /// Serializes metadata mutations and reconciliation scheduling; per-key
    /// owner changes update aggregate counters within the same boundary.
    mutation_lock: Mutex<MutationState>,
    owners_total: AtomicU64,
    /// Incremental stored-owner redundancy counters read by metric callbacks.
    redundancy: RedundancyCounters,
}

impl BlockHashStore {
    pub fn new() -> Self {
        Self::with_config(StoreConfig::default())
    }

    pub fn with_config(config: StoreConfig) -> Self {
        Self {
            blocks: DashMap::new(),
            nodes: DashMap::new(),
            config,
            mutation_lock: Mutex::new(MutationState::default()),
            owners_total: AtomicU64::new(0),
            redundancy: RedundancyCounters::default(),
        }
    }

    pub fn with_ttl(ttl_minutes: u64) -> Self {
        Self::with_config(StoreConfig {
            node_stale_after: Duration::from_secs(DEFAULT_NODE_STALE_SECS),
            ttl: Duration::from_secs(ttl_minutes * 60),
        })
    }

    pub fn config(&self) -> StoreConfig {
        self.config
    }

    pub fn heartbeat_node(&self, node: &str, node_id: Uuid) -> Result<(), StoreError> {
        let mut mutation = self.mutation_lock.lock().expect("mutation lock poisoned");
        let now = Instant::now();
        let Some(current) = self.nodes.get(node).map(|record| record.clone()) else {
            info!("MetaServer node registered: node={node} node_id={node_id}");
            self.nodes.insert(
                Arc::from(node),
                NodeRecord {
                    node_id,
                    last_seen: now,
                },
            );
            return Ok(());
        };

        let same_session = current.node_id == node_id;
        if same_session {
            return self.touch_node_session(node, node_id);
        }
        let stale_session = now.duration_since(current.last_seen) > self.config.node_stale_after;
        if stale_session {
            info!(
                "MetaServer node session takeover: node={} old_node_id={} new_node_id={}",
                node, current.node_id, node_id
            );
            let mut record = self
                .nodes
                .get_mut(node)
                .expect("node disappeared under mutation lock");
            record.node_id = node_id;
            record.last_seen = now;
            drop(record);
            mutation.reconcile_needed = true;
            return Ok(());
        }

        warn!(
            "MetaServer heartbeat rejected stale session: node={} current_node_id={} rejected_node_id={}",
            node, current.node_id, node_id
        );
        Err(StoreError::StaleSession)
    }

    pub fn unregister_node(&self, node: &str, node_id: Uuid) -> Result<usize, StoreError> {
        let _mutation_guard = self.mutation_lock.lock().expect("mutation lock poisoned");
        let Some(record) = self.nodes.get(node) else {
            warn!(
                "MetaServer unregister rejected unknown node: node={} node_id={}",
                node, node_id
            );
            return Err(StoreError::UnknownNode);
        };
        if record.node_id != node_id {
            warn!(
                "MetaServer unregister rejected stale session: node={} rejected_node_id={}",
                node, node_id
            );
            return Err(StoreError::StaleSession);
        }
        drop(record);

        // Session validation above also authorizes removal of this address's
        // historical owners, so removing the node cannot leave orphan records.
        let removed = self.retain_owners(|owner_node, _| owner_node.as_ref() != node);
        self.nodes
            .remove_if(node, |_, record| record.node_id == node_id);
        Ok(removed.removed_owners)
    }

    pub fn insert_hashes(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
        node: &str,
        node_id: Uuid,
    ) -> Result<Vec<Vec<u8>>, StoreError> {
        let node: Arc<str> = Arc::from(node);
        let mut reclaimable_hashes = Vec::new();
        for batch in hashes.chunks(MUTATION_BATCH_SIZE) {
            let _mutation_guard = self.mutation_lock.lock().expect("mutation lock poisoned");
            self.touch_node_session(&node, node_id)?;
            let now = Instant::now();
            for hash in batch {
                let key = BlockKey::new(namespace.to_string(), hash.clone());
                let mut owners = self.blocks.entry(key).or_default();
                let before_count = owners.len();
                let previous = owners.insert(
                    Arc::clone(&node),
                    OwnerRecord {
                        node_id,
                        key_register_time: now,
                    },
                );
                let is_new_owner = previous
                    .as_ref()
                    .is_none_or(|owner| owner.node_id != node_id);
                if is_new_owner
                    && owners
                        .iter()
                        .filter(|(node, owner)| self.is_owner_visible(node, owner, now))
                        .take(MIN_RECLAIMABLE_OWNER_COUNT)
                        .count()
                        == MIN_RECLAIMABLE_OWNER_COUNT
                {
                    reclaimable_hashes.push(hash.clone());
                }
                self.record_owner_change(before_count, owners.len());
                drop(owners);
            }
        }
        Ok(reclaimable_hashes)
    }

    pub fn remove_hashes(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
        node: &str,
        node_id: Uuid,
    ) -> Result<usize, StoreError> {
        let mut removed = 0;
        for batch in hashes.chunks(MUTATION_BATCH_SIZE) {
            let _mutation_guard = self.mutation_lock.lock().expect("mutation lock poisoned");
            self.touch_node_session(node, node_id)?;
            for hash in batch {
                let key = BlockKey::new(namespace.to_string(), hash.clone());
                let change = if let Some(mut owners) = self.blocks.get_mut(&key) {
                    let before_count = owners.len();
                    if owners
                        .get(node)
                        .is_some_and(|owner| owner.node_id == node_id)
                    {
                        owners.remove(node);
                        removed += 1;
                        Some((before_count, owners.len()))
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some((before_count, after_count)) = change {
                    self.record_owner_change(before_count, after_count);
                }
                if self
                    .blocks
                    .get(&key)
                    .is_some_and(|owners| owners.is_empty())
                {
                    self.blocks.remove_if(&key, |_, owners| owners.is_empty());
                }
            }
        }
        Ok(removed)
    }

    /// Query the longest prefix of `hashes` with at least one live owner.
    pub fn query_prefix(&self, namespace: &str, hashes: &[Vec<u8>]) -> Vec<PrefixEntry> {
        let now = Instant::now();
        let mut result = Vec::new();
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            let Some(owners) = self.blocks.get(&key) else {
                break;
            };

            let live_nodes: Vec<Arc<str>> = owners
                .iter()
                .filter_map(|(node, owner)| {
                    if self.is_owner_visible(node, owner, now) {
                        Some(Arc::clone(node))
                    } else {
                        None
                    }
                })
                .collect();

            if live_nodes.is_empty() {
                break;
            }

            result.push(PrefixEntry {
                block_hash: hash.clone(),
                nodes: live_nodes,
            });
        }
        result
    }

    /// Healthy sweeps only inspect nodes. A takeover or node TTL expiry triggers
    /// one block scan; owner registration age never causes background deletion.
    pub fn sweep_expired(&self) -> SweepStats {
        let mut mutation = self.mutation_lock.lock().expect("mutation lock poisoned");
        let now = Instant::now();
        let node_expired = self
            .nodes
            .iter()
            .any(|record| now.duration_since(record.last_seen) > self.config.ttl);
        if !node_expired && !mutation.reconcile_needed {
            return SweepStats::default();
        }

        let mut stats = self.retain_owners(|node, owner| {
            self.nodes.get(node.as_ref()).is_some_and(|record| {
                record.node_id == owner.node_id
                    && now.duration_since(record.last_seen) <= self.config.ttl
            })
        });
        self.nodes.retain(|node, record| {
            let keep = now.duration_since(record.last_seen) <= self.config.ttl;
            if !keep {
                stats.removed_nodes += 1;
                info!(
                    "MetaServer node swept: node={} node_id={} last_seen_age_secs={}",
                    node,
                    record.node_id,
                    now.duration_since(record.last_seen).as_secs()
                );
            }
            keep
        });
        // The same mutex serializes session changes and consuming the flag.
        // A takeover after this scan sets the flag for the next sweep.
        mutation.reconcile_needed = false;
        stats
    }

    /// Remove ownership records older than `max_age`, regardless of node
    /// liveness. This is reserved for explicit operator maintenance.
    pub fn remove_owners_older_than(&self, max_age: Duration) -> SweepStats {
        let now = Instant::now();
        let mut stats = SweepStats::default();

        let keys: Vec<BlockKey> = self
            .blocks
            .iter()
            .map(|entry| entry.key().clone())
            .collect();
        for batch in keys.chunks(MUTATION_BATCH_SIZE) {
            let _mutation_guard = self.mutation_lock.lock().expect("mutation lock poisoned");
            for key in batch {
                if let Some(mut owners) = self.blocks.get_mut(key) {
                    let before_count = owners.len();
                    owners.retain(|_, owner| {
                        now.saturating_duration_since(owner.key_register_time) <= max_age
                    });
                    self.record_owner_change(before_count, owners.len());
                    stats.removed_owners += before_count - owners.len();
                }
                if self.blocks.get(key).is_some_and(|owners| owners.is_empty()) {
                    self.blocks.remove_if(key, |_, owners| owners.is_empty());
                    stats.removed_keys += 1;
                }
            }
        }

        stats
    }

    /// Latest incrementally maintained stored-owner redundancy distribution.
    pub fn redundancy_snapshot(&self) -> RedundancySnapshot {
        self.redundancy.snapshot()
    }

    pub fn entry_count(&self) -> u64 {
        self.blocks.len() as u64
    }

    pub fn owner_count(&self) -> u64 {
        self.owners_total.load(Ordering::Relaxed)
    }

    pub fn node_counts(&self) -> (u64, u64) {
        let now = Instant::now();
        let mut active = 0;
        let mut stale = 0;
        for node in &self.nodes {
            let age = now.duration_since(node.last_seen);
            if age <= self.config.node_stale_after {
                active += 1;
            } else {
                stale += 1;
            }
        }
        (active, stale)
    }

    #[allow(
        dead_code,
        reason = "maintenance API reserved for explicit store cleanup"
    )]
    pub fn invalidate_all(&self) {
        let mut mutation = self.mutation_lock.lock().expect("mutation lock poisoned");
        self.blocks.clear();
        self.nodes.clear();
        mutation.reconcile_needed = false;
        self.redundancy.reset();
        self.owners_total.store(0, Ordering::Relaxed);
    }

    fn touch_node_session(&self, node: &str, node_id: Uuid) -> Result<(), StoreError> {
        let Some(mut record) = self.nodes.get_mut(node) else {
            warn!(
                "MetaServer metadata write rejected unknown node: node={} node_id={}",
                node, node_id
            );
            return Err(StoreError::UnknownNode);
        };
        if record.node_id != node_id {
            warn!(
                "MetaServer metadata write rejected stale session: node={} current_node_id={} rejected_node_id={}",
                node, record.node_id, node_id
            );
            return Err(StoreError::StaleSession);
        }
        record.last_seen = Instant::now();
        Ok(())
    }

    /// Called under mutation_lock. Counts follow retained records, so a session
    /// change cannot invalidate a subsequent mutation's before-count.
    fn record_owner_change(&self, before: usize, after: usize) {
        if after > before {
            self.owners_total
                .fetch_add((after - before) as u64, Ordering::Relaxed);
        } else if before > after {
            self.owners_total
                .fetch_sub((before - after) as u64, Ordering::Relaxed);
        }
        self.redundancy.adjust(before as u64, after as u64);
    }

    /// Lifecycle and explicit node removal share one accounting boundary.
    /// Callers must hold mutation_lock throughout the scan.
    fn retain_owners(&self, mut keep: impl FnMut(&Arc<str>, &OwnerRecord) -> bool) -> SweepStats {
        let mut stats = SweepStats::default();
        self.blocks.retain(|_, owners| {
            let before = owners.len();
            owners.retain(|node, owner| keep(node, owner));
            self.record_owner_change(before, owners.len());
            stats.removed_owners += before - owners.len();
            if owners.is_empty() {
                stats.removed_keys += 1;
                return false;
            }
            true
        });
        stats
    }

    fn is_owner_visible(&self, node: &Arc<str>, owner: &OwnerRecord, now: Instant) -> bool {
        let Some(record) = self.nodes.get(node.as_ref()) else {
            return false;
        };
        let node_age = now.duration_since(record.last_seen);
        record.node_id == owner.node_id && node_age <= self.config.node_stale_after
    }
}

impl Default for BlockHashStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn manual_cleanup_fixture() -> BlockHashStore {
        let store = BlockHashStore::new();
        let a = heartbeat_node(&store, "a");
        let b = heartbeat_node(&store, "b");
        store
            .insert_hashes("ns", &[vec![1], vec![2], vec![3]], "a", a)
            .unwrap();
        for mut owners in store.blocks.iter_mut() {
            owners.get_mut("a").unwrap().key_register_time =
                Instant::now() - Duration::from_secs(MANUAL_CLEANUP_AGE_SECS + 1);
        }
        // Refresh one old owner and add a fresh replica to another old key.
        store.insert_hashes("ns", &[vec![3]], "a", a).unwrap();
        store.insert_hashes("ns", &[vec![2]], "b", b).unwrap();
        store
    }

    fn assert_stored_counts(store: &BlockHashStore) {
        let mut expected = RedundancySnapshot::default();
        for owners in &store.blocks {
            match owners.len() {
                0 => panic!("empty key retained"),
                1 => expected.keys_1 += 1,
                2 => expected.keys_2 += 1,
                3 => expected.keys_3 += 1,
                _ => expected.keys_4plus += 1,
            }
            expected.copies += owners.len() as u64;
        }
        assert_eq!(store.owner_count(), expected.copies);
        assert_eq!(store.redundancy_snapshot(), expected);
    }

    #[test]
    fn stale_node_survives_sweep_and_recovers_without_insert() {
        let store = BlockHashStore::new();
        let id = heartbeat_node(&store, "node-a");
        store.insert_hashes("ns", &[vec![1]], "node-a", id).unwrap();
        store.nodes.get_mut("node-a").unwrap().last_seen = Instant::now() - Duration::from_secs(31);
        assert!(store.query_prefix("ns", &[vec![1]]).is_empty());

        // A held block shard must not prevent the healthy-node-table fast path.
        let guard = store
            .blocks
            .get_mut(&BlockKey::new("ns".into(), vec![1]))
            .unwrap();
        std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::channel();
            let store_ref = &store;
            scope.spawn(move || tx.send(store_ref.sweep_expired()).unwrap());
            let result = rx.recv_timeout(Duration::from_secs(2));
            drop(guard);
            assert_eq!(result.unwrap(), SweepStats::default());
        });
        assert_eq!(store.node_counts(), (0, 1));
        store.heartbeat_node("node-a", id).unwrap();
        assert_eq!(store.query_prefix("ns", &[vec![1]]).len(), 1);
        assert_stored_counts(&store);
    }

    #[test]
    fn node_ttl_cleanup_preserves_old_blocks_on_active_nodes() {
        let store = BlockHashStore::new();
        let a = heartbeat_node(&store, "a");
        let b = heartbeat_node(&store, "b");
        store
            .insert_hashes("ns", &[vec![1], vec![2]], "a", a)
            .unwrap();
        store.insert_hashes("ns", &[vec![1]], "b", b).unwrap();
        let old = Instant::now() - store.config.ttl - Duration::from_secs(1);
        for mut owners in store.blocks.iter_mut() {
            for owner in owners.values_mut() {
                owner.key_register_time = old;
            }
        }
        store.nodes.get_mut("a").unwrap().last_seen = old;
        assert_eq!(
            store.sweep_expired(),
            SweepStats {
                removed_owners: 2,
                removed_keys: 1,
                removed_nodes: 1,
            }
        );
        assert_eq!(
            store.query_prefix("ns", &[vec![1]])[0].nodes[0].as_ref(),
            "b"
        );
        assert_stored_counts(&store);
    }

    #[test]
    fn mutations_between_takeovers_and_reconciliation_keep_counts_consistent() {
        let store = BlockHashStore::new();
        let mut a = heartbeat_node(&store, "a");
        let b = heartbeat_node(&store, "b");
        store
            .insert_hashes("ns", &[vec![1], vec![2], vec![3]], "a", a)
            .unwrap();
        store
            .insert_hashes("ns", &[vec![1], vec![2], vec![3]], "b", b)
            .unwrap();
        for _ in 0..2 {
            store.nodes.get_mut("a").unwrap().last_seen = Instant::now() - Duration::from_secs(31);
            a = heartbeat_node(&store, "a");
            store
                .insert_hashes("ns", &[vec![1], vec![1]], "a", a)
                .unwrap();
            store.remove_hashes("ns", &[vec![1]], "b", b).unwrap();
            assert_stored_counts(&store);
        }
        assert_eq!(store.sweep_expired().removed_owners, 2);
        assert_eq!(store.owner_count(), 3);
        assert_stored_counts(&store);
        // A later takeover must set a new flag, even after one sweep consumed it.
        store.nodes.get_mut("a").unwrap().last_seen = Instant::now() - Duration::from_secs(31);
        let new_a = heartbeat_node(&store, "a");
        assert_ne!(a, new_a);
        assert_eq!(store.sweep_expired().removed_owners, 1);
        assert_stored_counts(&store);
        assert!(store.sweep_expired().is_empty());
    }

    #[test]
    fn unregister_after_takeover_cleans_historical_owners() {
        let store = BlockHashStore::new();
        let old = heartbeat_node(&store, "a");
        store
            .insert_hashes("ns", &[vec![1], vec![2]], "a", old)
            .unwrap();
        store.nodes.get_mut("a").unwrap().last_seen = Instant::now() - Duration::from_secs(31);
        let current = heartbeat_node(&store, "a");
        store.insert_hashes("ns", &[vec![1]], "a", current).unwrap();
        assert_eq!(
            store.unregister_node("a", old),
            Err(StoreError::StaleSession)
        );
        assert_eq!(store.unregister_node("a", current), Ok(2));
        assert!(store.sweep_expired().is_empty());
        assert_stored_counts(&store);
        assert_eq!(store.node_counts(), (0, 0));
    }

    fn heartbeat_node(store: &BlockHashStore, node: &str) -> Uuid {
        let node_id = Uuid::new_v4();
        store.heartbeat_node(node, node_id).unwrap();
        node_id
    }

    #[test]
    fn test_register_insert_and_query() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let node = "10.0.0.1:50055";
        let node_id = heartbeat_node(&store, node);

        let hashes = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]];

        let reclaimable = store
            .insert_hashes(namespace, &hashes, node, node_id)
            .unwrap();
        assert!(reclaimable.is_empty());

        let existing = store.query_prefix(namespace, &hashes);
        assert_eq!(existing.len(), 3);
        for entry in &existing {
            assert_eq!(entry.nodes.len(), 1);
            assert_eq!(entry.nodes[0].as_ref(), node);
        }

        let mixed_hashes = vec![vec![1, 2, 3, 4], vec![99, 99, 99, 99], vec![5, 6, 7, 8]];
        let existing = store.query_prefix(namespace, &mixed_hashes);
        assert_eq!(existing.len(), 1);

        let existing = store.query_prefix("other-namespace", &hashes);
        assert_eq!(existing.len(), 0);
    }

    #[test]
    fn test_multi_owner() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let hash = vec![1, 2, 3, 4];
        let node_a = "node-a:50055";
        let node_b = "node-b:50055";
        let node_a_id = heartbeat_node(&store, node_a);
        let node_b_id = heartbeat_node(&store, node_b);

        store
            .insert_hashes(namespace, std::slice::from_ref(&hash), node_a, node_a_id)
            .unwrap();
        store
            .insert_hashes(namespace, std::slice::from_ref(&hash), node_b, node_b_id)
            .unwrap();

        let existing = store.query_prefix(namespace, std::slice::from_ref(&hash));
        assert_eq!(existing.len(), 1);
        assert_eq!(existing[0].nodes.len(), 2);

        let mut node_names: Vec<&str> = existing[0].nodes.iter().map(|n| n.as_ref()).collect();
        node_names.sort();
        assert_eq!(node_names, vec!["node-a:50055", "node-b:50055"]);
    }

    #[test]
    fn insert_returns_only_new_third_owner_hashes() {
        let store = BlockHashStore::new();
        let node_a = heartbeat_node(&store, "node-a:50055");
        let node_b = heartbeat_node(&store, "node-b:50055");
        let node_c = heartbeat_node(&store, "node-c:50055");
        let node_d = heartbeat_node(&store, "node-d:50055");
        let hashes = vec![vec![1], vec![2], vec![1]];

        assert_eq!(
            store
                .insert_hashes("ns", &hashes, "node-a:50055", node_a)
                .unwrap(),
            Vec::<Vec<u8>>::new()
        );
        assert_eq!(
            store
                .insert_hashes("ns", &hashes, "node-b:50055", node_b)
                .unwrap(),
            Vec::<Vec<u8>>::new()
        );
        assert_eq!(
            store
                .insert_hashes("ns", &hashes, "node-c:50055", node_c)
                .unwrap(),
            vec![vec![1], vec![2]]
        );
        assert_eq!(
            store
                .insert_hashes("ns", &hashes, "node-c:50055", node_c)
                .unwrap(),
            Vec::<Vec<u8>>::new()
        );
        assert_eq!(
            store
                .insert_hashes("ns", &hashes, "node-d:50055", node_d)
                .unwrap(),
            vec![vec![1], vec![2]]
        );
    }

    #[test]
    fn stale_owner_does_not_count_toward_reclaim_hint() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::from_secs(30),
            ttl: Duration::from_secs(60),
        });
        let hash = vec![1];
        let node_a = heartbeat_node(&store, "node-a");
        let node_b = heartbeat_node(&store, "node-b");
        let node_c = heartbeat_node(&store, "node-c");
        let node_d = heartbeat_node(&store, "node-d");

        store
            .insert_hashes("ns", std::slice::from_ref(&hash), "node-a", node_a)
            .unwrap();
        store.nodes.get_mut("node-a").unwrap().last_seen = Instant::now() - Duration::from_secs(31);

        store
            .insert_hashes("ns", std::slice::from_ref(&hash), "node-b", node_b)
            .unwrap();
        assert!(
            store
                .insert_hashes("ns", std::slice::from_ref(&hash), "node-c", node_c)
                .unwrap()
                .is_empty(),
            "the stale owner must not make node-c the third live owner"
        );
        assert_eq!(
            store
                .insert_hashes("ns", std::slice::from_ref(&hash), "node-d", node_d)
                .unwrap(),
            vec![hash]
        );
    }

    #[test]
    fn new_session_at_same_address_counts_as_new_owner() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::from_secs(30),
            ttl: Duration::from_secs(60),
        });
        let hash = vec![1];
        let old_node_a = heartbeat_node(&store, "node-a");
        let node_b = heartbeat_node(&store, "node-b");
        let node_c = heartbeat_node(&store, "node-c");

        for (node, node_id) in [
            ("node-a", old_node_a),
            ("node-b", node_b),
            ("node-c", node_c),
        ] {
            store
                .insert_hashes("ns", std::slice::from_ref(&hash), node, node_id)
                .unwrap();
        }

        store.nodes.get_mut("node-a").unwrap().last_seen = Instant::now() - Duration::from_secs(31);
        let new_node_a = heartbeat_node(&store, "node-a");
        assert_ne!(old_node_a, new_node_a);

        assert_eq!(
            store
                .insert_hashes("ns", std::slice::from_ref(&hash), "node-a", new_node_a,)
                .unwrap(),
            vec![hash]
        );
    }

    #[test]
    fn test_empty_store() {
        let store = BlockHashStore::new();
        assert_eq!(store.entry_count(), 0);
        assert_eq!(store.owner_count(), 0);
        assert_eq!(store.node_counts(), (0, 0));

        let hashes = vec![vec![1, 2, 3]];
        let existing = store.query_prefix("any-namespace", &hashes);
        assert_eq!(existing.len(), 0);
    }

    #[test]
    fn test_remove_own_blocks() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let hash = vec![1, 2, 3, 4];
        let node = "node-a";
        let node_id = heartbeat_node(&store, node);

        store
            .insert_hashes(namespace, std::slice::from_ref(&hash), node, node_id)
            .unwrap();

        let removed = store
            .remove_hashes(namespace, std::slice::from_ref(&hash), node, node_id)
            .unwrap();
        assert_eq!(removed, 1);
        assert_eq!(store.query_prefix(namespace, &[hash]).len(), 0);
    }

    #[test]
    fn test_remove_other_nodes_block_is_noop() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let hash = vec![1, 2, 3, 4];
        let node_b_id = heartbeat_node(&store, "node-b");
        let node_a_id = heartbeat_node(&store, "node-a");

        store
            .insert_hashes(namespace, std::slice::from_ref(&hash), "node-b", node_b_id)
            .unwrap();

        let removed = store
            .remove_hashes(namespace, std::slice::from_ref(&hash), "node-a", node_a_id)
            .unwrap();
        assert_eq!(removed, 0);
        assert_eq!(store.query_prefix(namespace, &[hash]).len(), 1);
    }

    #[test]
    fn test_remove_one_owner_keeps_others() {
        let store = BlockHashStore::new();
        let hash = vec![1, 2, 3];
        let node_a_id = heartbeat_node(&store, "node-a");
        let node_b_id = heartbeat_node(&store, "node-b");

        store
            .insert_hashes("ns", std::slice::from_ref(&hash), "node-a", node_a_id)
            .unwrap();
        store
            .insert_hashes("ns", std::slice::from_ref(&hash), "node-b", node_b_id)
            .unwrap();

        let removed = store
            .remove_hashes("ns", std::slice::from_ref(&hash), "node-a", node_a_id)
            .unwrap();
        assert_eq!(removed, 1);

        let existing = store.query_prefix("ns", std::slice::from_ref(&hash));
        assert_eq!(existing.len(), 1);
        assert_eq!(existing[0].nodes.len(), 1);
        assert_eq!(existing[0].nodes[0].as_ref(), "node-b");
    }

    #[test]
    fn test_remove_nonexistent_is_noop() {
        let store = BlockHashStore::new();
        let node_id = heartbeat_node(&store, "node-a");
        let removed = store
            .remove_hashes("ns", &[vec![9, 9, 9]], "node-a", node_id)
            .unwrap();
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_heartbeat_rejects_active_different_session() {
        let store = BlockHashStore::new();
        let old_id = heartbeat_node(&store, "node-a");
        let new_id = Uuid::new_v4();
        assert_ne!(old_id, new_id);

        let err = store.heartbeat_node("node-a", new_id).unwrap_err();
        assert_eq!(err, StoreError::StaleSession);
    }

    #[test]
    fn test_query_filters_superseded_node_session() {
        let store = BlockHashStore::new();
        let old_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", old_id)
            .unwrap();
        store.nodes.get_mut("node-a").unwrap().last_seen =
            Instant::now() - Duration::from_secs(DEFAULT_NODE_STALE_SECS + 1);
        let new_id = heartbeat_node(&store, "node-a");
        assert_ne!(old_id, new_id);

        let existing = store.query_prefix("ns", &[vec![1]]);
        assert!(existing.is_empty());
        assert_eq!(store.owner_count(), 1);
    }

    #[test]
    fn test_sweep_reconciles_superseded_owner_before_ttl() {
        let store = BlockHashStore::new();
        let old_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", old_id)
            .unwrap();
        store.nodes.get_mut("node-a").unwrap().last_seen =
            Instant::now() - Duration::from_secs(DEFAULT_NODE_STALE_SECS + 1);
        let new_id = heartbeat_node(&store, "node-a");
        assert_ne!(old_id, new_id);

        let removed = store.sweep_expired();
        assert_eq!(removed.removed_owners, 1);
        assert_eq!(removed.removed_keys, 1);
        assert_eq!(removed.removed_nodes, 0);
        assert_eq!(store.owner_count(), 0);
        assert!(store.query_prefix("ns", &[vec![1]]).is_empty());
    }

    #[test]
    fn test_sweep_removes_superseded_owner_after_key_purge_age() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::ZERO,
            ttl: Duration::ZERO,
        });
        let old_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", old_id)
            .unwrap();
        store.nodes.get_mut("node-a").unwrap().last_seen = Instant::now() - Duration::from_secs(1);
        let new_id = heartbeat_node(&store, "node-a");
        assert_ne!(old_id, new_id);

        let removed = store.sweep_expired();
        assert_eq!(
            removed,
            SweepStats {
                removed_owners: 1,
                removed_keys: 1,
                removed_nodes: 1,
            }
        );
        assert_eq!(store.owner_count(), 0);
    }

    #[test]
    fn test_late_unregister_old_session_does_not_remove_current_node() {
        let store = BlockHashStore::new();
        let old_id = heartbeat_node(&store, "node-a");
        store.nodes.get_mut("node-a").unwrap().last_seen =
            Instant::now() - Duration::from_secs(DEFAULT_NODE_STALE_SECS + 1);
        let new_id = heartbeat_node(&store, "node-a");
        assert_ne!(old_id, new_id);

        store
            .insert_hashes("ns", &[vec![1]], "node-a", new_id)
            .unwrap();

        let err = store.unregister_node("node-a", old_id).unwrap_err();
        assert_eq!(err, StoreError::StaleSession);

        let existing = store.query_prefix("ns", &[vec![1]]);
        assert_eq!(existing.len(), 1);
        assert_eq!(existing[0].nodes[0].as_ref(), "node-a");
    }

    #[test]
    fn test_unregistered_insert_is_rejected() {
        let store = BlockHashStore::new();
        let err = store
            .insert_hashes("ns", &[vec![1]], "node-a", Uuid::new_v4())
            .unwrap_err();
        assert_eq!(err, StoreError::UnknownNode);
    }

    #[test]
    fn test_query_filters_stale_node() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::from_millis(1),
            ttl: Duration::from_secs(60),
        });
        let node_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", node_id)
            .unwrap();
        std::thread::sleep(Duration::from_millis(2));

        let existing = store.query_prefix("ns", &[vec![1]]);
        assert!(existing.is_empty());
        assert_eq!(store.entry_count(), 1);
        assert_eq!(store.owner_count(), 1);
    }

    #[test]
    fn takeover_reconciles_old_session_owners_on_next_sweep() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::ZERO,
            ttl: Duration::from_secs(60),
        });
        let old_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", old_id)
            .unwrap();
        store.nodes.get_mut("node-a").unwrap().last_seen = Instant::now() - Duration::from_secs(1);

        let new_id = heartbeat_node(&store, "node-a");
        assert!(store.query_prefix("ns", &[vec![1]]).is_empty());
        assert_eq!(store.owner_count(), 1);

        let stats = store.sweep_expired();
        assert_eq!(stats.removed_owners, 1);
        assert_eq!(stats.removed_nodes, 0);
        assert_eq!(store.owner_count(), 0);
        assert!(store.query_prefix("ns", &[vec![1]]).is_empty());
        assert_eq!(store.nodes.get("node-a").unwrap().node_id, new_id);
    }

    #[test]
    fn test_insert_refreshes_node_liveness() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::from_secs(60),
            ttl: Duration::from_secs(60),
        });
        let node_id = heartbeat_node(&store, "node-a");
        store.nodes.get_mut("node-a").unwrap().last_seen = Instant::now() - Duration::from_secs(61);

        store
            .insert_hashes("ns", &[vec![1]], "node-a", node_id)
            .unwrap();

        assert_eq!(store.node_counts(), (1, 0));
        let existing = store.query_prefix("ns", &[vec![1]]);
        assert_eq!(existing.len(), 1);
        assert_eq!(existing[0].nodes[0].as_ref(), "node-a");
    }

    #[test]
    fn test_remove_refreshes_node_liveness() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::from_secs(60),
            ttl: Duration::from_secs(60),
        });
        let node_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", node_id)
            .unwrap();
        store.nodes.get_mut("node-a").unwrap().last_seen = Instant::now() - Duration::from_secs(61);

        store
            .remove_hashes("ns", &[vec![2]], "node-a", node_id)
            .unwrap();

        assert_eq!(store.node_counts(), (1, 0));
    }

    #[test]
    fn test_sweep_expired() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::ZERO,
            ttl: Duration::ZERO,
        });
        let node_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1], vec![2]], "node-a", node_id)
            .unwrap();
        assert_eq!(store.entry_count(), 2);
        assert_eq!(store.owner_count(), 2);

        let removed = store.sweep_expired();
        assert_eq!(
            removed,
            SweepStats {
                removed_owners: 2,
                removed_keys: 2,
                removed_nodes: 1,
            }
        );
        assert_eq!(store.entry_count(), 0);
        assert_eq!(store.owner_count(), 0);
        assert_eq!(store.node_counts(), (0, 0));
    }

    #[test]
    fn test_sweep_keeps_fresh_entries() {
        let store = BlockHashStore::new();
        let node_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1], vec![2]], "node-a", node_id)
            .unwrap();

        let removed = store.sweep_expired();
        assert_eq!(removed, SweepStats::default());
        assert_eq!(store.entry_count(), 2);
    }

    #[test]
    fn test_sweep_keeps_old_owner_on_active_node() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::from_secs(60),
            ttl: Duration::from_secs(1),
        });
        let node_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", node_id)
            .unwrap();
        store
            .blocks
            .get_mut(&BlockKey::new("ns".to_string(), vec![1]))
            .unwrap()
            .get_mut("node-a")
            .unwrap()
            .key_register_time = Instant::now() - Duration::from_secs(2);

        let removed = store.sweep_expired();
        assert_eq!(removed, SweepStats::default());
        assert_eq!(store.owner_count(), 1);
        assert_eq!(store.query_prefix("ns", &[vec![1]]).len(), 1);
    }

    #[test]
    fn test_remove_owners_older_than_removes_only_expired_owners() {
        let store = BlockHashStore::new();
        let node_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1], vec![2]], "node-a", node_id)
            .unwrap();
        store
            .blocks
            .get_mut(&BlockKey::new("ns".to_string(), vec![1]))
            .unwrap()
            .get_mut("node-a")
            .unwrap()
            .key_register_time = Instant::now() - Duration::from_secs(3601);

        let removed = store.remove_owners_older_than(Duration::from_secs(3600));
        assert_eq!(
            removed,
            SweepStats {
                removed_owners: 1,
                removed_keys: 1,
                removed_nodes: 0,
            }
        );
        assert_eq!(store.owner_count(), 1);
        assert_eq!(store.entry_count(), 1);
    }

    #[test]
    fn test_concurrent_insert_and_remove() {
        use std::sync::Arc;

        let store = Arc::new(BlockHashStore::new());
        let hash = vec![1, 2, 3, 4];

        for _ in 0..100 {
            let node_a_id = heartbeat_node(&store, "node-a");
            let node_b_id = heartbeat_node(&store, "node-b");
            store
                .insert_hashes("ns", std::slice::from_ref(&hash), "node-a", node_a_id)
                .unwrap();

            let store_a = Arc::clone(&store);
            let store_b = Arc::clone(&store);
            let hash_a = hash.clone();
            let hash_b = hash.clone();

            std::thread::scope(|s| {
                s.spawn(|| {
                    store_a
                        .remove_hashes("ns", &[hash_a], "node-a", node_a_id)
                        .unwrap();
                });
                s.spawn(|| {
                    store_b
                        .insert_hashes("ns", &[hash_b], "node-b", node_b_id)
                        .unwrap();
                });
            });

            let existing = store.query_prefix("ns", std::slice::from_ref(&hash));
            assert_eq!(existing.len(), 1, "key must exist after concurrent ops");
            assert!(
                existing[0].nodes.iter().any(|n| n.as_ref() == "node-b"),
                "node-b must be present"
            );

            store.invalidate_all();
        }
    }

    #[test]
    fn test_invalidate_all() {
        let store = BlockHashStore::new();
        let namespace = "model-test";
        let node = "10.0.0.1:50055";
        let node_id = heartbeat_node(&store, node);

        let hashes = vec![vec![1, 2, 3], vec![4, 5, 6]];
        store
            .insert_hashes(namespace, &hashes, node, node_id)
            .unwrap();

        let existing = store.query_prefix(namespace, &hashes);
        assert_eq!(existing.len(), 2);

        store.invalidate_all();

        let existing = store.query_prefix(namespace, &hashes);
        assert_eq!(existing.len(), 0);
        assert_eq!(store.entry_count(), 0);
        assert_eq!(store.owner_count(), 0);
        assert_eq!(store.node_counts(), (0, 0));
    }

    #[test]
    fn test_redundancy_snapshot_buckets_stored_owners() {
        let store = BlockHashStore::new();
        let a = heartbeat_node(&store, "n-a");
        let b = heartbeat_node(&store, "n-b");
        let c = heartbeat_node(&store, "n-c");
        let d = heartbeat_node(&store, "n-d");

        // 1 owner, 2 owners, and 4 owners across three distinct keys.
        store.insert_hashes("ns", &[vec![1]], "n-a", a).unwrap();
        store.insert_hashes("ns", &[vec![2]], "n-a", a).unwrap();
        store.insert_hashes("ns", &[vec![2]], "n-b", b).unwrap();
        for (node, id) in [("n-a", a), ("n-b", b), ("n-c", c), ("n-d", d)] {
            store.insert_hashes("ns", &[vec![3]], node, id).unwrap();
        }

        store.sweep_expired();

        assert_eq!(
            store.redundancy_snapshot(),
            RedundancySnapshot {
                keys_1: 1,
                keys_2: 1,
                keys_3: 0,
                keys_4plus: 1,
                copies: 1 + 2 + 4,
            }
        );
    }

    #[test]
    fn test_redundancy_snapshot_excludes_superseded_owner() {
        // A block whose only owner is a superseded session is removed by the
        // next reconciliation sweep.
        let store = BlockHashStore::new();
        let old_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", old_id)
            .unwrap();
        store.nodes.get_mut("node-a").unwrap().last_seen =
            Instant::now() - Duration::from_secs(DEFAULT_NODE_STALE_SECS + 1);
        let new_id = heartbeat_node(&store, "node-a");
        assert_ne!(old_id, new_id);

        store.sweep_expired();

        assert_eq!(store.owner_count(), 0);
        assert_eq!(store.entry_count(), 0);
        assert_eq!(store.redundancy_snapshot(), RedundancySnapshot::default());
    }

    #[test]
    fn test_redundancy_counters_follow_owner_mutations() {
        let store = BlockHashStore::new();
        let node_a = heartbeat_node(&store, "node-a");
        let node_b = heartbeat_node(&store, "node-b");
        let hash = vec![1];

        store
            .insert_hashes("ns", std::slice::from_ref(&hash), "node-a", node_a)
            .unwrap();
        assert_eq!(store.redundancy_snapshot().keys_1, 1);

        store
            .insert_hashes("ns", std::slice::from_ref(&hash), "node-b", node_b)
            .unwrap();
        assert_eq!(store.redundancy_snapshot().keys_2, 1);
        assert_eq!(store.redundancy_snapshot().copies, 2);

        store
            .remove_hashes("ns", std::slice::from_ref(&hash), "node-a", node_a)
            .unwrap();
        assert_eq!(store.redundancy_snapshot().keys_1, 1);
        assert_eq!(store.redundancy_snapshot().keys_2, 0);
        assert_eq!(store.redundancy_snapshot().copies, 1);

        store
            .blocks
            .get_mut(&BlockKey::new("ns".to_string(), hash))
            .unwrap()
            .get_mut("node-b")
            .unwrap()
            .key_register_time = Instant::now() - Duration::from_secs(3601);
        let cleanup = store.remove_owners_older_than(Duration::from_secs(3600));
        assert_eq!(cleanup.removed_owners, 1);
        assert_eq!(store.redundancy_snapshot(), RedundancySnapshot::default());
    }

    #[test]
    fn heartbeat_does_not_recount_unswept_stale_owners() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::from_secs(30),
            ..StoreConfig::default()
        });
        let node_a = heartbeat_node(&store, "node-a");
        let node_b = heartbeat_node(&store, "node-b");
        let hash = vec![7];
        store
            .insert_hashes("ns", std::slice::from_ref(&hash), "node-a", node_a)
            .unwrap();
        store
            .insert_hashes("ns", std::slice::from_ref(&hash), "node-b", node_b)
            .unwrap();
        let before = store.redundancy_snapshot();

        store.nodes.get_mut("node-b").unwrap().last_seen = Instant::now() - Duration::from_secs(31);
        store.heartbeat_node("node-a", node_a).unwrap();

        assert_eq!(store.redundancy_snapshot(), before);
    }
}
