use opentelemetry::metrics::{Counter, Histogram, ObservableGauge};
use opentelemetry::{KeyValue, global};
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Instant;
use tonic::Status;

use crate::store::{BlockHashStore, SweepStats};

// ---------------------------------------------------------------------------
// Store gauges
// ---------------------------------------------------------------------------

struct StoreGaugeHandles {
    _entries: ObservableGauge<u64>,
    _owners: ObservableGauge<u64>,
    _nodes: ObservableGauge<u64>,
}

static STORE_GAUGES: OnceLock<StoreGaugeHandles> = OnceLock::new();

/// Register observable gauges backed by the given store.
pub fn register_store_gauges(store: &Arc<BlockHashStore>) {
    let s = Arc::clone(store);
    STORE_GAUGES.get_or_init(|| {
        let meter = global::meter("pegaflow_metaserver");
        let entries_store = Arc::clone(&s);
        let entries = meter
            .u64_observable_gauge("pegaflow_metaserver_store_entries")
            .with_description("Number of unique block keys in the store")
            .with_callback(move |observer| {
                observer.observe(entries_store.entry_count(), &[]);
            })
            .build();
        let owners_store = Arc::clone(&s);
        let owners = meter
            .u64_observable_gauge("pegaflow_metaserver_block_owners")
            .with_description("Number of node ownership records across all block keys")
            .with_callback(move |observer| {
                observer.observe(owners_store.owner_count(), &[]);
            })
            .build();
        let nodes_store = Arc::clone(&s);
        let nodes = meter
            .u64_observable_gauge("pegaflow_metaserver_nodes")
            .with_description("Number of registered nodes by liveness state")
            .with_callback(move |observer| {
                let (active, stale) = nodes_store.node_counts();
                observer.observe(active, &[KeyValue::new("state", "active")]);
                observer.observe(stale, &[KeyValue::new("state", "stale")]);
            })
            .build();
        StoreGaugeHandles {
            _entries: entries,
            _owners: owners,
            _nodes: nodes,
        }
    });
}

// ---------------------------------------------------------------------------
// Node lifecycle sweep counters
// ---------------------------------------------------------------------------

struct SweepMetrics {
    removed_owners: Counter<u64>,
    removed_keys: Counter<u64>,
    removed_nodes: Counter<u64>,
}

static SWEEP_METRICS: LazyLock<SweepMetrics> = LazyLock::new(|| {
    let meter = global::meter("pegaflow_metaserver");
    SweepMetrics {
        removed_owners: meter
            .u64_counter("pegaflow_metaserver_sweep_removed_owners")
            .with_description("Total node ownership records removed by lifecycle sweep")
            .build(),
        removed_keys: meter
            .u64_counter("pegaflow_metaserver_sweep_removed_keys")
            .with_description("Total block keys removed by lifecycle sweep")
            .build(),
        removed_nodes: meter
            .u64_counter("pegaflow_metaserver_sweep_removed_nodes")
            .with_description("Total node records removed by lifecycle sweep")
            .build(),
    }
});

pub fn record_sweep(stats: SweepStats) {
    SWEEP_METRICS
        .removed_owners
        .add(stats.removed_owners as u64, &[]);
    SWEEP_METRICS
        .removed_keys
        .add(stats.removed_keys as u64, &[]);
    SWEEP_METRICS
        .removed_nodes
        .add(stats.removed_nodes as u64, &[]);
}

// ---------------------------------------------------------------------------
// RPC metrics
// ---------------------------------------------------------------------------

struct RpcMetrics {
    request_count: Counter<u64>,
    request_duration: Histogram<f64>,
}

impl RpcMetrics {
    fn new() -> Self {
        let meter = global::meter("pegaflow_metaserver_rpc");
        let request_count = meter
            .u64_counter("pegaflow_metaserver_rpc_requests")
            .with_description("Total RPC requests handled by pegaflow metaserver")
            .build();
        let request_duration = meter
            .f64_histogram("pegaflow_metaserver_rpc_duration")
            .with_description("RPC latency in seconds")
            .with_unit("s")
            .with_boundaries(
                [
                    0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0,
                ]
                .into(),
            )
            .build();
        Self {
            request_count,
            request_duration,
        }
    }

    fn record(&self, method: &'static str, status: &str, duration: f64) {
        let labels = [
            KeyValue::new("method", method.to_string()),
            KeyValue::new("status", status.to_string()),
        ];
        self.request_count.add(1, &labels);
        self.request_duration.record(duration, &labels);
    }
}

static RPC_METRICS: LazyLock<RpcMetrics> = LazyLock::new(RpcMetrics::new);

pub fn record_rpc_result<T>(method: &'static str, result: &Result<T, Status>, start: Instant) {
    let status = match result {
        Ok(_) => "ok".to_string(),
        Err(status) => status.code().to_string(),
    };
    let duration = start.elapsed().as_secs_f64();
    RPC_METRICS.record(method, &status, duration);
}
