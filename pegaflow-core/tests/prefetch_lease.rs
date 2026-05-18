//! Query lease protocol tests.
//!
//! Verifies the scheduler->worker contract: query_prefetch returns an opaque
//! lease that owns ready blocks, and load consumes one lease share per worker.

mod common;

use common::*;
use pegaflow_core::LoadState;

/// vLLM worker must not load after the scheduler releases the query lease.
#[tokio::test]
async fn load_requires_query_prefetch() {
    let env = TestEnvBuilder::new("test-load-needs-query", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;

    // Query and immediately release — no lease held.
    assert_eq!(env.count_hits_then_release(&hashes).await, 4);

    let released = env.assert_all_hit_lease(&hashes).await;
    env.release(&released);
    env.expect_load_error(released, hashes.len(), "query lease is unknown or expired");
}

/// One scheduler query lease is consumed by one registered worker load.
#[tokio::test]
async fn query_then_load_consumes_reservation_budget() {
    let env = TestEnvBuilder::new("test-query-lease", "test-ns")
        .layer("layer_0", 4, 1024)
        .world_size(2)
        .build();
    let hashes = env.hashes(22);

    env.save_and_wait(&hashes).await;
    let lease = env.assert_all_hit_lease(&hashes).await;

    env.data().zero_gpu();
    env.load_to_gpu(lease, hashes.len()).await;
    env.data().assert_gpu_matches_expected();

    let block_ids: Vec<i32> = (0..hashes.len() as i32).collect();
    let layer_names: Vec<&str> = env.layers.iter().map(|l| l.name.as_str()).collect();
    let load_state = LoadState::new().expect("create LoadState");
    let err = env
        .engine
        .batch_load_kv_blocks_multi_layer(
            &env.instance_id,
            0,
            0,
            load_state.shm_name(),
            &layer_names,
            &[(lease, block_ids)],
        )
        .expect_err("second load should fail");
    assert!(
        err.to_string()
            .contains("query lease is unknown or expired")
    );
}

/// A worker may load a strict prefix of a query lease when the scheduler leaves
/// the boundary block for local recomputation.
#[tokio::test]
async fn load_can_consume_query_lease_prefix() {
    let env = TestEnvBuilder::new("test-query-lease-prefix", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(33);

    env.save_and_wait(&hashes).await;
    let lease = env.assert_all_hit_lease(&hashes).await;

    env.data().zero_gpu();
    env.load_to_gpu(lease, hashes.len() - 1).await;

    let gpu = env.data().copy_to_host();
    let expected = env.data().expected();
    let loaded_bytes = (hashes.len() - 1) * env.data().block_size;
    assert_eq!(&gpu[..loaded_bytes], &expected[..loaded_bytes]);
    assert!(
        gpu[loaded_bytes..].iter().all(|&b| b == 0),
        "prefix load must not write the unrequested suffix block"
    );
}
