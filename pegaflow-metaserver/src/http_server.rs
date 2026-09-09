use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{
    Json, Router,
    routing::{get, post},
};
use log::{info, warn};
use prometheus::{Registry, TextEncoder};
use serde::Serialize;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Notify;

use crate::store::{BlockHashStore, MANUAL_CLEANUP_AGE_SECS};

#[derive(Clone)]
struct AppState {
    prometheus_registry: Registry,
    store: Arc<BlockHashStore>,
}

#[derive(Debug, Serialize)]
struct CleanupResponse {
    removed_owners: usize,
    removed_keys: usize,
}

async fn health_handler() -> &'static str {
    "ok"
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = state.prometheus_registry.gather();
    (
        StatusCode::OK,
        encoder
            .encode_to_string(&metric_families)
            .unwrap_or_else(|e| format!("# Error encoding metrics: {e}")),
    )
}

async fn cleanup_expired_blocks_handler(State(state): State<AppState>) -> Json<CleanupResponse> {
    let stats = state
        .store
        .remove_owners_older_than(std::time::Duration::from_secs(MANUAL_CLEANUP_AGE_SECS));
    Json(CleanupResponse {
        removed_owners: stats.removed_owners,
        removed_keys: stats.removed_keys,
    })
}

fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route(
            "/admin/cleanup-expired-blocks",
            post(cleanup_expired_blocks_handler),
        )
        .with_state(state)
}

pub async fn start_http_server(
    addr: std::net::SocketAddr,
    prometheus_registry: Registry,
    store: Arc<BlockHashStore>,
    shutdown: Arc<Notify>,
) -> Result<tokio::task::JoinHandle<()>, std::io::Error> {
    let listener = TcpListener::bind(addr).await?;

    let state = AppState {
        prometheus_registry,
        store,
    };

    let app = app(state);

    info!(
        "Starting HTTP server on {} (/health, /metrics, POST /admin/cleanup-expired-blocks)",
        addr
    );

    let handle = tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                shutdown.notified().await;
            })
            .await
        {
            warn!("HTTP server stopped with error: {err}");
        }
    });

    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[tokio::test]
    async fn cleanup_route_accepts_post_and_returns_stats() {
        let response = app(AppState {
            prometheus_registry: Registry::new(),
            store: Arc::new(BlockHashStore::new()),
        })
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/cleanup-expired-blocks")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .as_ref(),
            br#"{"removed_owners":0,"removed_keys":0}"#
        );
    }
}
