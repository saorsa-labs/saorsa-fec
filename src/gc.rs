//! Garbage collection for unreferenced chunks
//!
//! This module provides configurable retention policies and safe garbage
//! collection of unreferenced chunks.

use anyhow::Result;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

use crate::chunk_registry::ChunkRegistry;
use crate::storage::{Cid, StorageBackend};
use crate::version::VersionNode;

/// Retention policy for garbage collection
#[derive(Clone, Serialize, Deserialize, Default)]
pub enum RetentionPolicy {
    /// Keep all versions and chunks
    #[default]
    KeepAll,
    /// Keep only the last N versions
    KeepLastN(usize),
    /// Keep specific tagged versions
    KeepTagged(HashSet<[u8; 32]>),
    /// Keep versions newer than a certain age (seconds)
    KeepRecent(u64),
    /// Custom policy (not serializable)
    #[serde(skip)]
    Custom(Arc<dyn Fn(&VersionNode) -> bool + Send + Sync>),
}

impl std::fmt::Debug for RetentionPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KeepAll => write!(f, "KeepAll"),
            Self::KeepLastN(n) => write!(f, "KeepLastN({})", n),
            Self::KeepTagged(tags) => write!(f, "KeepTagged({:?})", tags),
            Self::KeepRecent(secs) => write!(f, "KeepRecent({})", secs),
            Self::Custom(_) => write!(f, "Custom(<function>)"),
        }
    }
}

/// Garbage collector for managing chunk lifecycle
pub struct GarbageCollector {
    /// Retention policy to apply
    pub policy: RetentionPolicy,
    /// Reference to chunk registry
    chunk_registry: Arc<RwLock<ChunkRegistry>>,
    /// Storage backend for chunk deletion
    storage: Arc<dyn StorageBackend>,
}

impl GarbageCollector {
    /// Create a new garbage collector
    pub fn new(
        policy: RetentionPolicy,
        chunk_registry: Arc<RwLock<ChunkRegistry>>,
        storage: Arc<dyn StorageBackend>,
    ) -> Self {
        Self {
            policy,
            chunk_registry,
            storage,
        }
    }

    /// Mark and sweep to identify chunks for collection
    /// Returns list of chunk IDs that can be safely deleted
    pub fn mark_sweep(&self) -> Vec<[u8; 32]> {
        let registry = self.chunk_registry.read();

        match &self.policy {
            RetentionPolicy::KeepAll => {
                // Never delete anything
                Vec::new()
            }
            _ => {
                // Get all unreferenced chunks
                let unreferenced = registry.get_unreferenced();

                // Additional filtering based on policy
                unreferenced
                    .into_iter()
                    .filter(|chunk_id| self.should_collect_chunk(chunk_id))
                    .collect()
            }
        }
    }

    /// Collect (delete) specified chunks
    pub async fn collect(&self, chunk_ids: Vec<[u8; 32]>) -> Result<CollectionReport> {
        let mut report = CollectionReport::new();

        for chunk_id in chunk_ids {
            // Double-check that chunk is still unreferenced
            {
                let registry = self.chunk_registry.read();
                if let Some(count) = registry.get_ref_count(&chunk_id) {
                    if count > 0 {
                        report.skipped += 1;
                        continue;
                    }
                } else {
                    // Chunk not in registry anymore
                    report.skipped += 1;
                    continue;
                }
            }

            // Attempt to delete from storage
            let cid = Cid::new(chunk_id);
            match self.storage.delete_shard(&cid).await {
                Ok(()) => {
                    // Remove from registry after successful deletion
                    let mut registry = self.chunk_registry.write();
                    if let Err(e) = registry.remove_chunk(&chunk_id) {
                        tracing::warn!("Failed to remove chunk from registry: {}", e);
                    }

                    report.collected += 1;
                    report.bytes_freed += registry.get_chunk_size(&chunk_id).unwrap_or(0) as u64;
                }
                Err(e) => {
                    tracing::error!("Failed to delete chunk {:?}: {}", chunk_id, e);
                    report.failed += 1;
                }
            }
        }

        Ok(report)
    }

    /// Run a full garbage collection cycle
    pub async fn run(&self) -> Result<CollectionReport> {
        let chunks_to_collect = self.mark_sweep();

        if chunks_to_collect.is_empty() {
            Ok(CollectionReport::new())
        } else {
            self.collect(chunks_to_collect).await
        }
    }

    /// Check if a specific chunk should be collected
    fn should_collect_chunk(&self, chunk_id: &[u8; 32]) -> bool {
        let registry = self.chunk_registry.read();

        // Get chunk metadata
        let metadata = match registry.get_metadata(chunk_id) {
            Some(m) => m,
            None => return false, // Don't collect if we don't have metadata
        };

        // Never collect chunks with references
        if metadata.ref_count > 0 {
            return false;
        }

        // Apply age-based policies
        match &self.policy {
            RetentionPolicy::KeepRecent(max_age_seconds) => {
                if let Some(age) = metadata.age_seconds() {
                    age > *max_age_seconds
                } else {
                    false // Keep if we can't determine age
                }
            }
            _ => true, // Other policies handled at version level
        }
    }

    /// Update retention policy
    pub fn set_policy(&mut self, policy: RetentionPolicy) {
        self.policy = policy;
    }

    /// Estimate space that can be reclaimed
    pub fn estimate_reclaimable(&self) -> u64 {
        let chunks_to_collect = self.mark_sweep();
        let registry = self.chunk_registry.read();

        chunks_to_collect
            .iter()
            .filter_map(|id| registry.get_chunk_size(id))
            .map(|size| size as u64)
            .sum()
    }

    /// Perform a dry run without actually deleting
    pub fn dry_run(&self) -> GCDryRun {
        let chunks_to_collect = self.mark_sweep();
        let registry = self.chunk_registry.read();

        let total_size: u64 = chunks_to_collect
            .iter()
            .filter_map(|id| registry.get_chunk_size(id))
            .map(|size| size as u64)
            .sum();

        GCDryRun {
            chunks_to_delete: chunks_to_collect.len(),
            bytes_to_free: total_size,
            chunk_ids: chunks_to_collect,
        }
    }
}

/// Report from a garbage collection run
#[derive(Debug, Clone, Default)]
pub struct CollectionReport {
    /// Number of chunks successfully collected
    pub collected: usize,
    /// Number of chunks skipped (became referenced)
    pub skipped: usize,
    /// Number of chunks that failed to delete
    pub failed: usize,
    /// Total bytes freed
    pub bytes_freed: u64,
    /// Time taken in milliseconds
    pub duration_ms: u64,
}

impl CollectionReport {
    /// Create a new empty report
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if collection was successful
    pub fn is_successful(&self) -> bool {
        self.failed == 0
    }

    /// Get total chunks processed
    pub fn total_processed(&self) -> usize {
        self.collected + self.skipped + self.failed
    }
}

/// Dry run results
#[derive(Debug, Clone)]
pub struct GCDryRun {
    /// Number of chunks that would be deleted
    pub chunks_to_delete: usize,
    /// Bytes that would be freed
    pub bytes_to_free: u64,
    /// Actual chunk IDs that would be deleted
    pub chunk_ids: Vec<[u8; 32]>,
}

/// Garbage collection scheduler
pub struct GCScheduler {
    /// Garbage collector instance
    gc: Arc<GarbageCollector>,
    /// Minimum time between collections (seconds)
    min_interval: u64,
    /// Minimum reclaimable space before triggering (bytes)
    min_reclaimable: u64,
    /// Last collection timestamp
    last_run: Option<u64>,
}

impl GCScheduler {
    /// Create a new scheduler
    pub fn new(gc: Arc<GarbageCollector>, min_interval: u64, min_reclaimable: u64) -> Self {
        Self {
            gc,
            min_interval,
            min_reclaimable,
            last_run: None,
        }
    }

    /// Check if garbage collection should run
    pub fn should_run(&self) -> bool {
        // Check time since last run
        if let Some(last) = self.last_run {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);

            if now - last < self.min_interval {
                return false;
            }
        }

        // Check reclaimable space
        self.gc.estimate_reclaimable() >= self.min_reclaimable
    }

    /// Run garbage collection if needed
    pub async fn run_if_needed(&mut self) -> Result<Option<CollectionReport>> {
        if !self.should_run() {
            return Ok(None);
        }

        let report = self.gc.run().await?;

        self.last_run = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        );

        Ok(Some(report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EncryptionMode, FecError, FileMetadata, GcReport, Shard, ShardHeader, StorageStats,
    };
    use async_trait::async_trait;

    // Mock storage backend for testing
    struct MockStorage {
        deleted: Arc<RwLock<Vec<[u8; 32]>>>,
        fail_on: HashSet<[u8; 32]>,
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                deleted: Arc::new(RwLock::new(Vec::new())),
                fail_on: HashSet::new(),
            }
        }

        #[allow(dead_code)]
        fn with_failures(mut self, chunks: Vec<[u8; 32]>) -> Self {
            self.fail_on = chunks.into_iter().collect();
            self
        }
    }

    #[async_trait]
    impl StorageBackend for MockStorage {
        async fn put_shard(&self, _cid: &Cid, _shard: &Shard) -> Result<(), FecError> {
            Ok(())
        }

        async fn get_shard(&self, _cid: &Cid) -> Result<Shard, FecError> {
            let header = ShardHeader::new(EncryptionMode::Convergent, (3, 2), 0, [0u8; 32]);
            Ok(Shard::new(header, vec![]))
        }

        async fn delete_shard(&self, cid: &Cid) -> Result<(), FecError> {
            if self.fail_on.contains(cid.as_bytes()) {
                return Err(FecError::Backend("Mock deletion failure".to_string()));
            }
            self.deleted.write().push(*cid.as_bytes());
            Ok(())
        }

        async fn has_shard(&self, _cid: &Cid) -> Result<bool, FecError> {
            Ok(false)
        }

        async fn list_shards(&self) -> Result<Vec<Cid>, FecError> {
            Ok(vec![])
        }

        async fn put_metadata(&self, _metadata: &FileMetadata) -> Result<(), FecError> {
            Ok(())
        }

        async fn get_metadata(&self, _file_id: &[u8; 32]) -> Result<FileMetadata, FecError> {
            Err(FecError::Backend("Mock metadata not found".to_string()))
        }

        async fn delete_metadata(&self, _file_id: &[u8; 32]) -> Result<(), FecError> {
            Ok(())
        }

        async fn list_metadata(&self) -> Result<Vec<FileMetadata>, FecError> {
            Ok(vec![])
        }

        async fn stats(&self) -> Result<StorageStats, FecError> {
            Ok(StorageStats {
                total_shards: 0,
                total_size: 0,
                metadata_count: 0,
                unreferenced_shards: 0,
            })
        }

        async fn garbage_collect(&self) -> Result<GcReport, FecError> {
            Ok(GcReport {
                shards_deleted: 0,
                bytes_freed: 0,
                duration_ms: 0,
            })
        }
    }

    #[tokio::test]
    async fn test_gc_keep_all_policy() {
        let registry = Arc::new(RwLock::new(ChunkRegistry::new()));
        let storage = Arc::new(MockStorage::new());

        // Add unreferenced chunk
        {
            let mut reg = registry.write();
            reg.increment_ref(&[1u8; 32]).unwrap();
            reg.decrement_ref(&[1u8; 32]).unwrap();
        }

        let gc = GarbageCollector::new(RetentionPolicy::KeepAll, registry.clone(), storage);

        let chunks = gc.mark_sweep();
        assert_eq!(chunks.len(), 0); // KeepAll policy keeps everything
    }

    #[tokio::test]
    async fn test_gc_collection() {
        let registry = Arc::new(RwLock::new(ChunkRegistry::new()));
        let storage = Arc::new(MockStorage::new());

        // Add unreferenced chunks
        {
            let mut reg = registry.write();
            for i in 1..=3 {
                reg.increment_ref(&[i; 32]).unwrap();
                reg.decrement_ref(&[i; 32]).unwrap();
            }
        }

        let gc = GarbageCollector::new(
            RetentionPolicy::KeepLastN(0), // Keep nothing
            registry.clone(),
            storage.clone(),
        );

        let report = gc.run().await.unwrap();
        assert_eq!(report.collected, 3);
        assert_eq!(report.failed, 0);

        // Verify chunks were deleted
        let deleted = storage.deleted.read();
        assert_eq!(deleted.len(), 3);
    }

    #[tokio::test]
    async fn test_gc_dry_run() {
        let registry = Arc::new(RwLock::new(ChunkRegistry::new()));
        let storage = Arc::new(MockStorage::new());

        // Add unreferenced chunks with sizes
        {
            let mut reg = registry.write();
            use crate::metadata::ChunkReference;

            let chunks = vec![
                ChunkReference::new([1u8; 32], 0, 0, 1024),
                ChunkReference::new([2u8; 32], 0, 1, 2048),
            ];

            reg.increment_refs(&chunks).unwrap();
            reg.decrement_refs(&[[1u8; 32], [2u8; 32]]).unwrap();
        }

        let gc = GarbageCollector::new(RetentionPolicy::KeepLastN(0), registry, storage);

        let dry_run = gc.dry_run();
        assert_eq!(dry_run.chunks_to_delete, 2);
        assert_eq!(dry_run.bytes_to_free, 3072);
    }

    #[tokio::test]
    async fn test_gc_scheduler() {
        let registry = Arc::new(RwLock::new(ChunkRegistry::new()));
        let storage = Arc::new(MockStorage::new());

        let gc = Arc::new(GarbageCollector::new(
            RetentionPolicy::KeepLastN(0),
            registry,
            storage,
        ));

        let scheduler = GCScheduler::new(
            gc, 60,   // 60 seconds min interval
            1024, // 1KB min reclaimable
        );

        // Should not run immediately
        assert!(!scheduler.should_run());
    }
}
