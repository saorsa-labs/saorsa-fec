//! Chunk registry for managing chunk lifecycle and reference counting
//!
//! This module tracks all chunks in the system, their reference counts,
//! and manages chunk lifecycle for garbage collection.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::metadata::ChunkReference;

/// Registry for tracking chunk metadata and references
#[derive(Debug, Clone)]
pub struct ChunkRegistry {
    /// All chunks indexed by their ID
    chunks: HashMap<[u8; 32], ChunkMetadata>,
}

/// Information about a chunk
#[derive(Debug, Clone)]
pub struct ChunkInfo {
    /// Chunk identifier
    pub id: ChunkId,
    /// Associated data ID
    pub data_id: DataId,
    /// Size of the chunk
    pub size: usize,
    /// Encrypted size
    pub encrypted_size: usize,
    /// Share IDs for this chunk
    pub share_ids: Vec<ShareId>,
    /// Encryption key hash
    pub encryption_key_hash: [u8; 32],
    /// Creation time
    pub created_at: std::time::SystemTime,
}

use crate::types::{ChunkId, DataId, ShareId};

impl ChunkRegistry {
    /// Create a new chunk registry
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    /// Increment reference counts for multiple chunks
    pub fn increment_refs(&mut self, chunk_refs: &[ChunkReference]) -> Result<()> {
        for chunk_ref in chunk_refs {
            self.increment_ref(&chunk_ref.chunk_id)?;

            // Update size if not already recorded
            if let Some(metadata) = self.chunks.get_mut(&chunk_ref.chunk_id) {
                if metadata.size == 0 {
                    metadata.size = chunk_ref.size;
                }
            }
        }
        Ok(())
    }

    /// Increment reference count for a single chunk
    pub fn increment_ref(&mut self, chunk_id: &[u8; 32]) -> Result<()> {
        let metadata = self
            .chunks
            .entry(*chunk_id)
            .or_insert_with(|| ChunkMetadata::new(0));

        metadata.ref_count = metadata
            .ref_count
            .checked_add(1)
            .context("Reference count overflow")?;

        Ok(())
    }

    /// Decrement reference counts for multiple chunks
    /// Returns chunks that are now unreferenced
    pub fn decrement_refs(&mut self, chunk_ids: &[[u8; 32]]) -> Result<Vec<[u8; 32]>> {
        let mut unreferenced = Vec::new();

        for chunk_id in chunk_ids {
            if self.decrement_ref(chunk_id)? == 0 {
                unreferenced.push(*chunk_id);
            }
        }

        Ok(unreferenced)
    }

    /// Decrement reference count for a single chunk
    /// Returns the new reference count
    pub fn decrement_ref(&mut self, chunk_id: &[u8; 32]) -> Result<u32> {
        let metadata = self
            .chunks
            .get_mut(chunk_id)
            .context("Chunk not found in registry")?;

        if metadata.ref_count == 0 {
            anyhow::bail!("Cannot decrement reference count below zero");
        }

        metadata.ref_count -= 1;

        // Update last accessed time
        metadata.update_access_time();

        Ok(metadata.ref_count)
    }

    /// Get all unreferenced chunks
    pub fn get_unreferenced(&self) -> Vec<[u8; 32]> {
        self.chunks
            .iter()
            .filter_map(|(id, metadata)| {
                if metadata.ref_count == 0 {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get chunk metadata
    pub fn get_metadata(&self, chunk_id: &[u8; 32]) -> Option<&ChunkMetadata> {
        self.chunks.get(chunk_id)
    }

    /// Get chunk size
    pub fn get_chunk_size(&self, chunk_id: &[u8; 32]) -> Option<u32> {
        self.chunks.get(chunk_id).map(|m| m.size)
    }

    /// Get reference count for a chunk
    pub fn get_ref_count(&self, chunk_id: &[u8; 32]) -> Option<u32> {
        self.chunks.get(chunk_id).map(|m| m.ref_count)
    }

    /// Check if a chunk exists in the registry
    pub fn contains(&self, chunk_id: &[u8; 32]) -> bool {
        self.chunks.contains_key(chunk_id)
    }

    /// Add version that uses a chunk
    pub fn add_version_ref(&mut self, chunk_id: &[u8; 32], version_id: [u8; 32]) -> Result<()> {
        let metadata = self
            .chunks
            .get_mut(chunk_id)
            .context("Chunk not found in registry")?;

        metadata.versions_using.insert(version_id);
        Ok(())
    }

    /// Remove version reference from a chunk
    pub fn remove_version_ref(&mut self, chunk_id: &[u8; 32], version_id: &[u8; 32]) -> Result<()> {
        let metadata = self
            .chunks
            .get_mut(chunk_id)
            .context("Chunk not found in registry")?;

        metadata.versions_using.remove(version_id);
        Ok(())
    }

    /// Get all versions using a chunk
    pub fn get_versions_using(&self, chunk_id: &[u8; 32]) -> Option<&HashSet<[u8; 32]>> {
        self.chunks.get(chunk_id).map(|m| &m.versions_using)
    }

    /// Remove chunk from registry (after successful deletion)
    pub fn remove_chunk(&mut self, chunk_id: &[u8; 32]) -> Result<()> {
        let metadata = self
            .chunks
            .remove(chunk_id)
            .context("Chunk not found in registry")?;

        if metadata.ref_count > 0 {
            // Restore it - this is a safety check
            self.chunks.insert(*chunk_id, metadata);
            anyhow::bail!("Cannot remove chunk with non-zero reference count");
        }

        Ok(())
    }

    /// Get total size of all chunks
    pub fn total_size(&self) -> u64 {
        self.chunks.values().map(|m| m.size as u64).sum()
    }

    /// Get total size of referenced chunks
    pub fn referenced_size(&self) -> u64 {
        self.chunks
            .values()
            .filter(|m| m.ref_count > 0)
            .map(|m| m.size as u64)
            .sum()
    }

    /// Get total size of unreferenced chunks
    pub fn unreferenced_size(&self) -> u64 {
        self.chunks
            .values()
            .filter(|m| m.ref_count == 0)
            .map(|m| m.size as u64)
            .sum()
    }

    /// Register a new chunk
    pub fn register_chunk(&mut self, chunk_info: ChunkInfo) {
        let metadata = ChunkMetadata::new(chunk_info.size as u32);
        self.chunks.insert(chunk_info.encryption_key_hash, metadata);
    }

    /// Unregister a chunk
    pub fn unregister_chunk(&mut self, _chunk_id: &ChunkId) {
        // Simplified implementation - would need proper mapping
    }

    /// Get chunk information by ID
    pub fn get_chunk(&self, _chunk_id: &ChunkId) -> Option<ChunkInfo> {
        // Simplified implementation - would need proper mapping
        None
    }

    /// Get statistics about the registry
    pub fn stats(&self) -> RegistryStats {
        RegistryStats {
            total_chunks: self.chunks.len(),
            referenced_chunks: self.chunks.values().filter(|m| m.ref_count > 0).count(),
            unreferenced_chunks: self.chunks.values().filter(|m| m.ref_count == 0).count(),
            total_size: self.total_size(),
            referenced_size: self.referenced_size(),
            unreferenced_size: self.unreferenced_size(),
        }
    }

    /// Export registry to persistent storage
    pub fn export(&self) -> Result<Vec<u8>> {
        postcard::to_stdvec(&self.chunks).context("Failed to serialize chunk registry")
    }

    /// Import registry from persistent storage
    pub fn import(data: &[u8]) -> Result<Self> {
        let chunks = postcard::from_bytes(data).context("Failed to deserialize chunk registry")?;

        Ok(Self { chunks })
    }

    /// Merge another registry into this one
    pub fn merge(&mut self, other: &ChunkRegistry) -> Result<()> {
        for (chunk_id, other_metadata) in &other.chunks {
            match self.chunks.get_mut(chunk_id) {
                Some(metadata) => {
                    // Merge metadata - take maximum ref count
                    metadata.ref_count = metadata.ref_count.max(other_metadata.ref_count);
                    metadata
                        .versions_using
                        .extend(&other_metadata.versions_using);
                }
                None => {
                    // Add new chunk
                    self.chunks.insert(*chunk_id, other_metadata.clone());
                }
            }
        }
        Ok(())
    }
}

impl Default for ChunkRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata for a single chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Reference count for this chunk
    pub ref_count: u32,
    /// Size of chunk in bytes
    pub size: u32,
    /// Set of version IDs that reference this chunk
    pub versions_using: HashSet<[u8; 32]>,
    /// Unix timestamp when first seen locally
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_seen_locally: Option<u64>,
    /// Unix timestamp when last accessed locally
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_accessed_locally: Option<u64>,
}

impl ChunkMetadata {
    /// Create new chunk metadata
    pub fn new(size: u32) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .ok();

        Self {
            ref_count: 0,
            size,
            versions_using: HashSet::new(),
            first_seen_locally: now,
            last_accessed_locally: now,
        }
    }

    /// Update last accessed time
    pub fn update_access_time(&mut self) {
        self.last_accessed_locally = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .ok();
    }

    /// Check if chunk is orphaned (no versions using it)
    pub fn is_orphaned(&self) -> bool {
        self.versions_using.is_empty() && self.ref_count == 0
    }

    /// Get age in seconds since first seen
    pub fn age_seconds(&self) -> Option<u64> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .ok()?;

        self.first_seen_locally
            .map(|first| now.saturating_sub(first))
    }

    /// Get time since last access in seconds
    pub fn idle_seconds(&self) -> Option<u64> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .ok()?;

        self.last_accessed_locally
            .map(|last| now.saturating_sub(last))
    }
}

/// Statistics about the chunk registry
#[derive(Debug, Clone)]
pub struct RegistryStats {
    /// Total number of chunks
    pub total_chunks: usize,
    /// Number of referenced chunks
    pub referenced_chunks: usize,
    /// Number of unreferenced chunks
    pub unreferenced_chunks: usize,
    /// Total size in bytes
    pub total_size: u64,
    /// Size of referenced chunks
    pub referenced_size: u64,
    /// Size of unreferenced chunks
    pub unreferenced_size: u64,
}

impl RegistryStats {
    /// Get percentage of space that could be reclaimed
    pub fn reclaimable_percentage(&self) -> f64 {
        if self.total_size == 0 {
            0.0
        } else {
            (self.unreferenced_size as f64 / self.total_size as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_registry_basic() {
        let mut registry = ChunkRegistry::new();
        let chunk_id = [1u8; 32];

        // Initial state
        assert_eq!(registry.get_ref_count(&chunk_id), None);

        // Increment reference
        registry.increment_ref(&chunk_id).unwrap();
        assert_eq!(registry.get_ref_count(&chunk_id), Some(1));

        // Increment again
        registry.increment_ref(&chunk_id).unwrap();
        assert_eq!(registry.get_ref_count(&chunk_id), Some(2));

        // Decrement
        let new_count = registry.decrement_ref(&chunk_id).unwrap();
        assert_eq!(new_count, 1);

        // Decrement to zero
        let new_count = registry.decrement_ref(&chunk_id).unwrap();
        assert_eq!(new_count, 0);

        // Check unreferenced
        let unreferenced = registry.get_unreferenced();
        assert_eq!(unreferenced.len(), 1);
        assert_eq!(unreferenced[0], chunk_id);
    }

    #[test]
    fn test_chunk_registry_versions() {
        let mut registry = ChunkRegistry::new();
        let chunk_id = [1u8; 32];
        let version1 = [10u8; 32];
        let version2 = [20u8; 32];

        registry.increment_ref(&chunk_id).unwrap();
        registry.add_version_ref(&chunk_id, version1).unwrap();
        registry.add_version_ref(&chunk_id, version2).unwrap();

        let versions = registry.get_versions_using(&chunk_id).unwrap();
        assert_eq!(versions.len(), 2);
        assert!(versions.contains(&version1));
        assert!(versions.contains(&version2));

        registry.remove_version_ref(&chunk_id, &version1).unwrap();
        let versions = registry.get_versions_using(&chunk_id).unwrap();
        assert_eq!(versions.len(), 1);
    }

    #[test]
    fn test_chunk_registry_stats() {
        let mut registry = ChunkRegistry::new();

        // Add some chunks
        let chunk_refs = vec![
            ChunkReference::new([1u8; 32], 0, 0, 1024),
            ChunkReference::new([2u8; 32], 0, 1, 2048),
            ChunkReference::new([3u8; 32], 0, 2, 512),
        ];

        registry.increment_refs(&chunk_refs).unwrap();

        // Decrement one to make it unreferenced
        registry.decrement_ref(&[3u8; 32]).unwrap();

        let stats = registry.stats();
        assert_eq!(stats.total_chunks, 3);
        assert_eq!(stats.referenced_chunks, 2);
        assert_eq!(stats.unreferenced_chunks, 1);
        assert_eq!(stats.total_size, 3584);
        assert_eq!(stats.referenced_size, 3072);
        assert_eq!(stats.unreferenced_size, 512);
    }

    #[test]
    fn test_chunk_registry_export_import() {
        let mut registry = ChunkRegistry::new();

        // Add some data
        registry.increment_ref(&[1u8; 32]).unwrap();
        registry.increment_ref(&[2u8; 32]).unwrap();
        registry.add_version_ref(&[1u8; 32], [10u8; 32]).unwrap();

        // Export
        let data = registry.export().unwrap();

        // Import
        let imported = ChunkRegistry::import(&data).unwrap();

        assert_eq!(imported.get_ref_count(&[1u8; 32]), Some(1));
        assert_eq!(imported.get_ref_count(&[2u8; 32]), Some(1));
        assert!(imported
            .get_versions_using(&[1u8; 32])
            .unwrap()
            .contains(&[10u8; 32]));
    }

    #[test]
    fn test_chunk_removal_safety() {
        let mut registry = ChunkRegistry::new();
        let chunk_id = [1u8; 32];

        registry.increment_ref(&chunk_id).unwrap();

        // Should fail to remove chunk with non-zero ref count
        let result = registry.remove_chunk(&chunk_id);
        assert!(result.is_err());

        // Decrement to zero
        registry.decrement_ref(&chunk_id).unwrap();

        // Should succeed now
        let result = registry.remove_chunk(&chunk_id);
        assert!(result.is_ok());
        assert!(!registry.contains(&chunk_id));
    }
}
