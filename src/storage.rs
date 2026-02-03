//! Storage backend abstraction for shard storage
//!
//! This module provides a trait for different storage implementations
//! (local filesystem, memory, network, multi-backend) that work with
//! the v0.3 shard format with 96-byte headers and CID-based addressing.

use crate::config::EncryptionMode;
use crate::FecError;
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Content Identifier (CID) for addressing shards
/// Uses BLAKE3 hash for content-addressable storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Cid([u8; 32]);

impl Cid {
    /// Create CID from raw bytes
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Create CID from data using BLAKE3
    pub fn from_data(data: &[u8]) -> Self {
        let hash = blake3::hash(data);
        Self(*hash.as_bytes())
    }

    /// Get raw bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl From<[u8; 32]> for Cid {
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl From<blake3::Hash> for Cid {
    fn from(hash: blake3::Hash) -> Self {
        Self(*hash.as_bytes())
    }
}

/// Shard header (106 bytes fixed size) for v0.3
///
/// Note: With postcard serialization, the actual serialized data is smaller
/// but we pad to 106 bytes for backwards compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardHeader {
    /// Shard format version
    pub version: u8,
    /// Encryption mode used
    pub encryption_mode: EncryptionMode,
    /// FEC parameters (k, n-k)
    pub nspec: (u8, u8),
    /// Encrypted data size
    pub data_size: u32,
    /// Nonce for encryption (32 bytes)
    pub nonce: [u8; 32],
    /// Reserved bytes for future use
    #[serde(with = "serde_bytes")]
    pub reserved: Vec<u8>,
}

impl ShardHeader {
    const SIZE: usize = 106; // Fixed header size for compatibility

    /// Create new shard header
    pub fn new(
        encryption_mode: EncryptionMode,
        nspec: (u8, u8),
        data_size: u32,
        nonce: [u8; 32],
    ) -> Self {
        Self {
            version: 1,
            encryption_mode,
            nspec,
            data_size,
            nonce,
            reserved: vec![0u8; 55],
        }
    }

    /// Serialize to bytes (padded to fixed size)
    pub fn to_bytes(&self) -> Result<[u8; Self::SIZE], FecError> {
        let serialized = postcard::to_stdvec(self)
            .map_err(|e| FecError::Backend(format!("Failed to serialize header: {}", e)))?;

        // Pad to fixed size for backwards compatibility
        let mut result = [0u8; Self::SIZE];
        if serialized.len() > Self::SIZE {
            return Err(FecError::Backend(format!(
                "Header too large: {} > {}",
                serialized.len(),
                Self::SIZE
            )));
        }
        result[..serialized.len()].copy_from_slice(&serialized);
        // Store the actual serialized length in the last byte for parsing
        result[Self::SIZE - 1] = serialized.len() as u8;
        Ok(result)
    }

    /// Deserialize from bytes (handles padded format)
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FecError> {
        if bytes.len() != Self::SIZE {
            return Err(FecError::Backend(format!(
                "Invalid header size: expected {}, got {}",
                Self::SIZE,
                bytes.len()
            )));
        }
        // Read actual length from last byte
        let actual_len = bytes[Self::SIZE - 1] as usize;
        if actual_len == 0 || actual_len > Self::SIZE - 1 {
            // Fallback: try parsing the whole buffer (legacy bincode format)
            return postcard::from_bytes(bytes)
                .map_err(|e| FecError::Backend(format!("Failed to deserialize header: {}", e)));
        }
        postcard::from_bytes(&bytes[..actual_len])
            .map_err(|e| FecError::Backend(format!("Failed to deserialize header: {}", e)))
    }
}

/// Complete shard with header and encrypted data
#[derive(Debug, Clone)]
pub struct Shard {
    /// 96-byte header
    pub header: ShardHeader,
    /// Encrypted data payload
    pub data: Vec<u8>,
}

impl Shard {
    /// Create new shard
    pub fn new(header: ShardHeader, data: Vec<u8>) -> Self {
        Self { header, data }
    }

    /// Get CID for this shard (computed over header + data)
    pub fn cid(&self) -> Result<Cid, FecError> {
        let header_bytes = self.header.to_bytes()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(&header_bytes);
        hasher.update(&self.data);
        Ok(Cid::from(hasher.finalize()))
    }

    /// Serialize shard to bytes (header + data)
    pub fn to_bytes(&self) -> Result<Vec<u8>, FecError> {
        let header_bytes = self.header.to_bytes()?;
        let mut result = Vec::with_capacity(ShardHeader::SIZE + self.data.len());
        result.extend_from_slice(&header_bytes);
        result.extend_from_slice(&self.data);
        Ok(result)
    }

    /// Deserialize shard from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FecError> {
        if bytes.len() < ShardHeader::SIZE {
            return Err(FecError::Backend(
                "Insufficient data for shard header".to_string(),
            ));
        }

        let header = ShardHeader::from_bytes(&bytes[..ShardHeader::SIZE])?;
        let data = bytes[ShardHeader::SIZE..].to_vec();

        Ok(Self { header, data })
    }
}

/// Chunk metadata as specified in v0.3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMeta {
    /// FEC parameters (k, n-k)
    pub nspec: (u8, u8),
    /// Encryption mode used
    pub mode: EncryptionMode,
    /// CIDs of all shards for this chunk
    pub shard_ids: Vec<String>,
}

impl ChunkMeta {
    /// Create new chunk metadata
    pub fn new(nspec: (u8, u8), mode: EncryptionMode, shard_ids: Vec<String>) -> Self {
        Self {
            nspec,
            mode,
            shard_ids,
        }
    }
}

/// File metadata as specified in v0.3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    /// File identifier
    pub file_id: [u8; 32],
    /// Original file size
    pub file_size: u64,
    /// Chunks comprising this file
    pub chunks: Vec<ChunkMeta>,
    /// Creation timestamp
    pub created_at: u64,
    /// Version number
    pub version: u8,
}

impl FileMetadata {
    /// Create new file metadata
    pub fn new(file_id: [u8; 32], file_size: u64, chunks: Vec<ChunkMeta>) -> Self {
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            file_id,
            file_size,
            chunks,
            created_at,
            version: 1,
        }
    }
}

/// Abstract storage backend interface for v0.3 specification
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Store a shard with the given CID
    async fn put_shard(&self, cid: &Cid, shard: &Shard) -> Result<(), FecError>;

    /// Retrieve a shard by CID
    async fn get_shard(&self, cid: &Cid) -> Result<Shard, FecError>;

    /// Delete a shard by CID
    async fn delete_shard(&self, cid: &Cid) -> Result<(), FecError>;

    /// Check if a shard exists
    async fn has_shard(&self, cid: &Cid) -> Result<bool, FecError>;

    /// List all shard CIDs in storage
    async fn list_shards(&self) -> Result<Vec<Cid>, FecError>;

    /// Store file metadata
    async fn put_metadata(&self, metadata: &FileMetadata) -> Result<(), FecError>;

    /// Retrieve file metadata
    async fn get_metadata(&self, file_id: &[u8; 32]) -> Result<FileMetadata, FecError>;

    /// Delete file metadata
    async fn delete_metadata(&self, file_id: &[u8; 32]) -> Result<(), FecError>;

    /// List all file metadata
    async fn list_metadata(&self) -> Result<Vec<FileMetadata>, FecError>;

    /// Get storage statistics
    async fn stats(&self) -> Result<StorageStats, FecError>;

    /// Run garbage collection
    async fn garbage_collect(&self) -> Result<GcReport, FecError>;
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// Total number of shards
    pub total_shards: u64,
    /// Total storage size in bytes
    pub total_size: u64,
    /// Number of file metadata entries
    pub metadata_count: u64,
    /// Number of unreferenced shards
    pub unreferenced_shards: u64,
}

/// Garbage collection report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcReport {
    /// Number of shards deleted
    pub shards_deleted: u64,
    /// Bytes freed
    pub bytes_freed: u64,
    /// Duration of GC run
    pub duration_ms: u64,
}

/// Local filesystem storage implementation
/// Stores shards and metadata on local filesystem with CID-based addressing
pub struct LocalStorage {
    /// Base directory for shard storage
    base_path: PathBuf,
    /// Directory for metadata storage
    metadata_path: PathBuf,
    /// Number of directory levels for sharding
    shard_levels: usize,
}

impl LocalStorage {
    /// Create a new local storage backend
    pub async fn new(base_path: PathBuf) -> Result<Self, FecError> {
        let metadata_path = base_path.join("metadata");

        fs::create_dir_all(&base_path).await.map_err(FecError::Io)?;
        fs::create_dir_all(&metadata_path)
            .await
            .map_err(FecError::Io)?;

        Ok(Self {
            base_path,
            metadata_path,
            shard_levels: 2, // Use 2 levels of sharding by default
        })
    }

    /// Get the path for a shard based on its CID
    fn shard_path(&self, cid: &Cid) -> PathBuf {
        let hex = cid.to_hex();

        // Create sharded path (e.g., ab/cd/abcdef...)
        let mut path = self.base_path.join("shards");

        for level in 0..self.shard_levels {
            if hex.len() > level * 2 + 2 {
                path = path.join(&hex[level * 2..level * 2 + 2]);
            }
        }

        path.join(format!("{}.shard", hex))
    }

    /// Get the path for file metadata
    fn metadata_file_path(&self, file_id: &[u8; 32]) -> PathBuf {
        let hex = hex::encode(file_id);
        self.metadata_path.join(format!("{}.meta", hex))
    }

    /// Ensure parent directory exists
    async fn ensure_parent(&self, path: &Path) -> Result<(), FecError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await.map_err(FecError::Io)?;
        }
        Ok(())
    }
}

#[async_trait]
impl StorageBackend for LocalStorage {
    async fn put_shard(&self, cid: &Cid, shard: &Shard) -> Result<(), FecError> {
        let path = self.shard_path(cid);

        // Ensure parent directory exists
        self.ensure_parent(&path).await?;

        // Serialize shard to bytes
        let shard_bytes = shard.to_bytes()?;

        // Write shard atomically using temp file
        let temp_path = path.with_extension("tmp");

        let mut file = fs::File::create(&temp_path).await.map_err(FecError::Io)?;

        file.write_all(&shard_bytes).await.map_err(FecError::Io)?;

        file.sync_all().await.map_err(FecError::Io)?;

        // Atomic rename
        fs::rename(temp_path, path).await.map_err(FecError::Io)?;

        Ok(())
    }

    async fn get_shard(&self, cid: &Cid) -> Result<Shard, FecError> {
        let path = self.shard_path(cid);

        let mut file = fs::File::open(&path).await.map_err(|e| {
            FecError::Backend(format!("Failed to open shard file {:?}: {}", path, e))
        })?;

        let mut data = Vec::new();
        file.read_to_end(&mut data).await.map_err(FecError::Io)?;

        Shard::from_bytes(&data)
    }

    async fn delete_shard(&self, cid: &Cid) -> Result<(), FecError> {
        let path = self.shard_path(cid);

        if path.exists() {
            fs::remove_file(path).await.map_err(FecError::Io)?;
        }

        Ok(())
    }

    async fn has_shard(&self, cid: &Cid) -> Result<bool, FecError> {
        let path = self.shard_path(cid);
        Ok(path.exists())
    }

    async fn list_shards(&self) -> Result<Vec<Cid>, FecError> {
        let mut shards = Vec::new();
        let shards_dir = self.base_path.join("shards");

        // Walk directory tree
        let mut stack = vec![shards_dir];

        while let Some(dir) = stack.pop() {
            if !dir.exists() {
                continue;
            }

            let mut entries = fs::read_dir(&dir).await.map_err(|e| {
                FecError::Backend(format!("Failed to read directory {:?}: {}", dir, e))
            })?;

            while let Some(entry) = entries.next_entry().await.map_err(FecError::Io)? {
                let path = entry.path();

                if path.is_dir() {
                    stack.push(path);
                } else if let Some(name) = path.file_name() {
                    if let Some(name_str) = name.to_str() {
                        if name_str.ends_with(".shard") {
                            // Extract hex CID from filename
                            let hex = name_str.trim_end_matches(".shard");
                            if let Ok(cid_bytes) = hex::decode(hex) {
                                if cid_bytes.len() == 32 {
                                    let mut cid_array = [0u8; 32];
                                    cid_array.copy_from_slice(&cid_bytes);
                                    shards.push(Cid::new(cid_array));
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(shards)
    }

    async fn put_metadata(&self, metadata: &FileMetadata) -> Result<(), FecError> {
        let path = self.metadata_file_path(&metadata.file_id);

        let serialized = postcard::to_stdvec(metadata)
            .map_err(|e| FecError::Backend(format!("Failed to serialize metadata: {}", e)))?;

        let temp_path = path.with_extension("tmp");

        let mut file = fs::File::create(&temp_path).await.map_err(FecError::Io)?;

        file.write_all(&serialized).await.map_err(FecError::Io)?;

        file.sync_all().await.map_err(FecError::Io)?;

        // Atomic rename
        fs::rename(temp_path, path).await.map_err(FecError::Io)?;

        Ok(())
    }

    async fn get_metadata(&self, file_id: &[u8; 32]) -> Result<FileMetadata, FecError> {
        let path = self.metadata_file_path(file_id);

        let data = fs::read(&path).await.map_err(|e| {
            FecError::Backend(format!("Failed to read metadata file {:?}: {}", path, e))
        })?;

        postcard::from_bytes(&data)
            .map_err(|e| FecError::Backend(format!("Failed to deserialize metadata: {}", e)))
    }

    async fn delete_metadata(&self, file_id: &[u8; 32]) -> Result<(), FecError> {
        let path = self.metadata_file_path(file_id);

        if path.exists() {
            fs::remove_file(path).await.map_err(FecError::Io)?;
        }

        Ok(())
    }

    async fn list_metadata(&self) -> Result<Vec<FileMetadata>, FecError> {
        let mut metadata_list = Vec::new();

        let mut entries = fs::read_dir(&self.metadata_path)
            .await
            .map_err(FecError::Io)?;

        while let Some(entry) = entries.next_entry().await.map_err(FecError::Io)? {
            let path = entry.path();
            if let Some(name) = path.file_name() {
                if let Some(name_str) = name.to_str() {
                    if name_str.ends_with(".meta") {
                        let data = fs::read(&path).await.map_err(FecError::Io)?;
                        if let Ok(metadata) = postcard::from_bytes::<FileMetadata>(&data) {
                            metadata_list.push(metadata);
                        }
                    }
                }
            }
        }

        Ok(metadata_list)
    }
    async fn stats(&self) -> Result<StorageStats, FecError> {
        let shards = self.list_shards().await?;
        let metadata = self.list_metadata().await?;

        // Calculate total size by reading all shards
        let mut total_size = 0u64;
        for cid in &shards {
            if let Ok(shard) = self.get_shard(cid).await {
                total_size += shard.data.len() as u64 + ShardHeader::SIZE as u64;
            }
        }

        // Count unreferenced shards (shards not referenced in any metadata)
        let mut referenced_cids = std::collections::HashSet::new();
        for meta in &metadata {
            for chunk in &meta.chunks {
                for shard_id in &chunk.shard_ids {
                    if let Ok(cid_bytes) = hex::decode(shard_id) {
                        if cid_bytes.len() == 32 {
                            let mut cid_array = [0u8; 32];
                            cid_array.copy_from_slice(&cid_bytes);
                            referenced_cids.insert(Cid::new(cid_array));
                        }
                    }
                }
            }
        }

        let unreferenced_shards = shards
            .iter()
            .filter(|cid| !referenced_cids.contains(cid))
            .count() as u64;

        Ok(StorageStats {
            total_shards: shards.len() as u64,
            total_size,
            metadata_count: metadata.len() as u64,
            unreferenced_shards,
        })
    }

    async fn garbage_collect(&self) -> Result<GcReport, FecError> {
        let start_time = std::time::Instant::now();
        let mut shards_deleted = 0u64;
        let mut bytes_freed = 0u64;

        // Get all shards and metadata
        let shards = self.list_shards().await?;
        let metadata = self.list_metadata().await?;

        // Build set of referenced shards
        let mut referenced_cids = std::collections::HashSet::new();
        for meta in &metadata {
            for chunk in &meta.chunks {
                for shard_id in &chunk.shard_ids {
                    if let Ok(cid_bytes) = hex::decode(shard_id) {
                        if cid_bytes.len() == 32 {
                            let mut cid_array = [0u8; 32];
                            cid_array.copy_from_slice(&cid_bytes);
                            referenced_cids.insert(Cid::new(cid_array));
                        }
                    }
                }
            }
        }

        // Delete unreferenced shards
        for cid in shards {
            if !referenced_cids.contains(&cid) {
                if let Ok(shard) = self.get_shard(&cid).await {
                    let shard_size = shard.data.len() as u64 + ShardHeader::SIZE as u64;
                    if self.delete_shard(&cid).await.is_ok() {
                        shards_deleted += 1;
                        bytes_freed += shard_size;
                    }
                }
            }
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(GcReport {
            shards_deleted,
            bytes_freed,
            duration_ms,
        })
    }
}

/// In-memory storage implementation for testing and caching
/// Stores shards and metadata in HashMap structures
pub struct MemoryStorage {
    /// In-memory shard storage
    shards: Arc<RwLock<HashMap<Cid, Shard>>>,
    /// In-memory metadata storage
    metadata: Arc<RwLock<HashMap<[u8; 32], FileMetadata>>>,
}

impl MemoryStorage {
    /// Create a new memory storage backend
    pub fn new() -> Self {
        Self {
            shards: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Clear all stored data
    pub fn clear(&self) {
        // Handle poisoned locks by recovering the data
        match self.shards.write() {
            Ok(mut guard) => guard.clear(),
            Err(poisoned) => poisoned.into_inner().clear(),
        }
        match self.metadata.write() {
            Ok(mut guard) => guard.clear(),
            Err(poisoned) => poisoned.into_inner().clear(),
        }
    }

    /// Get the number of stored shards
    pub fn shard_count(&self) -> usize {
        match self.shards.read() {
            Ok(guard) => guard.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }

    /// Get the number of stored metadata entries
    pub fn metadata_count(&self) -> usize {
        match self.metadata.read() {
            Ok(guard) => guard.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl StorageBackend for MemoryStorage {
    async fn put_shard(&self, cid: &Cid, shard: &Shard) -> Result<(), FecError> {
        let mut shards = match self.shards.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        shards.insert(*cid, shard.clone());
        Ok(())
    }

    async fn get_shard(&self, cid: &Cid) -> Result<Shard, FecError> {
        let shards = match self.shards.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        shards
            .get(cid)
            .cloned()
            .ok_or_else(|| FecError::Backend(format!("Shard not found: {}", cid.to_hex())))
    }

    async fn delete_shard(&self, cid: &Cid) -> Result<(), FecError> {
        let mut shards = match self.shards.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        shards.remove(cid);
        Ok(())
    }

    async fn has_shard(&self, cid: &Cid) -> Result<bool, FecError> {
        let shards = match self.shards.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        Ok(shards.contains_key(cid))
    }

    async fn list_shards(&self) -> Result<Vec<Cid>, FecError> {
        let shards = match self.shards.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        Ok(shards.keys().copied().collect())
    }

    async fn put_metadata(&self, metadata: &FileMetadata) -> Result<(), FecError> {
        let mut metadata_store = match self.metadata.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        metadata_store.insert(metadata.file_id, metadata.clone());
        Ok(())
    }

    async fn get_metadata(&self, file_id: &[u8; 32]) -> Result<FileMetadata, FecError> {
        let metadata_store = match self.metadata.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        metadata_store.get(file_id).cloned().ok_or_else(|| {
            FecError::Backend(format!("Metadata not found: {}", hex::encode(file_id)))
        })
    }

    async fn delete_metadata(&self, file_id: &[u8; 32]) -> Result<(), FecError> {
        let mut metadata_store = match self.metadata.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        metadata_store.remove(file_id);
        Ok(())
    }

    async fn list_metadata(&self) -> Result<Vec<FileMetadata>, FecError> {
        let metadata_store = match self.metadata.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        Ok(metadata_store.values().cloned().collect())
    }

    async fn stats(&self) -> Result<StorageStats, FecError> {
        let shards = match self.shards.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let metadata = match self.metadata.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        let total_size: u64 = shards
            .values()
            .map(|shard| shard.data.len() as u64 + ShardHeader::SIZE as u64)
            .sum();

        // Count unreferenced shards
        let mut referenced_cids = std::collections::HashSet::new();
        for meta in metadata.values() {
            for chunk in &meta.chunks {
                for shard_id in &chunk.shard_ids {
                    if let Ok(cid_bytes) = hex::decode(shard_id) {
                        if cid_bytes.len() == 32 {
                            let mut cid_array = [0u8; 32];
                            cid_array.copy_from_slice(&cid_bytes);
                            referenced_cids.insert(Cid::new(cid_array));
                        }
                    }
                }
            }
        }

        let unreferenced_shards = shards
            .keys()
            .filter(|cid| !referenced_cids.contains(cid))
            .count() as u64;

        Ok(StorageStats {
            total_shards: shards.len() as u64,
            total_size,
            metadata_count: metadata.len() as u64,
            unreferenced_shards,
        })
    }

    async fn garbage_collect(&self) -> Result<GcReport, FecError> {
        let start_time = std::time::Instant::now();
        let mut shards_deleted = 0u64;
        let mut bytes_freed = 0u64;

        // Get snapshot of current state
        let shards = match self.shards.read() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        let metadata = match self.metadata.read() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };

        // Build set of referenced shards
        let mut referenced_cids = std::collections::HashSet::new();
        for meta in metadata.values() {
            for chunk in &meta.chunks {
                for shard_id in &chunk.shard_ids {
                    if let Ok(cid_bytes) = hex::decode(shard_id) {
                        if cid_bytes.len() == 32 {
                            let mut cid_array = [0u8; 32];
                            cid_array.copy_from_slice(&cid_bytes);
                            referenced_cids.insert(Cid::new(cid_array));
                        }
                    }
                }
            }
        }

        // Delete unreferenced shards
        let mut shards_write = match self.shards.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        for (cid, shard) in shards {
            if !referenced_cids.contains(&cid) {
                let shard_size = shard.data.len() as u64 + ShardHeader::SIZE as u64;
                shards_write.remove(&cid);
                shards_deleted += 1;
                bytes_freed += shard_size;
            }
        }
        drop(shards_write);

        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(GcReport {
            shards_deleted,
            bytes_freed,
            duration_ms,
        })
    }
}

/// Network storage node endpoint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeEndpoint {
    /// Node address (IP or hostname)
    pub address: String,
    /// Node port
    pub port: u16,
    /// Optional node ID
    pub node_id: Option<[u8; 32]>,
}

/// Network-based storage implementation
pub struct NetworkStorage {
    /// List of storage nodes
    nodes: Vec<NodeEndpoint>,
    /// Replication factor
    replication: usize,
}

impl NetworkStorage {
    /// Create a new network storage backend
    pub fn new(nodes: Vec<NodeEndpoint>, replication: usize) -> Self {
        Self { nodes, replication }
    }

    /// Select nodes for storing a shard
    fn select_nodes(&self, shard_id: &[u8; 32]) -> Vec<&NodeEndpoint> {
        // Simple deterministic selection based on shard ID
        let mut selected = Vec::new();
        let target_count = self.replication.min(self.nodes.len());

        // Use different parts of the hash to select unique nodes
        for i in 0..target_count {
            let hash_offset = i * 4;
            let index = if hash_offset + 3 < shard_id.len() {
                u32::from_le_bytes([
                    shard_id[hash_offset],
                    shard_id[hash_offset + 1],
                    shard_id[hash_offset + 2],
                    shard_id[hash_offset + 3],
                ]) as usize
            } else {
                // Use XOR of all bytes if we run out of unique positions
                shard_id
                    .iter()
                    .enumerate()
                    .map(|(j, &b)| (j + i) * b as usize)
                    .sum::<usize>()
            };

            let mut node_index = index % self.nodes.len();
            let mut attempts = 0;

            // Find a node we haven't selected yet
            while selected.iter().any(|n| *n == &self.nodes[node_index])
                && attempts < self.nodes.len()
            {
                node_index = (node_index + 1) % self.nodes.len();
                attempts += 1;
            }

            if attempts < self.nodes.len() {
                selected.push(&self.nodes[node_index]);
            }
        }

        selected
    }
}

#[async_trait]
impl StorageBackend for NetworkStorage {
    async fn put_shard(&self, cid: &Cid, _shard: &Shard) -> Result<(), FecError> {
        let nodes = self.select_nodes(cid.as_bytes());

        if nodes.is_empty() {
            return Err(FecError::Backend(
                "No nodes available for storage".to_string(),
            ));
        }

        // Store to selected nodes
        let mut success_count = 0;

        for node in nodes {
            // In a real implementation, this would make network calls
            // For now, we'll simulate success
            tracing::debug!(
                "Storing shard {} to node: {}:{}",
                cid.to_hex(),
                node.address,
                node.port
            );
            success_count += 1;
        }

        if success_count == 0 {
            return Err(FecError::Backend(
                "Failed to store shard to any node".to_string(),
            ));
        }

        Ok(())
    }

    async fn get_shard(&self, cid: &Cid) -> Result<Shard, FecError> {
        let nodes = self.select_nodes(cid.as_bytes());

        if let Some(node) = nodes.into_iter().next() {
            // Try to retrieve from the first node
            // In a real implementation, this would make network calls
            tracing::debug!(
                "Retrieving shard {} from node: {}:{}",
                cid.to_hex(),
                node.address,
                node.port
            );

            // Simulate successful retrieval with dummy data
            let header = ShardHeader::new(EncryptionMode::Convergent, (16, 4), 1024, [0u8; 32]);
            let shard = Shard::new(header, vec![0u8; 1024]);
            return Ok(shard);
        }

        Err(FecError::Backend("Shard not found on any node".to_string()))
    }

    async fn delete_shard(&self, cid: &Cid) -> Result<(), FecError> {
        let nodes = self.select_nodes(cid.as_bytes());

        for node in nodes {
            // Delete from each node
            tracing::debug!(
                "Deleting shard {} from node: {}:{}",
                cid.to_hex(),
                node.address,
                node.port
            );
        }

        Ok(())
    }

    async fn has_shard(&self, cid: &Cid) -> Result<bool, FecError> {
        let nodes = self.select_nodes(cid.as_bytes());

        if let Some(node) = nodes.into_iter().next() {
            // Check the first node
            tracing::debug!(
                "Checking shard {} on node: {}:{}",
                cid.to_hex(),
                node.address,
                node.port
            );
            return Ok(true); // Simulate found
        }

        Ok(false)
    }

    async fn list_shards(&self) -> Result<Vec<Cid>, FecError> {
        // This would require querying all nodes and deduplicating
        // For now, return empty list
        Ok(Vec::new())
    }

    async fn put_metadata(&self, _metadata: &FileMetadata) -> Result<(), FecError> {
        // In a real implementation, this would distribute metadata across nodes
        // For now, simulate success
        Ok(())
    }

    async fn get_metadata(&self, _file_id: &[u8; 32]) -> Result<FileMetadata, FecError> {
        // In a real implementation, this would query nodes for metadata
        // For now, return an error
        Err(FecError::Backend(
            "Network metadata retrieval not implemented".to_string(),
        ))
    }

    async fn delete_metadata(&self, _file_id: &[u8; 32]) -> Result<(), FecError> {
        // In a real implementation, this would delete from all nodes
        Ok(())
    }

    async fn list_metadata(&self) -> Result<Vec<FileMetadata>, FecError> {
        // This would require querying all nodes
        Ok(Vec::new())
    }

    async fn stats(&self) -> Result<StorageStats, FecError> {
        // In a real implementation, this would aggregate stats from all nodes
        Ok(StorageStats {
            total_shards: 0,
            total_size: 0,
            metadata_count: 0,
            unreferenced_shards: 0,
        })
    }

    async fn garbage_collect(&self) -> Result<GcReport, FecError> {
        // In a real implementation, this would trigger GC on all nodes
        Ok(GcReport {
            shards_deleted: 0,
            bytes_freed: 0,
            duration_ms: 0,
        })
    }
}

/// Multi-backend storage that combines multiple backends for redundancy and load balancing
/// Implements failover capabilities and load distribution
pub struct MultiStorage {
    /// Ordered list of storage backends (priority order)
    backends: Vec<Arc<dyn StorageBackend>>,
    /// Strategy for backend selection
    strategy: MultiStorageStrategy,
}

/// Strategy for multi-backend operations
#[derive(Debug, Clone)]
pub enum MultiStorageStrategy {
    /// Write to all backends, read from first available
    Redundant,
    /// Load balance across backends
    LoadBalance,
    /// Use primary backend with failover to secondary
    Failover,
}

impl MultiStorage {
    /// Create a new multi-backend storage with redundant strategy
    pub fn new(backends: Vec<Arc<dyn StorageBackend>>) -> Self {
        Self {
            backends,
            strategy: MultiStorageStrategy::Redundant,
        }
    }

    /// Create with specific strategy
    pub fn with_strategy(
        backends: Vec<Arc<dyn StorageBackend>>,
        strategy: MultiStorageStrategy,
    ) -> Self {
        Self { backends, strategy }
    }

    /// Add a backend
    pub fn add_backend(&mut self, backend: Arc<dyn StorageBackend>) {
        self.backends.push(backend);
    }

    /// Remove a backend
    pub fn remove_backend(&mut self, index: usize) -> Option<Arc<dyn StorageBackend>> {
        if index < self.backends.len() {
            Some(self.backends.remove(index))
        } else {
            None
        }
    }

    /// Get number of backends
    pub fn backend_count(&self) -> usize {
        self.backends.len()
    }
}

#[async_trait]
impl StorageBackend for MultiStorage {
    async fn put_shard(&self, cid: &Cid, shard: &Shard) -> Result<(), FecError> {
        match self.strategy {
            MultiStorageStrategy::Redundant => {
                // Store in all backends
                let mut success_count = 0;
                let mut last_error = None;

                for backend in &self.backends {
                    match backend.put_shard(cid, shard).await {
                        Ok(()) => success_count += 1,
                        Err(e) => {
                            tracing::warn!("Failed to store shard in backend: {}", e);
                            last_error = Some(e);
                        }
                    }
                }

                if success_count > 0 {
                    Ok(())
                } else if let Some(e) = last_error {
                    Err(e)
                } else {
                    Err(FecError::Backend("No backends available".to_string()))
                }
            }
            MultiStorageStrategy::LoadBalance => {
                // Select backend based on CID hash
                let index = cid.as_bytes()[0] as usize % self.backends.len();
                self.backends[index].put_shard(cid, shard).await
            }
            MultiStorageStrategy::Failover => {
                // Try primary backend first, then failover
                for backend in &self.backends {
                    match backend.put_shard(cid, shard).await {
                        Ok(()) => return Ok(()),
                        Err(e) => {
                            tracing::warn!("Backend failed, trying next: {}", e);
                        }
                    }
                }
                Err(FecError::Backend("All backends failed".to_string()))
            }
        }
    }

    async fn get_shard(&self, cid: &Cid) -> Result<Shard, FecError> {
        // Try each backend in order until we find the shard
        for backend in &self.backends {
            match backend.get_shard(cid).await {
                Ok(shard) => return Ok(shard),
                Err(e) => {
                    tracing::debug!("Backend failed to get shard: {}", e);
                }
            }
        }

        Err(FecError::Backend(
            "Shard not found in any backend".to_string(),
        ))
    }

    async fn delete_shard(&self, cid: &Cid) -> Result<(), FecError> {
        // Delete from all backends that have it
        for backend in &self.backends {
            if let Err(e) = backend.delete_shard(cid).await {
                tracing::warn!("Failed to delete shard from backend: {}", e);
            }
        }
        Ok(())
    }

    async fn has_shard(&self, cid: &Cid) -> Result<bool, FecError> {
        // Check if any backend has the shard
        for backend in &self.backends {
            if backend.has_shard(cid).await? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn list_shards(&self) -> Result<Vec<Cid>, FecError> {
        let mut all_shards = std::collections::HashSet::new();

        // Collect from all backends
        for backend in &self.backends {
            if let Ok(shards) = backend.list_shards().await {
                all_shards.extend(shards);
            }
        }

        Ok(all_shards.into_iter().collect())
    }

    async fn put_metadata(&self, metadata: &FileMetadata) -> Result<(), FecError> {
        match self.strategy {
            MultiStorageStrategy::Redundant => {
                // Store in all backends
                let mut success_count = 0;
                let mut last_error = None;

                for backend in &self.backends {
                    match backend.put_metadata(metadata).await {
                        Ok(()) => success_count += 1,
                        Err(e) => {
                            tracing::warn!("Failed to store metadata in backend: {}", e);
                            last_error = Some(e);
                        }
                    }
                }

                if success_count > 0 {
                    Ok(())
                } else if let Some(e) = last_error {
                    Err(e)
                } else {
                    Err(FecError::Backend("No backends available".to_string()))
                }
            }
            MultiStorageStrategy::LoadBalance => {
                // Select backend based on file_id hash
                let index = metadata.file_id[0] as usize % self.backends.len();
                self.backends[index].put_metadata(metadata).await
            }
            MultiStorageStrategy::Failover => {
                // Try primary backend first, then failover
                for backend in &self.backends {
                    match backend.put_metadata(metadata).await {
                        Ok(()) => return Ok(()),
                        Err(e) => {
                            tracing::warn!("Backend failed, trying next: {}", e);
                        }
                    }
                }
                Err(FecError::Backend("All backends failed".to_string()))
            }
        }
    }

    async fn get_metadata(&self, file_id: &[u8; 32]) -> Result<FileMetadata, FecError> {
        // Try each backend in order
        for backend in &self.backends {
            match backend.get_metadata(file_id).await {
                Ok(metadata) => return Ok(metadata),
                Err(e) => {
                    tracing::debug!("Backend failed to get metadata: {}", e);
                }
            }
        }

        Err(FecError::Backend(
            "Metadata not found in any backend".to_string(),
        ))
    }

    async fn delete_metadata(&self, file_id: &[u8; 32]) -> Result<(), FecError> {
        // Delete from all backends
        for backend in &self.backends {
            if let Err(e) = backend.delete_metadata(file_id).await {
                tracing::warn!("Failed to delete metadata from backend: {}", e);
            }
        }
        Ok(())
    }

    async fn list_metadata(&self) -> Result<Vec<FileMetadata>, FecError> {
        let mut all_metadata = std::collections::HashMap::new();

        // Collect from all backends, deduplicating by file_id
        for backend in &self.backends {
            if let Ok(metadata_list) = backend.list_metadata().await {
                for metadata in metadata_list {
                    all_metadata.insert(metadata.file_id, metadata);
                }
            }
        }

        Ok(all_metadata.into_values().collect())
    }

    async fn stats(&self) -> Result<StorageStats, FecError> {
        let mut combined_stats = StorageStats {
            total_shards: 0,
            total_size: 0,
            metadata_count: 0,
            unreferenced_shards: 0,
        };

        // Aggregate stats from all backends
        for backend in &self.backends {
            if let Ok(stats) = backend.stats().await {
                combined_stats.total_shards += stats.total_shards;
                combined_stats.total_size += stats.total_size;
                combined_stats.metadata_count += stats.metadata_count;
                combined_stats.unreferenced_shards += stats.unreferenced_shards;
            }
        }

        Ok(combined_stats)
    }

    async fn garbage_collect(&self) -> Result<GcReport, FecError> {
        let mut combined_report = GcReport {
            shards_deleted: 0,
            bytes_freed: 0,
            duration_ms: 0,
        };

        let start_time = std::time::Instant::now();

        // Run GC on all backends
        for backend in &self.backends {
            if let Ok(report) = backend.garbage_collect().await {
                combined_report.shards_deleted += report.shards_deleted;
                combined_report.bytes_freed += report.bytes_freed;
            }
        }

        combined_report.duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(combined_report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_local_storage_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();

        let header = ShardHeader::new(EncryptionMode::Convergent, (16, 4), 13, [1u8; 32]);
        let shard = Shard::new(header, b"Hello, World!".to_vec());
        let cid = shard.cid().unwrap();

        // Store shard
        storage.put_shard(&cid, &shard).await.unwrap();

        // Verify it exists
        assert!(storage.has_shard(&cid).await.unwrap());

        // Retrieve shard
        let retrieved = storage.get_shard(&cid).await.unwrap();
        assert_eq!(retrieved.data, shard.data);

        // Delete shard
        storage.delete_shard(&cid).await.unwrap();
        assert!(!storage.has_shard(&cid).await.unwrap());
    }

    #[tokio::test]
    async fn test_local_storage_list() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalStorage::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();

        // Store multiple shards
        let mut shards = Vec::new();
        let mut cids = Vec::new();

        for i in 1..=3 {
            let header = ShardHeader::new(EncryptionMode::Convergent, (16, 4), 4, [i; 32]);
            let shard = Shard::new(header, b"data".to_vec());
            let cid = shard.cid().unwrap();
            storage.put_shard(&cid, &shard).await.unwrap();
            shards.push(shard);
            cids.push(cid);
        }

        // List shards
        let listed = storage.list_shards().await.unwrap();
        assert_eq!(listed.len(), 3);

        for cid in cids {
            assert!(listed.contains(&cid));
        }
    }

    #[test]
    fn test_network_storage_node_selection() {
        let nodes = vec![
            NodeEndpoint {
                address: "node1".to_string(),
                port: 8080,
                node_id: None,
            },
            NodeEndpoint {
                address: "node2".to_string(),
                port: 8080,
                node_id: None,
            },
            NodeEndpoint {
                address: "node3".to_string(),
                port: 8080,
                node_id: None,
            },
        ];

        let storage = NetworkStorage::new(nodes, 2);

        let shard_id = [42u8; 32];
        let selected = storage.select_nodes(&shard_id);

        assert_eq!(selected.len(), 2);

        // Should select same nodes for same shard ID
        let selected2 = storage.select_nodes(&shard_id);
        assert_eq!(selected, selected2);

        // Different shard should select different nodes (probably)
        let shard_id2 = [99u8; 32];
        let selected3 = storage.select_nodes(&shard_id2);
        // May or may not be different, but should be deterministic
        assert_eq!(selected3.len(), 2);
    }

    #[tokio::test]
    async fn test_multi_storage() {
        let temp_dir1 = TempDir::new().unwrap();
        let temp_dir2 = TempDir::new().unwrap();

        let backend1 = Arc::new(
            LocalStorage::new(temp_dir1.path().to_path_buf())
                .await
                .unwrap(),
        );
        let backend2 = Arc::new(
            LocalStorage::new(temp_dir2.path().to_path_buf())
                .await
                .unwrap(),
        );

        let multi = MultiStorage::new(vec![backend1.clone(), backend2.clone()]);

        let header = ShardHeader::new(EncryptionMode::Convergent, (16, 4), 9, [42u8; 32]);
        let shard = Shard::new(header, b"Test data".to_vec());
        let cid = shard.cid().unwrap();

        // Store through multi-backend
        multi.put_shard(&cid, &shard).await.unwrap();

        // Verify both backends have the shard
        assert!(backend1.has_shard(&cid).await.unwrap());
        assert!(backend2.has_shard(&cid).await.unwrap());

        // Delete from first backend
        backend1.delete_shard(&cid).await.unwrap();

        // Multi-backend should still find it in second backend
        let retrieved = multi.get_shard(&cid).await.unwrap();
        assert_eq!(retrieved.data, shard.data);
    }

    #[tokio::test]
    async fn test_memory_storage() {
        let storage = MemoryStorage::new();

        let header = ShardHeader::new(EncryptionMode::Convergent, (16, 4), 11, [1u8; 32]);
        let shard = Shard::new(header, b"Memory test".to_vec());
        let cid = shard.cid().unwrap();

        // Store shard
        storage.put_shard(&cid, &shard).await.unwrap();

        // Verify it exists
        assert!(storage.has_shard(&cid).await.unwrap());
        assert_eq!(storage.shard_count(), 1);

        // Retrieve shard
        let retrieved = storage.get_shard(&cid).await.unwrap();
        assert_eq!(retrieved.data, shard.data);

        // Test metadata
        let metadata = FileMetadata::new(
            [1u8; 32],
            1024,
            vec![ChunkMeta::new(
                (16, 4),
                EncryptionMode::Convergent,
                vec![cid.to_hex()],
            )],
        );

        storage.put_metadata(&metadata).await.unwrap();
        assert_eq!(storage.metadata_count(), 1);

        let retrieved_meta = storage.get_metadata(&metadata.file_id).await.unwrap();
        assert_eq!(retrieved_meta.file_id, metadata.file_id);

        // Clear storage
        storage.clear();
        assert_eq!(storage.shard_count(), 0);
        assert_eq!(storage.metadata_count(), 0);
    }

    #[tokio::test]
    async fn test_garbage_collection() {
        let storage = MemoryStorage::new();

        // Create unreferenced shard
        let header1 = ShardHeader::new(EncryptionMode::Convergent, (16, 4), 10, [1u8; 32]);
        let shard1 = Shard::new(header1, b"Unreferenced".to_vec());
        let cid1 = shard1.cid().unwrap();
        storage.put_shard(&cid1, &shard1).await.unwrap();

        // Create referenced shard
        let header2 = ShardHeader::new(EncryptionMode::Convergent, (16, 4), 10, [2u8; 32]);
        let shard2 = Shard::new(header2, b"Referenced".to_vec());
        let cid2 = shard2.cid().unwrap();
        storage.put_shard(&cid2, &shard2).await.unwrap();

        // Create metadata referencing only shard2
        let metadata = FileMetadata::new(
            [1u8; 32],
            1024,
            vec![ChunkMeta::new(
                (16, 4),
                EncryptionMode::Convergent,
                vec![cid2.to_hex()],
            )],
        );
        storage.put_metadata(&metadata).await.unwrap();

        // Run garbage collection
        let gc_report = storage.garbage_collect().await.unwrap();

        assert_eq!(gc_report.shards_deleted, 1);
        assert!(gc_report.bytes_freed > 0);
        assert!(!storage.has_shard(&cid1).await.unwrap()); // Unreferenced shard deleted
        assert!(storage.has_shard(&cid2).await.unwrap()); // Referenced shard kept
    }

    #[test]
    fn test_shard_header_serialization() {
        let header = ShardHeader::new(
            EncryptionMode::ConvergentWithSecret,
            (20, 5),
            2048,
            [42u8; 32],
        );

        let bytes = header.to_bytes().unwrap();
        assert_eq!(bytes.len(), ShardHeader::SIZE);

        let deserialized = ShardHeader::from_bytes(&bytes).unwrap();
        assert_eq!(deserialized.version, header.version);
        assert_eq!(deserialized.encryption_mode, header.encryption_mode);
        assert_eq!(deserialized.nspec, header.nspec);
        assert_eq!(deserialized.data_size, header.data_size);
        assert_eq!(deserialized.nonce, header.nonce);
    }

    #[test]
    fn test_shard_cid_calculation() {
        let header = ShardHeader::new(EncryptionMode::RandomKey, (16, 4), 1024, [0u8; 32]);
        let shard = Shard::new(header, vec![1, 2, 3, 4, 5]);

        let cid1 = shard.cid().unwrap();
        let cid2 = shard.cid().unwrap();

        // CID should be deterministic
        assert_eq!(cid1, cid2);

        // Different data should produce different CID
        let shard2 = Shard::new(shard.header.clone(), vec![1, 2, 3, 4, 6]);
        let cid3 = shard2.cid().unwrap();
        assert_ne!(cid1, cid3);
    }

    #[test]
    fn test_multi_storage_strategies() {
        let backend1 = Arc::new(MemoryStorage::new());
        let backend2 = Arc::new(MemoryStorage::new());

        // Test different strategies
        let redundant = MultiStorage::with_strategy(
            vec![backend1.clone(), backend2.clone()],
            MultiStorageStrategy::Redundant,
        );

        let load_balance = MultiStorage::with_strategy(
            vec![backend1.clone(), backend2.clone()],
            MultiStorageStrategy::LoadBalance,
        );

        let failover =
            MultiStorage::with_strategy(vec![backend1, backend2], MultiStorageStrategy::Failover);

        assert_eq!(redundant.backend_count(), 2);
        assert_eq!(load_balance.backend_count(), 2);
        assert_eq!(failover.backend_count(), 2);
    }

    #[test]
    fn test_cid_operations() {
        let data = b"test data";
        let cid1 = Cid::from_data(data);
        let cid2 = Cid::from_data(data);

        // Same data should produce same CID
        assert_eq!(cid1, cid2);

        // Different data should produce different CID
        let cid3 = Cid::from_data(b"different data");
        assert_ne!(cid1, cid3);

        // Test hex representation
        let hex = cid1.to_hex();
        assert_eq!(hex.len(), 64); // 32 bytes * 2 hex chars per byte

        // Test round-trip
        let bytes = cid1.as_bytes();
        let cid4 = Cid::new(*bytes);
        assert_eq!(cid1, cid4);
    }
}
