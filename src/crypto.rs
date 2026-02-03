//! Encryption module providing convergent and standard encryption for Saorsa FEC
//!
//! This module implements AES-256-GCM encryption with multiple modes:
//! - Convergent encryption for deduplication across all users
//! - Convergent with secret for controlled deduplication
//! - Random key for maximum privacy

use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    Aes256Gcm, Key, Nonce,
};
use anyhow::{Context, Result};
// blake3::Hasher removed as we're using SHA-256 for v0.3 spec
use hkdf::Hkdf;
use rand_core::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Whether encryption is enabled
    pub enabled: bool,
    /// Encryption mode to use
    pub mode: EncryptionMode,
    /// Optional convergence secret for controlled deduplication
    #[serde(skip_serializing, skip_deserializing)]
    pub convergence_secret: Option<ConvergenceSecret>,
}

/// Secret used for convergent encryption with controlled deduplication
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct ConvergenceSecret([u8; 32]);

impl ConvergenceSecret {
    /// Create a new convergence secret
    pub fn new(secret: [u8; 32]) -> Self {
        Self(secret)
    }

    /// Get the secret as bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Encryption mode selection
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EncryptionMode {
    /// Pure convergent encryption (deduplication across all users)
    Convergent,
    /// Convergent encryption with secret (controlled deduplication)
    ConvergentWithSecret,
    /// Random key encryption (no deduplication)
    RandomKey,
}

/// Encryption algorithm selection
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    /// AES-256-GCM authenticated encryption
    Aes256Gcm,
}

/// Key derivation method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyDerivation {
    /// Blake3-based convergent key derivation
    Blake3Convergent,
    /// Random key generation
    Random,
}

/// Metadata about how data was encrypted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionMetadata {
    /// Algorithm used for encryption
    pub algorithm: EncryptionAlgorithm,
    /// Key derivation method used
    pub key_derivation: KeyDerivation,
    /// ID of convergence secret if used (Blake3 hash of secret)
    pub convergence_secret_id: Option<[u8; 16]>,
    /// Nonce used for encryption
    pub nonce: [u8; 12],
}

/// Encryption key wrapper with secure handling
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct EncryptionKey([u8; 32]);

impl EncryptionKey {
    /// Create a new encryption key
    pub fn new(key: [u8; 32]) -> Self {
        Self(key)
    }

    /// Get the key as bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Main encryption engine
pub struct CryptoEngine {
    /// Last nonce used (for metadata)
    last_nonce: Option<[u8; 12]>,
}

impl CryptoEngine {
    /// Create a new crypto engine
    pub fn new() -> Self {
        Self { last_nonce: None }
    }

    /// Encrypt data using the specified key
    pub fn encrypt(&mut self, data: &[u8], key: &EncryptionKey) -> Result<Vec<u8>> {
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key.as_bytes()));
        let nonce_bytes = Aes256Gcm::generate_nonce(&mut OsRng);
        self.last_nonce = Some(nonce_bytes.into());

        let ciphertext = cipher
            .encrypt(&nonce_bytes, data)
            .map_err(|_| anyhow::anyhow!("Encryption failed"))?;

        // Prepend nonce to ciphertext for storage
        let mut result = Vec::with_capacity(12 + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend_from_slice(&ciphertext);

        Ok(result)
    }

    /// Decrypt data using the specified key
    pub fn decrypt(&self, encrypted_data: &[u8], key: &EncryptionKey) -> Result<Vec<u8>> {
        if encrypted_data.len() < 12 {
            anyhow::bail!("Encrypted data too short to contain nonce");
        }

        let (nonce_bytes, ciphertext) = encrypted_data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key.as_bytes()));
        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| anyhow::anyhow!("Decryption failed"))?;

        Ok(plaintext)
    }

    /// Get the last nonce used
    pub fn last_nonce(&self) -> [u8; 12] {
        self.last_nonce.unwrap_or([0u8; 12])
    }

    /// Reconstruct encryption key from metadata
    pub fn reconstruct_key(
        &self,
        metadata: &Option<EncryptionMetadata>,
        original_data: Option<&[u8]>,
        convergence_secret: Option<&ConvergenceSecret>,
    ) -> Result<EncryptionKey> {
        let metadata = metadata
            .as_ref()
            .context("No encryption metadata available")?;

        match metadata.key_derivation {
            KeyDerivation::Blake3Convergent => {
                let data = original_data
                    .context("Original data required for convergent key reconstruction")?;

                let secret = if metadata.convergence_secret_id.is_some() {
                    convergence_secret.map(|s| s.as_bytes())
                } else {
                    None
                };

                derive_convergent_key(data, secret)
            }
            KeyDerivation::Random => {
                anyhow::bail!("Random keys cannot be reconstructed without external storage")
            }
        }
    }
}

impl Default for CryptoEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Derive a convergent encryption key from content using SHA-256 HKDF
///
/// **SECURITY NOTE**: This implements the v0.3 specification for convergent
/// encryption. While deterministic for deduplication, it has security implications:
/// - Identical plaintexts produce identical keys and ciphertexts
/// - No semantic security for identical content
/// - Consider using ConvergentWithSecret or RandomKey modes for sensitive data
pub fn derive_convergent_key(content: &[u8], secret: Option<&[u8; 32]>) -> Result<EncryptionKey> {
    // Use SHA-256 hash of content as the input key material (IKM)
    let mut hasher = Sha256::new();

    // Include secret if provided for controlled deduplication
    if let Some(s) = secret {
        hasher.update(s);
    }

    // Include content for convergence
    hasher.update(content);
    let content_hash = hasher.finalize();

    // Use a fixed salt for deterministic behavior with domain separation
    let salt = {
        let mut salt_hasher = Sha256::new();
        salt_hasher.update(b"saorsa-fec-v0.3-salt");
        salt_hasher.update(b"convergent-encryption");
        salt_hasher.finalize()
    };

    // HKDF with proper salt and info for key derivation
    let hkdf = Hkdf::<Sha256>::new(Some(&salt), &content_hash);
    let mut key = [0u8; 32];
    hkdf.expand(b"saorsa-fec:aead:v1", &mut key)
        .map_err(|_| anyhow::anyhow!("HKDF expand failed unexpectedly"))?;

    let encryption_key = EncryptionKey::new(key);

    // Zeroize the intermediate key material
    let mut key_to_zero = key;
    key_to_zero.zeroize();

    Ok(encryption_key)
}

/// Generate a random encryption key using cryptographically secure RNG
pub fn generate_random_key() -> EncryptionKey {
    let mut key = [0u8; 32];
    OsRng.fill_bytes(&mut key);
    let encryption_key = EncryptionKey::new(key);

    // Zeroize the key array after use
    key.zeroize();

    encryption_key
}

/// Generate deterministic nonce for convergent encryption (v0.3 specification)
///
/// **SECURITY WARNING**: This function generates deterministic nonces as required
/// by the v0.3 specification for convergent encryption and deduplication. While
/// this enables deduplication, it has important security implications:
///
/// 1. **Identical plaintexts produce identical ciphertexts** - This can leak
///    information about data patterns across different files.
/// 2. **No semantic security** - Attackers who know plaintext can verify
///    their guesses by comparing ciphertexts.
/// 3. **Partial information leakage** - Common file patterns may be detectable.
///
/// This trade-off is intentional for deduplication purposes but should be
/// clearly understood by users of the system.
///
/// Formula: `H(file_id || chunk_index || shard_index)[..12]`
pub fn generate_deterministic_nonce(
    file_id: &[u8; 32],
    chunk_index: u32,
    shard_index: u16,
) -> [u8; 12] {
    let mut hasher = Sha256::new();

    // Domain separation for nonce generation
    hasher.update(b"saorsa-fec-nonce-v0.3");
    hasher.update(file_id);
    hasher.update(chunk_index.to_le_bytes());
    hasher.update(shard_index.to_le_bytes());

    let hash = hasher.finalize();
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&hash[..12]);
    nonce
}

/// Verify MAC in constant time to prevent timing attacks
///
/// **SECURITY**: Uses constant-time comparison to prevent timing-based
/// side-channel attacks that could be used to forge authentication tags.
pub fn verify_mac_constant_time(computed: &[u8], stored: &[u8]) -> bool {
    if computed.len() != stored.len() {
        return false;
    }

    computed.ct_eq(stored).into()
}

/// Derive MAC key with proper domain separation
///
/// Derives a separate key for message authentication to prevent
/// key correlation between encryption and authentication operations.
pub fn derive_mac_key(encryption_key: &EncryptionKey) -> Result<[u8; 32]> {
    let salt = {
        let mut salt_hasher = Sha256::new();
        salt_hasher.update(b"saorsa-fec-v0.3-salt");
        salt_hasher.update(b"mac-key-derivation");
        salt_hasher.finalize()
    };

    let hkdf = Hkdf::<Sha256>::new(Some(&salt), encryption_key.as_bytes());
    let mut mac_key = [0u8; 32];
    hkdf.expand(b"saorsa-fec:mac:v1", &mut mac_key)
        .map_err(|_| anyhow::anyhow!("HKDF expand failed unexpectedly"))?;

    Ok(mac_key)
}

/// Compute convergence secret ID
pub fn compute_secret_id(secret: &ConvergenceSecret) -> [u8; 16] {
    let hash = blake3::hash(secret.as_bytes());
    let mut id = [0u8; 16];
    id.copy_from_slice(&hash.as_bytes()[..16]);
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_roundtrip() {
        let mut engine = CryptoEngine::new();
        let data = b"Hello, World!";
        let key = derive_convergent_key(data, None).unwrap();

        let encrypted = engine.encrypt(data, &key).unwrap();
        assert_ne!(encrypted, data);
        assert!(encrypted.len() > data.len() + 12); // Nonce + tag overhead

        let decrypted = engine.decrypt(&encrypted, &key).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_convergent_key_deterministic() {
        let data = b"Test data";
        let key1 = derive_convergent_key(data, None).unwrap();
        let key2 = derive_convergent_key(data, None).unwrap();

        assert_eq!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_convergent_key_with_secret() {
        let data = b"Test data";
        let secret = ConvergenceSecret::new([42u8; 32]);

        let key_with_secret = derive_convergent_key(data, Some(secret.as_bytes())).unwrap();
        let key_without = derive_convergent_key(data, None).unwrap();

        assert_ne!(key_with_secret.as_bytes(), key_without.as_bytes());
    }

    #[test]
    fn test_random_key_uniqueness() {
        let key1 = generate_random_key();
        let key2 = generate_random_key();

        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_decrypt_invalid_data() {
        let engine = CryptoEngine::new();
        let key = generate_random_key();

        // Too short
        let result = engine.decrypt(&[0u8; 10], &key);
        assert!(result.is_err());

        // Invalid ciphertext
        let result = engine.decrypt(&[0u8; 30], &key);
        assert!(result.is_err());
    }

    #[test]
    fn test_encryption_metadata_serialization() {
        let metadata = EncryptionMetadata {
            algorithm: EncryptionAlgorithm::Aes256Gcm,
            key_derivation: KeyDerivation::Blake3Convergent,
            convergence_secret_id: Some([1u8; 16]),
            nonce: [2u8; 12],
        };

        let serialized = postcard::to_stdvec(&metadata).unwrap();
        let deserialized: EncryptionMetadata = postcard::from_bytes(&serialized).unwrap();

        assert_eq!(
            deserialized.convergence_secret_id,
            metadata.convergence_secret_id
        );
        assert_eq!(deserialized.nonce, metadata.nonce);
    }
}
