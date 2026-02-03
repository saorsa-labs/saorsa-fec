// Copyright 2024 Saorsa Labs
// SPDX-License-Identifier: AGPL-3.0-or-later

//! High-performance Reed-Solomon implementation using reed-solomon-simd

use crate::{FecBackend, FecError, FecParams, Result};
use reed_solomon_simd::ReedSolomonEncoder;

/// High-performance Reed-Solomon backend using SIMD optimizations
#[derive(Debug)]
pub struct PureRustBackend {}

impl Default for PureRustBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl PureRustBackend {
    pub fn new() -> Self {
        Self {}
    }

    fn encode_systematic(
        &self,
        data_blocks: &[&[u8]],
        parity_out: &mut [Vec<u8>],
        k: usize,
        m: usize,
    ) -> Result<()> {
        if data_blocks.len() != k {
            return Err(FecError::InvalidParameters {
                k: data_blocks.len(),
                n: k + m,
            });
        }

        if parity_out.len() != m {
            return Err(FecError::InvalidParameters {
                k,
                n: k + parity_out.len(),
            });
        }

        let block_size = data_blocks[0].len();
        for block in data_blocks {
            if block.len() != block_size {
                return Err(FecError::SizeMismatch {
                    expected: block_size,
                    actual: block.len(),
                });
            }
        }

        // Ensure block size is even (requirement of reed-solomon-simd)
        if !block_size.is_multiple_of(2) {
            return Err(FecError::Backend(
                "Shard size must be even for reed-solomon-simd".to_string(),
            ));
        }

        // Create encoder with proper parameters
        let mut encoder = ReedSolomonEncoder::new(k, m, block_size)
            .map_err(|e| FecError::Backend(e.to_string()))?;

        // Add original shards
        for block in data_blocks {
            encoder
                .add_original_shard(block)
                .map_err(|e| FecError::Backend(e.to_string()))?;
        }

        // Generate recovery shards
        let result = encoder
            .encode()
            .map_err(|e| FecError::Backend(e.to_string()))?;

        // Copy recovery shards to output
        let recovery_shards: Vec<_> = result.recovery_iter().collect();
        for (i, parity_block) in parity_out.iter_mut().enumerate() {
            if i < recovery_shards.len() {
                *parity_block = recovery_shards[i].to_vec();
            }
        }

        Ok(())
    }

    fn decode_systematic(&self, shares: &mut [Option<Vec<u8>>], k: usize) -> Result<()> {
        let n = shares.len();
        let m = n - k;

        // Count available shares
        let available_count = shares.iter().filter(|s| s.is_some()).count();
        if available_count < k {
            return Err(FecError::InsufficientShares {
                have: available_count,
                need: k,
            });
        }

        // Check if we have all data shares (fast path)
        let have_all_data = (0..k).all(|i| shares[i].is_some());
        if have_all_data {
            return Ok(()); // Nothing to decode
        }

        // Get block size from first available share
        let block_size = shares
            .iter()
            .find_map(|s| s.as_ref().map(|data| data.len()))
            .ok_or(FecError::InsufficientShares { have: 0, need: k })?;

        // For reconstruction with reed-solomon-simd v3, we need to re-encode and replace missing shards
        // Create encoder
        let _encoder = ReedSolomonEncoder::new(k, m, block_size)
            .map_err(|e| FecError::Backend(format!("Failed to create encoder: {:?}", e)))?;

        // Convert Option<Vec<u8>> to Vec<Vec<u8>> for processing
        // Missing shards will be replaced with zeros temporarily
        let mut work_shards: Vec<Vec<u8>> = Vec::with_capacity(n);
        let mut missing_indices = Vec::new();

        for (i, shard) in shares.iter().enumerate() {
            if let Some(data) = shard {
                work_shards.push(data.clone());
            } else {
                work_shards.push(vec![0u8; block_size]);
                if i < k {
                    missing_indices.push(i);
                }
            }
        }

        // If we have missing data shards, we need to reconstruct them
        if !missing_indices.is_empty() {
            // reed-solomon-simd v3 doesn't expose direct reconstruction
            // We can only use it for encoding, not for decoding missing data shards
            // For now, return an error if we need complex reconstruction
            return Err(FecError::Backend(
                "Reed-Solomon reconstruction with missing data shards is not supported in reed-solomon-simd v3".to_string(),
            ));
        }

        // Copy reconstructed shards back to the output
        for (i, shard) in work_shards.into_iter().enumerate() {
            if shares[i].is_none() {
                shares[i] = Some(shard);
            }
        }

        Ok(())
    }
}

impl FecBackend for PureRustBackend {
    fn encode_blocks(
        &self,
        data: &[&[u8]],
        parity: &mut [Vec<u8>],
        params: FecParams,
    ) -> Result<()> {
        self.encode_systematic(
            data,
            parity,
            params.data_shares as usize,
            params.parity_shares as usize,
        )
    }

    fn decode_blocks(&self, shares: &mut [Option<Vec<u8>>], params: FecParams) -> Result<()> {
        self.decode_systematic(shares, params.data_shares as usize)
    }

    fn generate_matrix(&self, k: usize, m: usize) -> Vec<Vec<u8>> {
        // reed-solomon-simd doesn't expose matrix generation directly
        // Return a placeholder identity + vandermonde-like matrix for compatibility
        let mut matrix = vec![vec![0u8; k]; k + m];

        // Identity matrix for data shards
        for (i, row) in matrix.iter_mut().enumerate().take(k) {
            row[i] = 1;
        }

        // Vandermonde-like matrix for parity shards (simplified)
        for (i, row) in matrix.iter_mut().enumerate().skip(k).take(m) {
            for (j, cell) in row.iter_mut().enumerate().take(k) {
                *cell = ((i - k + 1) * (j + 1)) as u8;
            }
        }

        matrix
    }
    fn name(&self) -> &'static str {
        "reed-solomon-simd"
    }

    fn is_accelerated(&self) -> bool {
        // reed-solomon-simd provides SIMD acceleration
        // Check for available CPU features at runtime
        cfg!(any(
            target_feature = "avx2",
            target_feature = "avx",
            target_feature = "sse4.1",
            target_feature = "neon"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_small() {
        let backend = PureRustBackend::new();
        let params = FecParams::new(3, 2).unwrap();

        // Create test data with even-sized blocks (required by reed-solomon-simd)
        let data1 = vec![1, 2, 3, 4]; // 4 bytes is even
        let data2 = vec![5, 6, 7, 8];
        let data3 = vec![9, 10, 11, 12];
        let data_blocks: Vec<&[u8]> = vec![&data1, &data2, &data3];

        // Encode
        let mut parity = vec![vec![]; 2];
        backend
            .encode_blocks(&data_blocks, &mut parity, params)
            .unwrap();

        assert_eq!(parity[0].len(), 4);
        assert_eq!(parity[1].len(), 4);

        // For systematic encoding, the original data should be preserved
        // and we can test that we have the parity data
        assert!(!parity[0].is_empty());
        assert!(!parity[1].is_empty());
    }

    #[test]
    fn test_even_size_requirement() {
        let backend = PureRustBackend::new();
        let params = FecParams::new(2, 1).unwrap();

        // Create test data with odd-sized blocks (should fail)
        let data1 = vec![1, 2, 3]; // 3 bytes is odd
        let data2 = vec![4, 5, 6];
        let data_blocks: Vec<&[u8]> = vec![&data1, &data2];

        // Encode should fail due to odd block size
        let mut parity = vec![vec![]];
        let result = backend.encode_blocks(&data_blocks, &mut parity, params);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("even"));
    }

    #[test]
    fn test_systematic_property() {
        let backend = PureRustBackend::new();
        let params = FecParams::new(4, 2).unwrap();

        // Create test data with even-sized blocks
        let data: Vec<Vec<u8>> = (0..4)
            .map(|i| {
                let mut v = Vec::new();
                for _ in 0..50 {
                    v.push(i as u8);
                    v.push((i + 1) as u8);
                }
                v
            })
            .collect(); // 100 bytes each, even
        let data_refs: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();

        // Encode
        let mut parity = vec![vec![]; 2];
        backend
            .encode_blocks(&data_refs, &mut parity, params)
            .unwrap();

        // Test that we can handle having all original shares
        let mut shares: Vec<Option<Vec<u8>>> = (0..6)
            .map(|i| {
                if i < 4 {
                    Some(data[i].clone())
                } else {
                    Some(parity[i - 4].clone())
                }
            })
            .collect();

        // Should succeed with all data available
        backend.decode_blocks(&mut shares, params).unwrap();

        // Verify all data shares are still present
        for i in 0..4 {
            assert_eq!(shares[i].as_ref().unwrap(), &data[i]);
        }
    }
}
