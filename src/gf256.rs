// Copyright 2024 Saorsa Labs
// SPDX-License-Identifier: AGPL-3.0-or-later

//! GF(256) Galois Field arithmetic for Reed-Solomon coding
//!
//! This module implements arithmetic operations over GF(2^8) using
//! the irreducible polynomial x^8 + x^4 + x^3 + x + 1 (0x11b)

use std::ops::{Add, Div, Mul, Sub};

/// GF(256) field element
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gf256(pub u8);

/// Precomputed logarithm table for GF(256)
static LOG_TABLE: [u8; 256] = generate_log_table();
/// Precomputed exponential table for GF(256)
static EXP_TABLE: [u8; 512] = generate_exp_table();

const fn generate_log_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut val = 1u8;
    let mut i = 0;

    while i < 255 {
        table[val as usize] = i;
        val = gf_mul_slow(val, 3); // generator = 3
        i += 1;
    }

    table
}

const fn generate_exp_table() -> [u8; 512] {
    let mut table = [0u8; 512];
    let mut val = 1u8;
    let mut i = 0;

    while i < 255 {
        table[i] = val;
        table[i + 255] = val; // Wrap around for easy modulo
        val = gf_mul_slow(val, 3);
        i += 1;
    }

    table
}

/// Slow multiplication for table generation (const fn compatible)
const fn gf_mul_slow(a: u8, b: u8) -> u8 {
    let mut result = 0u8;
    let mut aa = a;
    let mut bb = b;

    while bb != 0 {
        if bb & 1 != 0 {
            result ^= aa;
        }
        aa = if aa & 0x80 != 0 {
            (aa << 1) ^ 0x1b // polynomial reduction
        } else {
            aa << 1
        };
        bb >>= 1;
    }

    result
}

impl Gf256 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    /// Create a new GF(256) element
    pub const fn new(val: u8) -> Self {
        Self(val)
    }

    /// Get the multiplicative inverse
    pub fn inv(self) -> Result<Self, &'static str> {
        if self.0 == 0 {
            return Err("Cannot invert zero in GF(256)");
        }
        Ok(Self(EXP_TABLE[(255 - LOG_TABLE[self.0 as usize]) as usize]))
    }

    /// Raise to a power
    pub fn pow(self, exp: u8) -> Self {
        if self.0 == 0 {
            return Self::ZERO;
        }
        if exp == 0 {
            return Self::ONE;
        }

        let log_val = LOG_TABLE[self.0 as usize] as u32;
        let result = (log_val * exp as u32) % 255;
        Self(EXP_TABLE[result as usize])
    }

    /// Safe division that returns a Result
    pub fn safe_div(self, other: Self) -> Result<Self, &'static str> {
        if other.0 == 0 {
            return Err("Division by zero in GF(256)");
        }
        if self.0 == 0 {
            return Ok(Self::ZERO);
        }

        let log_diff =
            (LOG_TABLE[self.0 as usize] as i16 - LOG_TABLE[other.0 as usize] as i16 + 255) % 255;
        Ok(Self(EXP_TABLE[log_diff as usize]))
    }
}

impl Add for Gf256 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }
}

impl Sub for Gf256 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, other: Self) -> Self {
        Self(self.0 ^ other.0) // Addition and subtraction are the same in GF(256)
    }
}

impl Mul for Gf256 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        if self.0 == 0 || other.0 == 0 {
            return Self::ZERO;
        }

        let log_sum = LOG_TABLE[self.0 as usize] as u16 + LOG_TABLE[other.0 as usize] as u16;
        Self(EXP_TABLE[log_sum as usize])
    }
}

impl Div for Gf256 {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        if other.0 == 0 {
            // Division by zero in GF(256) is undefined, return zero
            // This should not happen in correct Reed-Solomon usage
            return Self::ZERO;
        }
        if self.0 == 0 {
            return Self::ZERO;
        }

        let log_diff =
            (LOG_TABLE[self.0 as usize] as i16 - LOG_TABLE[other.0 as usize] as i16 + 255) % 255;
        Self(EXP_TABLE[log_diff as usize])
    }
}

/// Perform vector-scalar multiplication in GF(256)
pub fn mul_slice(dst: &mut [u8], src: &[u8], scalar: Gf256) {
    if scalar.0 == 0 {
        dst.fill(0);
        return;
    }
    if scalar.0 == 1 {
        dst.copy_from_slice(src);
        return;
    }

    let log_scalar = LOG_TABLE[scalar.0 as usize] as u16;

    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        if s == 0 {
            *d = 0;
        } else {
            let log_val = LOG_TABLE[s as usize] as u16;
            *d = EXP_TABLE[(log_val + log_scalar) as usize];
        }
    }
}

/// Add two slices in GF(256) (XOR)
pub fn add_slice(dst: &mut [u8], src: &[u8]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d ^= s;
    }
}

/// Generate Cauchy matrix for Reed-Solomon
pub fn generate_cauchy_matrix(k: usize, m: usize) -> Vec<Vec<Gf256>> {
    let n = k + m;
    let mut matrix = vec![vec![Gf256::ZERO; n]; n];

    // Identity matrix for systematic encoding
    for (i, row) in matrix.iter_mut().enumerate().take(k) {
        row[i] = Gf256::ONE;
    }

    // Cauchy matrix for parity rows
    // Use carefully chosen values to avoid xi + yj = 0
    for i in 0..m {
        for (j, elem) in matrix[k + i].iter_mut().take(k).enumerate() {
            // Use non-overlapping ranges to ensure xi + yj never equals 0 in GF(256)
            let xi = Gf256::new((i + 1) as u8);
            let yj = Gf256::new((j + 128) as u8); // Offset by 128 to avoid overlap
            let sum = xi + yj;
            if sum.0 == 0 {
                // This shouldn't happen with our offset, but handle it gracefully
                *elem = Gf256::new(1);
            } else {
                *elem = Gf256::ONE / sum;
            }
        }
    }

    matrix
}

/// Invert a matrix in GF(256) using Gaussian elimination
pub fn invert_matrix(matrix: &[Vec<Gf256>]) -> Option<Vec<Vec<Gf256>>> {
    let n = matrix.len();
    let mut work = matrix.to_vec();
    let mut inv = vec![vec![Gf256::ZERO; n]; n];

    // Initialize inverse as identity
    for (i, row) in inv.iter_mut().enumerate().take(n) {
        row[i] = Gf256::ONE;
    }

    // Gaussian elimination
    for col in 0..n {
        // Find pivot
        let mut pivot_row = col;
        for row in (col + 1)..n {
            if work[row][col].0 != 0 && work[pivot_row][col].0 == 0 {
                pivot_row = row;
            }
        }

        if work[pivot_row][col].0 == 0 {
            return None; // Singular matrix
        }

        // Swap rows if needed
        if pivot_row != col {
            work.swap(pivot_row, col);
            inv.swap(pivot_row, col);
        }

        // Scale pivot row
        let pivot = work[col][col];
        let pivot_inv = match pivot.inv() {
            Ok(inv) => inv,
            Err(_) => return None, // Cannot invert zero
        };
        for j in 0..n {
            work[col][j] = work[col][j] * pivot_inv;
            inv[col][j] = inv[col][j] * pivot_inv;
        }

        // Eliminate column
        for row in 0..n {
            if row != col && work[row][col].0 != 0 {
                let factor = work[row][col];
                for j in 0..n {
                    work[row][j] = work[row][j] - factor * work[col][j];
                    inv[row][j] = inv[row][j] - factor * inv[col][j];
                }
            }
        }
    }

    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf256_arithmetic() {
        let a = Gf256::new(5);
        let b = Gf256::new(7);

        assert_eq!((a + b).0, 5 ^ 7);
        assert_eq!((a - b).0, 5 ^ 7); // Same as addition

        let c = a * b;
        assert_eq!(c / b, a);
        assert_eq!(c / a, b);
    }

    #[test]
    fn test_inverse() {
        for i in 1..=255 {
            let a = Gf256::new(i);
            let inv = a.inv().expect("Non-zero elements should have inverse");
            assert_eq!(a * inv, Gf256::ONE);
        }
    }

    #[test]
    fn test_cauchy_matrix() {
        let matrix = generate_cauchy_matrix(3, 2);
        assert_eq!(matrix.len(), 5);

        // Check identity portion
        for (i, row) in matrix.iter().enumerate().take(3) {
            for (j, &val) in row.iter().enumerate().take(3) {
                if i == j {
                    assert_eq!(val, Gf256::ONE);
                } else {
                    assert_eq!(val, Gf256::ZERO);
                }
            }
        }
    }

    #[test]
    fn test_matrix_inversion() {
        let matrix = vec![
            vec![Gf256::new(1), Gf256::new(2), Gf256::new(3)],
            vec![Gf256::new(4), Gf256::new(5), Gf256::new(6)],
            vec![Gf256::new(7), Gf256::new(8), Gf256::new(10)],
        ];

        let inv = invert_matrix(&matrix).expect("Matrix should be invertible");

        // Verify A * A^-1 = I
        for (i, row) in matrix.iter().enumerate().take(3) {
            for (j, _) in row.iter().enumerate().take(3) {
                let mut sum = Gf256::ZERO;
                for (k, &left) in row.iter().enumerate().take(3) {
                    sum = sum + left * inv[k][j];
                }
                if i == j {
                    assert_eq!(sum, Gf256::ONE);
                } else {
                    assert_eq!(sum, Gf256::ZERO);
                }
            }
        }
    }
}
