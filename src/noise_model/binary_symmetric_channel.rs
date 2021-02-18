use super::{NoiseModel, Probability};
use crate::SparseBinVec;
use itertools::Itertools;
use rand::distributions::{Bernoulli, Distribution};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A binary symmetric channel flips at bit with
/// the given probrability.
///
/// This noise model returns a `SparseBinVec` where
/// the positions of each 1s are associated to bit flips.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BinarySymmetricChannel {
    distribution: Bernoulli,
    probability: f64,
}

impl BinarySymmetricChannel {
    /// Creates a new binary symmetric channel with the given error probability.
    pub fn with_probability(probability: Probability) -> Self {
        Bernoulli::new(probability.value())
            .map(|distribution| Self {
                distribution,
                probability: probability.value(),
            })
            .unwrap()
    }
}

impl NoiseModel for BinarySymmetricChannel {
    type Error = SparseBinVec;

    fn sample_error_of_length<R: Rng>(&self, block_size: usize, rng: &mut R) -> Self::Error {
        let positions = self
            .distribution
            .sample_iter(rng)
            .take(block_size)
            .positions(|error| error)
            .collect();
        SparseBinVec::new(block_size, positions)
    }
}

impl fmt::Display for BinarySymmetricChannel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BSC({})", self.probability)
    }
}
