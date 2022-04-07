use super::{NoiseModel, Probability};
use sparse_bin_mat::SparseBinVec;
use itertools::Itertools;
use rand::distributions::{Bernoulli, Distribution};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

/// An erasure channel working for both classical and quantum codes.
///
/// This noise model returns a `SparseBinVec` where
/// the positions of each 1 is an erasure.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ErasureChannel {
    distribution: Bernoulli,
    probability: f64,
}

impl ErasureChannel {
    /// Creates a new erasure channel with the given error probability.
    pub fn with_probability(probability: Probability) -> Self {
        Bernoulli::new(probability.value())
            .map(|distribution| Self {
                distribution,
                probability: probability.value(),
            })
            .unwrap()
    }
}

impl NoiseModel for ErasureChannel {
    type Error = SparseBinVec;

    fn sample_error_of_length<R: Rng>(&self, length: usize, rng: &mut R) -> Self::Error {
        let positions = self
            .distribution
            .sample_iter(rng)
            .take(length)
            .positions(|error| error)
            .collect();
        SparseBinVec::new(length, positions)
    }
}

impl fmt::Display for ErasureChannel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Erasure({})", self.probability)
    }
}
