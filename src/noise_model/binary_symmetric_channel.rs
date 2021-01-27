use super::{NoiseModel, ProbabilityError};
use crate::SparseBinVec;
use itertools::Itertools;
use rand::distributions::{Bernoulli, Distribution};
use rand::Rng;

/// A binary symmetric channel flips at bit with
/// the given probrability.
///
/// This noise model returns a `SparseBinVec` where
/// the positions of each 1s are associated to bit flips.
pub struct BinarySymmetricChannel {
    distribution: Bernoulli,
}

impl BinarySymmetricChannel {
    /// Creates a new binary symmetric channel with the given error probability
    /// or returns an error if the probability is not between 0 and 1.
    pub fn try_with_probability(probability: f64) -> Result<Self, ProbabilityError> {
        Bernoulli::new(probability)
            .map(|distribution| Self { distribution })
            .map_err(|_| ProbabilityError(probability))
    }

    /// Creates a new binary symmetric channel with the given error probability.
    ///
    /// # Panic
    ///
    /// Panics if the probability is not between 0 and 1.
    pub fn with_probability(probability: f64) -> Self {
        let channel = Self::try_with_probability(probability);
        if let Ok(channel) = channel {
            channel
        } else {
            panic!("probability {} is not between 0 and 1", probability);
        }
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
