use super::{NoiseModel, Probability};
use serde::{Serialize, Deserialize};
use pauli::{Pauli, PauliOperator, X, Y, Z};
use rand::distributions::{Bernoulli, Distribution};
use rand::seq::SliceRandom;
use rand::Rng;
use std::fmt;

/// A depolarizing noise channel apply one of the 3 non-trivial Pauli
/// operator with the given probrability.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DepolarizingNoise {
    distribution: Bernoulli,
    probability: f64,
    non_trivial_paulis: [Pauli; 3],
}

impl DepolarizingNoise {
    /// Creates a new binary symmetric channel with the given error probability.
    pub fn with_probability(probability: Probability) -> Self {
        Bernoulli::new(probability.value())
            .map(|distribution| Self {
                distribution,
                probability: probability.value(),
                non_trivial_paulis: [X, Y, Z],
            })
            .unwrap()
    }
}

impl NoiseModel for DepolarizingNoise {
    type Error = PauliOperator;

    fn sample_error_of_length<R: Rng>(&self, length: usize, rng: &mut R) -> Self::Error {
        let (positions, paulis) = (0..length)
            .filter_map(|position| {
                if self.distribution.sample(rng) {
                    Some((
                        position,
                        self.non_trivial_paulis.choose(rng).cloned().unwrap(),
                    ))
                } else {
                    None
                }
            })
            .unzip();
        PauliOperator::new(length, positions, paulis)
    }
}

impl fmt::Display for DepolarizingNoise {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Depolarizing Noise (prob = {})", self.probability)
    }
}
