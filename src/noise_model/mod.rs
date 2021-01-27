//! Tools to generate random errors.
//!
//! Any type implementing the [`NoiseModel`](NoiseModel) trait
//! can be used to sample random errors.
//! Most function needing random errors in this crate require
//! a noise model.
//!
//! Some standard noise models such as
//! [`BinarySymmetricChannel`](BinarySymmetricChannel)
//! are implemented.
use rand::Rng;

mod binary_symmetric_channel;
pub use binary_symmetric_channel::BinarySymmetricChannel;

pub trait NoiseModel {
    /// The type of the generated errors.
    type Error;

    /// Generates a random error of the given length.
    fn sample_error_of_length<R: Rng>(&self, length: usize, rng: &mut R) -> Self::Error;
}

/// An error when a probability is not between 0.0 and 1.1 inclusively.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct ProbabilityError(f64);

impl std::fmt::Display for ProbabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} is not between 0.0 and 1.0", self.0)
    }
}

impl std::error::Error for ProbabilityError {}
