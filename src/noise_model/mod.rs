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

pub struct Probability(f64);

impl Probability {
    pub fn new(probability: f64) -> Self {
        Self::try_new(probability).expect("probability is not between 0 and 1")
    }

    pub fn try_new(probability: f64) -> Option<Self> {
        if 0.0 <= probability && probability <= 1.0 {
            Some(Self(probability))
        } else {
            None
        }
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}
