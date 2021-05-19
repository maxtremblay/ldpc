//! A toolbox for classical (and soon quantum) LDPC codes.
//!
//! For now,
//! only [classical linear codes](`LinearCode`) are implemented.
//!
//! There is also a generic implementation of [noise model](noise_model)
//! that can be used to generate random error for codes.  # Example
//!
//! ```
//! use ldpc::LinearCode;
//! use ldpc::noise_model::{Probability, BinarySymmetricChannel};
//! use rand::thread_rng;
//!
//! // This sample a random regular LDPC code.
//! // It may returns an error, thus the unwrap.
//! let code = LinearCode::random_regular_code()
//!     .block_size(40)
//!     .number_of_checks(20)
//!     .bit_degree(3)
//!     .check_degree(6)
//!     .sample_with(&mut thread_rng())
//!     .unwrap();
//!
//! let noise = BinarySymmetricChannel::with_probability(Probability::new(0.1));
//!
//! // The error is a sparse binary vector where each 1 represent a bit flip.
//! let error = code.random_error(&noise, &mut thread_rng());
//! ```

pub mod decoders;

mod linear_code;
pub use crate::linear_code::{Edge, Edges, LinearCode, RandomRegularCode};

pub mod noise_model;

pub use sparse_bin_mat::{SparseBinMat, SparseBinSlice, SparseBinVec};

pub mod quantum;
