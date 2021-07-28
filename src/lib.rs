//! A toolbox for classical and quantum LDPC codes.
//!
//! The crate is divided into three modules.
//!
//! The [classical module](classical) contains a [linear code](classical::LinearCode)
//! implementation and some decoders for it.
//!
//! For now, the [quantum module](quantum) contains only a [CSS code](quantum::CssCode)
//! implementation.
//!
//! Finally, the [noise model module](noise_model) contains a generic trait for noise generation.

pub mod classical;
pub mod noise_model;
pub mod quantum;
