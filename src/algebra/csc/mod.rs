#![allow(non_snake_case)]

mod core;
pub use self::core::*;
mod block_concatenate;
mod matrix_math;
pub(crate) use matrix_math::_csc_neg_At_and_A;
mod utils;
