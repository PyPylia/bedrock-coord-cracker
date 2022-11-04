//#![allow(improper_ctypes_definitions)]
#![feature(inline_const)]
#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

extern crate alloc;

pub mod legacy;
pub mod xoroshiro;

// use cuda_std::*;