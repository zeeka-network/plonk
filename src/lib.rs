// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

#![doc = include_str!("../README.md")]
#![doc(
    html_logo_url = "https://lh3.googleusercontent.com/SmwswGxtgIANTbDrCOn5EKcRBnVdHjmYsHYxLq2HZNXWCQ9-fZyaea-bNgdX9eR0XGSqiMFi=w128-h128-e365"
)]
#![doc(html_favicon_url = "https://dusk.network/lib/img/favicon-16x16.png")]
//!<a href="https://codecov.io/gh/dusk-network/plonk">
//!  <img src="https://codecov.io/gh/dusk-network/plonk/branch/master/graph/badge.svg" />
//!</a>
//! <a href="https://travis-ci.com/dusk-network/plonk">
//! <img src="https://travis-ci.com/dusk-network/plonk.svg?branch=master" />
//! </a>
//! <a href="https://github.com/dusk-network/plonk">
//! <img alt="GitHub issues" src="https://img.shields.io/github/issues-raw/dusk-network/plonk?style=plastic">
//! </a>
//! <a href="https://github.com/dusk-network/plonk/blob/master/LICENSE">
//! <img alt="GitHub" src="https://img.shields.io/github/license/dusk-network/plonk?color=%230E55EF">
//! </a>
//!
//!
//! Permutations over Lagrange-bases for Oecumenical Noninteractive
//! arguments of Knowledge (PLONK) is a zero knowledge proof system.
//!
//! This protocol was created by:
//! - Ariel Gabizon (Protocol Labs),
//! - Zachary J. Williamson (Aztec Protocol)
//! - Oana Ciobotaru
//!
//! This crate contains a pure-rust implementation done by the [DuskNetwork
//! team](dusk.network) of this algorithm using as a reference implementation
//! this one done by the creators of the protocol:
//!
//! <https://github.com/AztecProtocol/barretenberg/blob/master/barretenberg/src/aztec/plonk/>

// Bitshift/Bitwise ops are allowed to gain performance.
#![allow(clippy::suspicious_arithmetic_impl)]
// Some structs do not have AddAssign or MulAssign impl.
#![allow(clippy::suspicious_op_assign_impl)]
// Witness have always the same names in respect to wires.
#![allow(clippy::many_single_char_names)]
// Bool expr are usually easier to read with match statements.
#![allow(clippy::match_bool)]
// We have quite some functions that require quite some args by it's nature.
// It can be refactored but for now, we avoid these warns.
#![allow(clippy::too_many_arguments)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

cfg_if::cfg_if!(
if #[cfg(feature = "alloc")] {
    /// `macro_use` will declare `vec!`. However, if `libstd` is present, then this
    /// is declared in the prelude and there will be a conflicting implementation.
    ///
    /// We might have `no_std + alloc` or `std + alloc`, but `macro_use` should be
    /// used only for `no_std`
    #[cfg_attr(not(feature = "std"), macro_use)]
    extern crate alloc;

    mod bit_iterator;
    mod permutation;
    mod util;

    pub mod circuit;
    pub mod constraint_system;
    pub mod plonkup;
});

mod fft;
mod multicore;
mod transcript;

pub mod commitment_scheme;
pub mod error;
mod gpu;
pub mod prelude;
pub mod proof_system;
mod test_utils;

#[doc = include_str!("../docs/notes-intro.md")]
pub mod notes {
    #[doc = include_str!("../docs/notes-commitments.md")]
    pub mod commitment_schemes {}
    #[doc = include_str!("../docs/notes-snark.md")]
    pub mod snark_construction {}
    #[doc = include_str!("../docs/notes-prove-verify.md")]
    pub mod prove_verify {}
    #[doc = include_str!("../docs/notes-KZG10.md")]
    pub mod kzg10_docs {}
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    use rand_core::OsRng;

    use crate::prelude::*;

    #[derive(Debug, Clone, Copy)]
    struct BenchCircuit {
        degree: usize,
    }

    impl<T> From<T> for BenchCircuit
    where
        T: Into<usize>,
    {
        fn from(degree: T) -> Self {
            Self {
                degree: 1 << degree.into(),
            }
        }
    }

    impl Circuit for BenchCircuit {
        const CIRCUIT_ID: [u8; 32] = [0xff; 32];

        fn gadget(
            &mut self,
            composer: &mut TurboComposer,
        ) -> Result<(), Error> {
            let mut a = BlsScalar::from(2u64);
            let mut b = BlsScalar::from(3u64);
            let mut c;

            while composer.gates() < self.padded_gates() {
                a += BlsScalar::one();
                b += BlsScalar::one();
                c = a * b + a + b + BlsScalar::one();

                let x = composer.append_witness(a);
                let y = composer.append_witness(b);
                let z = composer.append_witness(c);

                let constraint = Constraint::new()
                    .mult(1)
                    .left(1)
                    .right(1)
                    .output(-BlsScalar::one())
                    .constant(1)
                    .a(x)
                    .b(y)
                    .o(z);

                composer.append_gate(constraint);
            }

            Ok(())
        }

        fn public_inputs(&self) -> Vec<PublicInputValue> {
            vec![]
        }

        fn padded_gates(&self) -> usize {
            self.degree
        }
    }

    fn constraint_system_prove(
        circuit: &mut BenchCircuit,
        pp: &PublicParameters,
        pk: &ProverKey,
        label: &'static [u8],
    ) -> Proof {
        circuit
            .prove(pp, pk, label, &mut OsRng)
            .expect("Failed to prove bench circuit!")
    }


    #[test]
    #[cfg(feature = "big-tests")]
    fn test_big_circuit() {
        let max_degree = 27;
        let degree = 26;

        let rng = &mut rand_core::OsRng;
        let label = b"dusk-network";
        let pp = PublicParameters::setup(1 << max_degree, rng)
            .expect("Failed to create PP");

        println!("compile circuit");
        let compile_st = Instant::now();
        let mut circuit = BenchCircuit::from(degree as usize);
        let (pk, vd) =
            circuit.compile(&pp).expect("Failed to compile circuit!");
        println!("compile took {:?}", compile_st.elapsed());

        let size = circuit.padded_gates();
        let power = (size as f64).log2() as usize;
        println!("Prove 2^{} = {} gates", power, size);
        let mut prove_st = Instant::now();
        let proof = constraint_system_prove(&mut circuit, &pp, &pk, label);
        println!("Prove took: {:?}", prove_st.elapsed());

        println!("Verify 2^{} = {} gates", power, size);
        let mut verify_st = Instant::now();
        BenchCircuit::verify(&pp, &vd, &proof, &[], label)
            .expect("Failed to verify bench circuit!");
        println!("Verify took: {:?}", verify_st.elapsed());
    }
}
