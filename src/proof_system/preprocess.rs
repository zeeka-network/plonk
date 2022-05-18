// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Methods to preprocess the constraint system for use in a proof

use dusk_bls12_381::BlsScalar;
use merlin::Transcript;

use crate::commitment_scheme::CommitKey;
use crate::constraint_system::TurboComposer;
use crate::error::Error;
use crate::fft::{EvaluationDomain, Evaluations, Polynomial};
use crate::gpu::LockedFFTKernel;
use crate::plonkup::PreprocessedLookupTable;
use crate::proof_system::{widget, ProverKey};

/// Struct that contains all selector and permutation [`Polynomials`]s
pub(crate) struct Polynomials {
    // selector polynomials defining arithmetic circuits
    q_m: Polynomial,
    q_l: Polynomial,
    q_r: Polynomial,
    q_o: Polynomial,
    q_c: Polynomial,

    // additional selector for 3-input gates added for efficiency of
    // implementation
    q_4: Polynomial,

    // selector polynomial which activates the lookup gates
    // --> if (ith gate is lookup_gate) q_k[i] = 1 else q_k[i] = 0
    q_k: Polynomial,

    // additional selectors for different kinds of circuits added for
    // efficiency of implementation
    q_arith: Polynomial,              // arithmetic circuits
    q_range: Polynomial,              // range proofs
    q_logic: Polynomial,              // boolean operations
    q_fixed_group_add: Polynomial,    // ecc circuits
    q_variable_group_add: Polynomial, // ecc circuits

    // copy permutation polynomials
    s_sigma_1: Polynomial,
    s_sigma_2: Polynomial,
    s_sigma_3: Polynomial,
    s_sigma_4: Polynomial, // for q_4
}

impl TurboComposer {
    /// Pads the circuit to the next power of two
    ///
    /// # Note:
    ///
    /// `diff` is the difference between circuit size and next power of two
    fn pad(&mut self, diff: usize) {
        // Add a zero variable to circuit
        let zero_scalar = BlsScalar::zero();
        let zero_var = Self::constant_zero();

        let zeroes_scalar = vec![zero_scalar; diff];
        let zeroes_var = vec![zero_var; diff];

        self.q_m.extend(zeroes_scalar.iter());
        self.q_l.extend(zeroes_scalar.iter());
        self.q_r.extend(zeroes_scalar.iter());
        self.q_o.extend(zeroes_scalar.iter());
        self.q_c.extend(zeroes_scalar.iter());
        self.q_4.extend(zeroes_scalar.iter());
        self.q_k.extend(zeroes_scalar.iter());
        self.q_arith.extend(zeroes_scalar.iter());
        self.q_range.extend(zeroes_scalar.iter());
        self.q_logic.extend(zeroes_scalar.iter());
        self.q_fixed_group_add.extend(zeroes_scalar.iter());
        self.q_variable_group_add.extend(zeroes_scalar.iter());

        self.a_w.extend(zeroes_var.iter());
        self.b_w.extend(zeroes_var.iter());
        self.c_w.extend(zeroes_var.iter());
        self.d_w.extend(zeroes_var.iter());

        self.n += diff;
    }

    /// Checks that all of the wires of the composer have the same length.
    fn check_poly_same_len(&self) -> Result<(), Error> {
        let k = self.q_m.len();

        if self.q_o.len() == k
            && self.q_l.len() == k
            && self.q_r.len() == k
            && self.q_c.len() == k
            && self.q_4.len() == k
            && self.q_k.len() == k
            && self.q_arith.len() == k
            && self.q_range.len() == k
            && self.q_logic.len() == k
            && self.q_fixed_group_add.len() == k
            && self.q_variable_group_add.len() == k
            && self.a_w.len() == k
            && self.b_w.len() == k
            && self.c_w.len() == k
            && self.d_w.len() == k
        {
            Ok(())
        } else {
            Err(Error::MismatchedPolyLen)
        }
    }

    /// These are the parts of preprocessing that the prover must compute
    /// Although the prover does not need the verification key, he must compute
    /// the commitments in order to seed the transcript, allowing both the
    /// prover and verifier to have the same view
    pub(crate) fn preprocess_prover(
        &mut self,
        commit_key: &CommitKey,
        transcript: &mut Transcript,
    ) -> Result<ProverKey, Error> {
        let (_, selectors, preprocessed_table, domain) =
            self.preprocess_shared(commit_key, transcript)?;

        // The polynomial needs an evaluation domain of 4n.
        // Plus, adding the blinding factors translates to
        // the polynomial not fitting in 4n, so now we need
        // 8n, the next power of 2
        let domain_8n = EvaluationDomain::new(8 * domain.size())?;

        let q_m_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.q_m),
            domain_8n,
        );
        let q_l_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.q_l),
            domain_8n,
        );
        let q_r_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.q_r),
            domain_8n,
        );
        let q_o_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.q_o),
            domain_8n,
        );
        let q_c_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.q_c),
            domain_8n,
        );
        let q_4_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.q_4),
            domain_8n,
        );
        let q_k_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.q_k),
            domain_8n,
        );
        let q_arith_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.q_arith),
            domain_8n,
        );
        let q_range_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.q_range),
            domain_8n,
        );
        let q_logic_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.q_logic),
            domain_8n,
        );
        let q_fixed_group_add_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.q_fixed_group_add),
            domain_8n,
        );
        let q_variable_group_add_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.q_variable_group_add),
            domain_8n,
        );

        let s_sigma_1_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.s_sigma_1),
            domain_8n,
        );
        let s_sigma_2_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.s_sigma_2),
            domain_8n,
        );
        let s_sigma_3_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.s_sigma_3),
            domain_8n,
        );
        let s_sigma_4_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&selectors.s_sigma_4),
            domain_8n,
        );

        let table_1_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&preprocessed_table.t_1.2),
            domain_8n,
        );
        let table_2_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&preprocessed_table.t_2.2),
            domain_8n,
        );
        let table_3_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&preprocessed_table.t_3.2),
            domain_8n,
        );
        let table_4_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&preprocessed_table.t_4.2),
            domain_8n,
        );
        // XXX: Remove this and compute it on the fly
        let linear_eval_8n = Evaluations::from_vec_and_domain(
            domain_8n.coset_fft(&[BlsScalar::zero(), BlsScalar::one()]),
            domain_8n,
        );

        // Prover Key for arithmetic circuits
        let arithmetic_prover_key = widget::arithmetic::ProverKey {
            q_m: (selectors.q_m, q_m_eval_8n),
            q_l: (selectors.q_l.clone(), q_l_eval_8n.clone()),
            q_r: (selectors.q_r.clone(), q_r_eval_8n.clone()),
            q_o: (selectors.q_o, q_o_eval_8n),
            q_c: (selectors.q_c.clone(), q_c_eval_8n.clone()),
            q_4: (selectors.q_4, q_4_eval_8n),
            q_arith: (selectors.q_arith, q_arith_eval_8n),
        };

        // Prover Key for range circuits
        let range_prover_key = widget::range::ProverKey {
            q_range: (selectors.q_range, q_range_eval_8n),
        };

        // Prover Key for logic circuits
        let logic_prover_key = widget::logic::ProverKey {
            q_c: (selectors.q_c.clone(), q_c_eval_8n.clone()),
            q_logic: (selectors.q_logic, q_logic_eval_8n),
        };

        // Prover Key for ecc circuits
        let ecc_prover_key = widget::ecc::scalar_mul::fixed_base::ProverKey {
            q_l: (selectors.q_l, q_l_eval_8n),
            q_r: (selectors.q_r, q_r_eval_8n),
            q_c: (selectors.q_c, q_c_eval_8n),
            q_fixed_group_add: (
                selectors.q_fixed_group_add,
                q_fixed_group_add_eval_8n,
            ),
        };

        // Prover Key for permutation argument
        let permutation_prover_key = widget::permutation::ProverKey {
            s_sigma_1: (selectors.s_sigma_1, s_sigma_1_eval_8n),
            s_sigma_2: (selectors.s_sigma_2, s_sigma_2_eval_8n),
            s_sigma_3: (selectors.s_sigma_3, s_sigma_3_eval_8n),
            s_sigma_4: (selectors.s_sigma_4, s_sigma_4_eval_8n),
            linear_evaluations: linear_eval_8n,
        };

        // Prover Key for curve addition
        let curve_addition_prover_key =
            widget::ecc::curve_addition::ProverKey {
                q_variable_group_add: (
                    selectors.q_variable_group_add,
                    q_variable_group_add_eval_8n,
                ),
            };

        // Prover key for lookup operations
        let lookup_prover_key = widget::lookup::ProverKey {
            q_k: (selectors.q_k, q_k_eval_8n),
            table_1: (
                preprocessed_table.t_1.0,
                preprocessed_table.t_1.2,
                table_1_eval_8n,
            ),
            table_2: (
                preprocessed_table.t_2.0,
                preprocessed_table.t_2.2,
                table_2_eval_8n,
            ),
            table_3: (
                preprocessed_table.t_3.0,
                preprocessed_table.t_3.2,
                table_3_eval_8n,
            ),
            table_4: (
                preprocessed_table.t_4.0,
                preprocessed_table.t_4.2,
                table_4_eval_8n,
            ),
        };

        let prover_key = ProverKey {
            n: domain.size(),
            arithmetic: arithmetic_prover_key,
            logic: logic_prover_key,
            range: range_prover_key,
            permutation: permutation_prover_key,
            variable_base: curve_addition_prover_key,
            fixed_base: ecc_prover_key,
            lookup: lookup_prover_key,
            // Compute 8n evaluations for X^n -1
            v_h_coset_8n: domain_8n
                .compute_vanishing_poly_over_coset(domain.size() as u64),
        };

        Ok(prover_key)
    }

    /// The verifier only requires the commitments in order to verify a
    /// [`Proof`](super::Proof) We can therefore speed up preprocessing for the
    /// verifier by skipping the FFTs needed to compute the 8n evaluations.
    pub(crate) fn preprocess_verifier(
        &mut self,
        commit_key: &CommitKey,
        transcript: &mut Transcript,
    ) -> Result<widget::VerifierKey, Error> {
        let (verifier_key, _, _, _) =
            self.preprocess_shared(commit_key, transcript)?;
        Ok(verifier_key)
    }

    /// Both the [`Prover`](super::Prover) and [`Verifier`](super::Verifier)
    /// must perform IFFTs on the selector polynomials and permutation
    /// polynomials in order to commit to them and have the same transcript
    /// view.
    fn preprocess_shared(
        &mut self,
        commit_key: &CommitKey,
        transcript: &mut Transcript,
    ) -> Result<
        (
            widget::VerifierKey,
            Polynomials,
            PreprocessedLookupTable,
            EvaluationDomain,
        ),
        Error,
    > {
        // FIXME total_size requires documentation
        // https://github.com/dusk-network/plonk/issues/580
        let total_size = core::cmp::max(self.n, self.lookup_table.0.len());

        let domain = EvaluationDomain::new(total_size)?;

        // Check that the length of the wires is consistent.
        self.check_poly_same_len()?;

        // 1. Pad circuit to a power of two
        self.pad(domain.size as usize - self.n);

        let mut kern = Some(LockedFFTKernel::new(false));

        let mut q_m_coeffs = self.q_m.clone();
        assert_eq!(q_m_coeffs.len() as u64, domain.size);
        let mut q_l_coeffs = self.q_l.clone();
        assert_eq!(q_l_coeffs.len() as u64, domain.size);
        let mut q_r_coeffs = self.q_r.clone();
        assert_eq!(q_r_coeffs.len() as u64, domain.size);
        let mut q_o_coeffs = self.q_o.clone();
        let mut q_c_coeffs = self.q_c.clone();
        let mut q_4_coeffs = self.q_4.clone();
        let mut q_k_coeffs = self.q_k.clone();
        let mut q_arith_coeffs = self.q_arith.clone();
        let mut q_range_coeffs = self.q_range.clone();
        let mut q_logic_coeffs = self.q_logic.clone();
        let mut q_fixed_group_add_coeffs = self.q_fixed_group_add.clone();
        let mut q_variable_grou_add_coeffs = self.q_variable_group_add.clone();
        domain.many_ifft(
            &mut [
                &mut q_m_coeffs,
                &mut q_l_coeffs,
                &mut q_r_coeffs,
                &mut q_o_coeffs,
                &mut q_c_coeffs,
                &mut q_4_coeffs,
                &mut q_k_coeffs,
                &mut q_arith_coeffs,
                &mut q_range_coeffs,
                &mut q_logic_coeffs,
                &mut q_fixed_group_add_coeffs,
                &mut q_variable_grou_add_coeffs,
            ],
            &mut kern,
        );
        drop(kern);

        let q_m_poly = Polynomial::from_coefficients_vec(q_m_coeffs);
        let q_l_poly = Polynomial::from_coefficients_vec(q_l_coeffs);
        let q_r_poly = Polynomial::from_coefficients_vec(q_r_coeffs);
        let q_o_poly = Polynomial::from_coefficients_vec(q_o_coeffs);
        let q_c_poly = Polynomial::from_coefficients_vec(q_c_coeffs);
        let q_4_poly = Polynomial::from_coefficients_vec(q_4_coeffs);
        let q_k_poly = Polynomial::from_coefficients_vec(q_k_coeffs);
        let q_arith_poly = Polynomial::from_coefficients_vec(q_arith_coeffs);
        let q_range_poly = Polynomial::from_coefficients_vec(q_range_coeffs);
        let q_logic_poly = Polynomial::from_coefficients_vec(q_logic_coeffs);
        let q_fixed_group_add_poly =
            Polynomial::from_coefficients_vec(q_fixed_group_add_coeffs);
        let q_variable_group_add_poly =
            Polynomial::from_coefficients_vec(q_variable_grou_add_coeffs);

        // 2. Compute the sigma polynomials
        let [s_sigma_1_poly, s_sigma_2_poly, s_sigma_3_poly, s_sigma_4_poly] =
            self.perm.compute_sigma_polynomials(self.n, &domain);

        // ==== 5n Start ====
        let q_m_poly_commit = commit_key.commit(&q_m_poly).unwrap_or_default();
        let q_l_poly_commit = commit_key.commit(&q_l_poly).unwrap_or_default();
        let q_r_poly_commit = commit_key.commit(&q_r_poly).unwrap_or_default();
        let q_o_poly_commit = commit_key.commit(&q_o_poly).unwrap_or_default();
        let q_c_poly_commit = commit_key.commit(&q_c_poly).unwrap_or_default();
        let q_4_poly_commit = commit_key.commit(&q_4_poly).unwrap_or_default();
        let q_k_poly_commit = commit_key.commit(&q_k_poly).unwrap_or_default();
        let q_arith_poly_commit =
            commit_key.commit(&q_arith_poly).unwrap_or_default();
        let q_range_poly_commit =
            commit_key.commit(&q_range_poly).unwrap_or_default();
        let q_logic_poly_commit =
            commit_key.commit(&q_logic_poly).unwrap_or_default();
        let q_fixed_group_add_poly_commit = commit_key
            .commit(&q_fixed_group_add_poly)
            .unwrap_or_default();
        let q_variable_group_add_poly_commit = commit_key
            .commit(&q_variable_group_add_poly)
            .unwrap_or_default();
        // ==== 5n End =====

        // ==== sigma start ====
        let s_sigma_1_poly_commit = commit_key.commit(&s_sigma_1_poly)?;
        let s_sigma_2_poly_commit = commit_key.commit(&s_sigma_2_poly)?;
        let s_sigma_3_poly_commit = commit_key.commit(&s_sigma_3_poly)?;
        let s_sigma_4_poly_commit = commit_key.commit(&s_sigma_4_poly)?;
        // ==== sigma end ====

        // 3. Preprocess the lookup table, this generates T_1, T_2, T_3 and T_4
        let preprocessed_table = PreprocessedLookupTable::preprocess(
            &self.lookup_table,
            commit_key,
            domain.size() as u32,
        )?;

        // Verifier Key for arithmetic circuits
        let arithmetic_verifier_key = widget::arithmetic::VerifierKey {
            q_m: q_m_poly_commit,
            q_l: q_l_poly_commit,
            q_r: q_r_poly_commit,
            q_o: q_o_poly_commit,
            q_c: q_c_poly_commit,
            q_4: q_4_poly_commit,
            q_arith: q_arith_poly_commit,
        };
        // Verifier Key for range circuits
        let range_verifier_key = widget::range::VerifierKey {
            q_range: q_range_poly_commit,
        };
        // Verifier Key for logic circuits
        let logic_verifier_key = widget::logic::VerifierKey {
            q_c: q_c_poly_commit,
            q_logic: q_logic_poly_commit,
        };
        // Verifier Key for ecc circuits
        let ecc_verifier_key =
            widget::ecc::scalar_mul::fixed_base::VerifierKey {
                q_l: q_l_poly_commit,
                q_r: q_r_poly_commit,
                q_fixed_group_add: q_fixed_group_add_poly_commit,
            };
        // Verifier Key for curve addition circuits
        let curve_addition_verifier_key =
            widget::ecc::curve_addition::VerifierKey {
                q_variable_group_add: q_variable_group_add_poly_commit,
            };

        // Verifier Key for lookup operations
        let lookup_verifier_key = widget::lookup::VerifierKey {
            q_k: q_k_poly_commit,
            table_1: preprocessed_table.t_1.1,
            table_2: preprocessed_table.t_2.1,
            table_3: preprocessed_table.t_3.1,
            table_4: preprocessed_table.t_4.1,
        };
        // Verifier Key for permutation argument
        let permutation_verifier_key = widget::permutation::VerifierKey {
            s_sigma_1: s_sigma_1_poly_commit,
            s_sigma_2: s_sigma_2_poly_commit,
            s_sigma_3: s_sigma_3_poly_commit,
            s_sigma_4: s_sigma_4_poly_commit,
        };

        let verifier_key = widget::VerifierKey {
            n: self.gates(),
            arithmetic: arithmetic_verifier_key,
            logic: logic_verifier_key,
            range: range_verifier_key,
            fixed_base: ecc_verifier_key,
            variable_base: curve_addition_verifier_key,
            permutation: permutation_verifier_key,
            lookup: lookup_verifier_key,
        };

        let selectors = Polynomials {
            q_m: q_m_poly,
            q_l: q_l_poly,
            q_r: q_r_poly,
            q_o: q_o_poly,
            q_c: q_c_poly,
            q_4: q_4_poly,
            q_k: q_k_poly,
            q_arith: q_arith_poly,
            q_range: q_range_poly,
            q_logic: q_logic_poly,
            q_fixed_group_add: q_fixed_group_add_poly,
            q_variable_group_add: q_variable_group_add_poly,
            s_sigma_1: s_sigma_1_poly,
            s_sigma_2: s_sigma_2_poly,
            s_sigma_3: s_sigma_3_poly,
            s_sigma_4: s_sigma_4_poly,
        };

        // Add the circuit description to the transcript
        verifier_key.seed_transcript(transcript);

        Ok((verifier_key, selectors, preprocessed_table, domain))
    }
}

#[cfg(feature = "std")]
#[cfg(test)]
mod test {
    use crate::constraint_system::helper::*;

    use super::*;

    #[test]
    /// Tests that the circuit gets padded to the correct length
    /// XXX: We can do this test without dummy_gadget method
    fn test_plonkup_pad() {
        let mut composer: TurboComposer = TurboComposer::new();
        dummy_gadget_plonkup(100, &mut composer);

        // Pad the circuit to next power of two
        let next_pow_2 = composer.n.next_power_of_two() as u64;
        composer.pad(next_pow_2 as usize - composer.n);

        let size = composer.n;
        assert!(size.is_power_of_two());
        assert_eq!(composer.q_m.len(), size);
        assert_eq!(composer.q_l.len(), size);
        assert_eq!(composer.q_o.len(), size);
        assert_eq!(composer.q_r.len(), size);
        assert_eq!(composer.q_c.len(), size);
        assert_eq!(composer.q_k.len(), size);
        assert_eq!(composer.q_arith.len(), size);
        assert_eq!(composer.q_range.len(), size);
        assert_eq!(composer.q_logic.len(), size);
        assert_eq!(composer.q_fixed_group_add.len(), size);
        assert_eq!(composer.q_variable_group_add.len(), size);
        assert_eq!(composer.a_w.len(), size);
        assert_eq!(composer.b_w.len(), size);
        assert_eq!(composer.c_w.len(), size);
    }
}
