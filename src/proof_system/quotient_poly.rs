// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use alloc::vec::Vec;

use dusk_bls12_381::BlsScalar;
#[cfg(feature = "std")]
use rayon::prelude::*;

use crate::gpu::LockedFFTKernel;
use crate::{
    error::Error,
    fft::{EvaluationDomain, Polynomial},
    proof_system::ProverKey,
};

/// Computes the Quotient [`Polynomial`] given the [`EvaluationDomain`], a
/// [`ProverKey`] and some other info.
pub(crate) fn compute(
    domain: &EvaluationDomain,
    prover_key: &ProverKey,
    z_poly: &Polynomial,
    p_poly: &Polynomial,
    (a_w_poly, b_w_poly, c_w_poly, d_w_poly): (
        &Polynomial,
        &Polynomial,
        &Polynomial,
        &Polynomial,
    ),
    f_poly: &Polynomial,
    t_poly: &Polynomial,
    h_1_poly: &Polynomial,
    h_2_poly: &Polynomial,
    public_inputs_poly: &Polynomial,
    (
        alpha,
        beta,
        gamma,
        delta,
        epsilon,
        zeta,
        range_challenge,
        logic_challenge,
        fixed_base_challenge,
        var_base_challenge,
        lookup_challenge,
    ): &(
        BlsScalar,
        BlsScalar,
        BlsScalar,
        BlsScalar,
        BlsScalar,
        BlsScalar,
        BlsScalar,
        BlsScalar,
        BlsScalar,
        BlsScalar,
        BlsScalar,
    ),
    kern: &mut Option<LockedFFTKernel>,
) -> Result<Polynomial, Error> {
    let mut poly_scaled_1 =
        first_lagrange_poly_scaled(domain, BlsScalar::one());
    let mut l1_alpha_sq_evals =
        first_lagrange_poly_scaled(domain, alpha.square());
    domain.many_ifft(&mut [&mut poly_scaled_1, &mut l1_alpha_sq_evals], kern);

    // Compute 8n evals
    let domain_8n = EvaluationDomain::new(8 * domain.size())?;
    let mut z_eval_8n = z_poly.coeffs.clone();
    z_eval_8n.resize(domain_8n.size(), BlsScalar::zero());
    let mut p_eval_8n = p_poly.coeffs.clone();
    p_eval_8n.resize(domain_8n.size(), BlsScalar::zero());
    let mut t_eval_8n = t_poly.coeffs.clone();
    t_eval_8n.resize(domain_8n.size(), BlsScalar::zero());
    let mut f_eval_8n = f_poly.coeffs.clone();
    f_eval_8n.resize(domain_8n.size(), BlsScalar::zero());
    let mut h_1_eval_8n = h_1_poly.coeffs.clone();
    h_1_eval_8n.resize(domain_8n.size(), BlsScalar::zero());
    let mut h_2_eval_8n = h_2_poly.coeffs.clone();
    h_2_eval_8n.resize(domain_8n.size(), BlsScalar::zero());
    let mut a_w_eval_8n = a_w_poly.coeffs.clone();
    a_w_eval_8n.resize(domain_8n.size(), BlsScalar::zero());
    let mut b_w_eval_8n = b_w_poly.coeffs.clone();
    b_w_eval_8n.resize(domain_8n.size(), BlsScalar::zero());
    let mut c_w_eval_8n = c_w_poly.coeffs.clone();
    c_w_eval_8n.resize(domain_8n.size(), BlsScalar::zero());
    let mut d_w_eval_8n = d_w_poly.coeffs.clone();
    d_w_eval_8n.resize(domain_8n.size(), BlsScalar::zero());
    let mut public_eval_8n = public_inputs_poly.coeffs.clone();
    public_eval_8n.resize(domain_8n.size(), BlsScalar::zero());
    let mut l1_eval_8n = poly_scaled_1;
    l1_eval_8n.resize(domain_8n.size(), BlsScalar::zero());
    l1_alpha_sq_evals.resize(domain_8n.size(), BlsScalar::zero());
    domain_8n.many_coset_fft(
        &mut [
            &mut z_eval_8n,
            &mut p_eval_8n,
            &mut t_eval_8n,
            &mut f_eval_8n,
            &mut h_1_eval_8n,
            &mut h_2_eval_8n,
            &mut a_w_eval_8n,
            &mut b_w_eval_8n,
            &mut c_w_eval_8n,
            &mut d_w_eval_8n,
            &mut public_eval_8n,
            &mut l1_eval_8n,
            &mut l1_alpha_sq_evals,
        ],
        kern,
    );

    // let l1_alpha_sq_evals = domain_8n.coset_fft(&ploy_scaled_alpha_square);

    // let public_eval_8n = domain_8n.coset_fft(public_inputs_poly);
    //
    // let l1_eval_8n = domain_8n.coset_fft(&poly_scaled_1);

    // let mut z_eval_8n = domain_8n.coset_fft(z_poly);
    // let mut p_eval_8n = domain_8n.coset_fft(p_poly);
    // let mut t_eval_8n = domain_8n.coset_fft(t_poly);
    //
    // let f_eval_8n = domain_8n.coset_fft(f_poly);
    //
    // let mut h_1_eval_8n = domain_8n.coset_fft(h_1_poly);
    // let mut h_2_eval_8n = domain_8n.coset_fft(h_2_poly);
    //
    // let mut a_w_eval_8n = domain_8n.coset_fft(a_w_poly);
    // let mut b_w_eval_8n = domain_8n.coset_fft(b_w_poly);
    // let c_w_eval_8n = domain_8n.coset_fft(c_w_poly);
    // let mut d_w_eval_8n = domain_8n.coset_fft(d_w_poly);

    for i in 0..8 {
        z_eval_8n.push(z_eval_8n[i]);
        p_eval_8n.push(p_eval_8n[i]);
        t_eval_8n.push(t_eval_8n[i]);
        h_1_eval_8n.push(h_1_eval_8n[i]);
        h_2_eval_8n.push(h_2_eval_8n[i]);
        a_w_eval_8n.push(a_w_eval_8n[i]);
        b_w_eval_8n.push(b_w_eval_8n[i]);
        // c_w_eval_8n push not required
        d_w_eval_8n.push(d_w_eval_8n[i]);
    }

    let t_1 = compute_circuit_satisfiability_equation(
        domain,
        (
            range_challenge,
            logic_challenge,
            fixed_base_challenge,
            var_base_challenge,
            lookup_challenge,
        ),
        prover_key,
        (&a_w_eval_8n, &b_w_eval_8n, &c_w_eval_8n, &d_w_eval_8n),
        zeta,
        (delta, epsilon),
        &f_eval_8n,
        &p_eval_8n,
        &t_eval_8n,
        &h_1_eval_8n,
        &h_2_eval_8n,
        &public_eval_8n,
        &l1_eval_8n,
    );

    let t_2 = compute_permutation_checks(
        domain_8n.size(),
        prover_key,
        (&a_w_eval_8n, &b_w_eval_8n, &c_w_eval_8n, &d_w_eval_8n),
        &z_eval_8n,
        (alpha, beta, gamma),
        &l1_alpha_sq_evals,
    );

    #[cfg(not(feature = "std"))]
    let range = (0..domain_8n.size()).into_iter();

    #[cfg(feature = "std")]
    let range = (0..domain_8n.size()).into_par_iter();

    let mut quotient: Vec<_> = range
        .map(|i| {
            let numerator = t_1[i] + t_2[i];
            let denominator = prover_key.v_h_coset_8n()[i];
            numerator * denominator.invert().unwrap()
        })
        .collect();
    domain_8n.many_coset_ifft(&mut [&mut quotient], kern);


    Ok(Polynomial::from_coefficients_vec(
        quotient,
    ))
}

// Ensures that the circuit is satisfied
// Ensures that the circuit is satisfied
fn compute_circuit_satisfiability_equation(
    domain: &EvaluationDomain,
    (
        range_challenge,
        logic_challenge,
        fixed_base_challenge,
        var_base_challenge,
        lookup_challenge,
    ): (&BlsScalar, &BlsScalar, &BlsScalar, &BlsScalar, &BlsScalar),
    prover_key: &ProverKey,
    (a_w_eval_8n, b_w_eval_8n, c_w_eval_8n, d_w_eval_8n): (
        &[BlsScalar],
        &[BlsScalar],
        &[BlsScalar],
        &[BlsScalar],
    ),
    zeta: &BlsScalar,
    (delta, epsilon): (&BlsScalar, &BlsScalar),
    f_eval_8n: &[BlsScalar],
    p_eval_8n: &[BlsScalar],
    t_eval_8n: &[BlsScalar],
    h_1_eval_8n: &[BlsScalar],
    h_2_eval_8n: &[BlsScalar],
    public_eval_8n: &[BlsScalar],
    l1_eval_8n: &[BlsScalar],
) -> Vec<BlsScalar> {
    // let domain_8n = EvaluationDomain::new(8 * domain.size()).unwrap();
    // let public_eval_8n = domain_8n.coset_fft(pi_poly);
    //
    // let l1_eval_8n = domain_8n.coset_fft(&poly_scaled_1);

    #[cfg(not(feature = "std"))]
    let range = (0..(8 * domain.size())).into_iter();

    #[cfg(feature = "std")]
    let range = (0..(8 * domain.size())).into_par_iter();

    let t: Vec<_> = range
        .map(|i| {
            let a_w = &a_w_eval_8n[i];
            let b_w = &b_w_eval_8n[i];
            let c_w = &c_w_eval_8n[i];
            let d_w = &d_w_eval_8n[i];
            let a_w_next = &a_w_eval_8n[i + 8];
            let b_w_next = &b_w_eval_8n[i + 8];
            let d_w_next = &d_w_eval_8n[i + 8];
            let pi = &public_eval_8n[i];
            let p = &p_eval_8n[i];
            let p_next = &p_eval_8n[i + 8];
            let fi = &f_eval_8n[i];
            let ti = &t_eval_8n[i];
            let ti_next = &t_eval_8n[i + 8];
            let h1 = &h_1_eval_8n[i];
            let h2 = &h_2_eval_8n[i];
            let h1_next = &h_1_eval_8n[i + 8];
            let l1i = &l1_eval_8n[i];

            let a = prover_key
                .arithmetic
                .compute_quotient_i(i, a_w, b_w, c_w, d_w);

            let b = prover_key.range.compute_quotient_i(
                i,
                range_challenge,
                a_w,
                b_w,
                c_w,
                d_w,
                d_w_next,
            );

            let c = prover_key.logic.compute_quotient_i(
                i,
                logic_challenge,
                a_w,
                a_w_next,
                b_w,
                b_w_next,
                c_w,
                d_w,
                d_w_next,
            );

            let d = prover_key.fixed_base.compute_quotient_i(
                i,
                fixed_base_challenge,
                a_w,
                a_w_next,
                b_w,
                b_w_next,
                c_w,
                d_w,
                d_w_next,
            );

            let e = prover_key.variable_base.compute_quotient_i(
                i,
                var_base_challenge,
                a_w,
                a_w_next,
                b_w,
                b_w_next,
                c_w,
                d_w,
                d_w_next,
            );

            let f = prover_key.lookup.compute_quotient_i(
                i,
                lookup_challenge,
                a_w,
                b_w,
                c_w,
                d_w,
                fi,
                p,
                p_next,
                ti,
                ti_next,
                h1,
                h1_next,
                h2,
                l1i,
                (delta, epsilon),
                zeta,
            );

            (a + pi) + b + c + d + e + f
        })
        .collect();
    t
}

fn compute_permutation_checks(
    domain_8n_size: usize,
    prover_key: &ProverKey,
    (a_w_eval_8n, b_w_eval_8n, c_w_eval_8n, d_w_eval_8n): (
        &[BlsScalar],
        &[BlsScalar],
        &[BlsScalar],
        &[BlsScalar],
    ),
    z_eval_8n: &[BlsScalar],
    (alpha, beta, gamma): (&BlsScalar, &BlsScalar, &BlsScalar),
    l1_alpha_sq_evals: &[BlsScalar],
) -> Vec<BlsScalar> {
    // let domain_8n = EvaluationDomain::new(8 * domain.size()).unwrap();
    // let l1_poly_alpha =
    //     compute_first_lagrange_poly_scaled(domain, alpha.square());
    // let l1_alpha_sq_evals = domain_8n.coset_fft(&ploy_scaled_alpha_square);

    #[cfg(not(feature = "std"))]
    let range = (0..domain_8n_size).into_iter();

    #[cfg(feature = "std")]
    let range = (0..domain_8n_size).into_par_iter();

    let t: Vec<_> = range
        .map(|i| {
            prover_key.permutation.compute_quotient_i(
                i,
                &a_w_eval_8n[i],
                &b_w_eval_8n[i],
                &c_w_eval_8n[i],
                &d_w_eval_8n[i],
                &z_eval_8n[i],
                &z_eval_8n[i + 8],
                alpha,
                &l1_alpha_sq_evals[i],
                beta,
                gamma,
            )
        })
        .collect();
    t
}

#[inline]
fn first_lagrange_poly_scaled(
    domain: &EvaluationDomain,
    scale: BlsScalar,
) -> Vec<BlsScalar> {
    let mut x_evals = vec![BlsScalar::zero(); domain.size()];
    x_evals[0] = scale;
    x_evals
}
