// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use crate::commitment_scheme::{CommitKey, Commitment};
use crate::error::Error;
use crate::fft::{EvaluationDomain, Polynomial};
use crate::gpu::LockedFFTKernel;
use crate::plonkup::{LookupTable, MultiSet};

/// This table will be the preprocessed version of the
/// precomputed table, T, with arity 4. This structure
/// is passed to the proof alongside the table of witness
/// values.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct PreprocessedLookupTable {
    /// This is the circuit size
    pub n: u32,

    /// This is the first column in the preprocessed
    /// table containing a MultiSet, Commitments to the
    /// MultiSet and the coefficients as a Polynomial
    pub(crate) t_1: (MultiSet, Commitment, Polynomial),

    /// This is the second column in the preprocessed
    /// table containing a MultiSet, Commitments to the
    /// MultiSet and the coefficients as a Polynomial
    pub(crate) t_2: (MultiSet, Commitment, Polynomial),

    /// This is the third column in the preprocessed
    /// table containing a MultiSet, Commitments to the
    /// MultiSet and the coefficients as a Polynomial
    pub(crate) t_3: (MultiSet, Commitment, Polynomial),

    /// This is the fourth column in the preprocessed
    /// table containing a MultiSet, Commitments to the
    /// MultiSet and the coefficients as a Polynomial
    pub(crate) t_4: (MultiSet, Commitment, Polynomial),
}

impl PreprocessedLookupTable {
    /// This function takes in a precomputed look up table and
    /// pads it to the length of the circuit entries, as a power
    /// of 2. The function then interpolates a polynomial from the
    /// padded table and makes a commitment to the poly. The
    /// outputted struct will be used in the proof alongside our
    /// circuit witness table.
    pub fn preprocess(
        table: &LookupTable,
        commit_key: &CommitKey,
        n: u32,
        kern: &mut Option<LockedFFTKernel>,
    ) -> Result<Self, Error> {
        let domain: EvaluationDomain =
            EvaluationDomain::new(n as usize).unwrap();

        let columned_table = table.vec_to_multiset();
        let mut t_1 = columned_table.0;
        let mut t_2 = columned_table.1;
        let mut t_3 = columned_table.2;
        let mut t_4 = columned_table.3;

        t_1.pad(n);
        t_2.pad(n);
        t_3.pad(n);
        t_4.pad(n);
        let mut t_1_coeffs = t_1.0.clone();
        let mut t_2_coeffs = t_2.0.clone();
        let mut t_3_coeffs = t_3.0.clone();
        let mut t_4_coeffs = t_4.0.clone();
        domain.many_ifft(
            &mut [
                &mut t_1_coeffs,
                &mut t_2_coeffs,
                &mut t_3_coeffs,
                &mut t_4_coeffs,
            ],
            kern,
        );

        let t_1_poly = Polynomial::from_coefficients_vec(t_1_coeffs);
        let t_2_poly = Polynomial::from_coefficients_vec(t_2_coeffs);
        let t_3_poly = Polynomial::from_coefficients_vec(t_3_coeffs);
        let t_4_poly = Polynomial::from_coefficients_vec(t_4_coeffs);

        // let t_1_poly = t_1.to_polynomial(&domain);
        // let t_2_poly = t_2.to_polynomial(&domain);
        // let t_3_poly = t_3.to_polynomial(&domain);
        // let t_4_poly = t_4.to_polynomial(&domain);

        let t_1_commit = commit_key.commit(&t_1_poly)?;
        let t_2_commit = commit_key.commit(&t_2_poly)?;
        let t_3_commit = commit_key.commit(&t_3_poly)?;
        let t_4_commit = commit_key.commit(&t_4_poly)?;

        Ok(PreprocessedLookupTable {
            n,
            t_1: (t_1, t_1_commit, t_1_poly),
            t_2: (t_2, t_2_commit, t_2_poly),
            t_3: (t_3, t_3_commit, t_3_poly),
            t_4: (t_4, t_4_commit, t_4_poly),
        })
    }
}
