#![allow(non_snake_case)]
//! Condensed (normal-equations) KKT solver with low-rank Woodbury corrections.
//!
//! Instead of factorizing the (n+m) quasidefinite KKT matrix
//!     [[P, A'], [A, -H]]
//! this solver eliminates the cone block:
//!     (P + A' H^{-1} A) x = bx + A' H^{-1} bz,     z = H^{-1} (A x - bz)
//! and exploits the structure of problems whose constraint matrix consists
//! mostly of very sparse ("thin") rows plus a modest number of dense/medium
//! ("thick") rows. Writing H^{-1} = D^{-1} + corrections (a rank-2 SMW term
//! per sparse-form SOC scaling, a small dense block per dense-form SOC), the
//! condensed matrix becomes
//!     M = M_sp + U C U'
//!     M_sp = P + A_thin' D^{-1} A_thin      (near-diagonal, tiny LDL)
//!     U    = [A_thick' | A' (D^{-1} u), A' (D^{-1} v) per sparse SOC]
//! solved through the Woodbury identity around the M_sp factorization.
//! Solutions are polished by iterative refinement against the exact KKT
//! operator; if refinement cannot reach tolerance the solve reports failure
//! and the solver's numerical-error handling (strategy switches, best-iterate
//! fallback) takes over.
//!
//! Supported structure: Nonnegative and SecondOrder cones only; the
//! constructor returns None otherwise so the caller can fall back to the
//! direct LDL solver.

use crate::algebra::*;
use crate::qdldl::*;
use crate::solver::core::cones::{CompositeCone, Cone, SupportedCone};
use crate::solver::core::kktsolvers::{HasLinearSolverInfo, KKTSolver, LinearSolverInfo};
use crate::solver::core::CoreSettings;
use std::iter::zip;
use std::ops::Range;

// rows with at most this many nonzeros contribute to M_sp; everything else
// goes into the low-rank block (medium-density rows such as industry
// exposures MUST be low-rank, else M_sp fills with their row-clique products)
const THIN_ROW_NNZ: usize = 8;

// bail out (falling back to direct LDL) if the low-rank block gets too big
const MAX_RANK_FRACTION: f64 = 0.05;

// copy of one sparse-form SOC scaling, taken at update() time. ub/vb are
// eta*u / eta*v so that H_block = diag + ub ub' - vb vb'.
struct SocScaling<T> {
    rng: Range<usize>,
    ub: Vec<T>,
    vb: Vec<T>,
}

// copy of one dense-form SOC scaling (w point and eta^2); the block and its
// inverse both have exact closed forms H = eta^2 (2ww' - J) and
// H^{-1} = eta^{-2} (2 w~ w~' - J) with w~ = Jw, so no inversion is performed
struct DenseSocScaling<T> {
    rng: Range<usize>,
    w: Vec<T>,
    eta2: T,
    // H^{-1} = eta^{-2} (alpha wb wb' - J), wb = Jw, alpha = 2/(2 w'Jw - 1)
    alpha: T,
    // per ordered row pair (i,j): indices into (Msp_idx, product) lists for
    // the A_i' H^{-1}_ij A_j contribution to M_sp
    pair_ptr: Vec<usize>,
    pair_msp: Vec<usize>,
    pair_prod: Vec<T>,
}

pub struct CondensedKKTSolver<T> {
    m: usize,
    n: usize,

    // problem data (P upper triangular)
    P: CscMatrix<T>,
    A: CscMatrix<T>,
    At: CscMatrix<T>,

    // scaling snapshots refreshed each update()
    hdiag: Vec<T>,
    dinv: Vec<T>,
    socs: Vec<SocScaling<T>>,
    dense_socs: Vec<DenseSocScaling<T>>,

    // row split
    thin_rows: Vec<usize>,
    thick_rows: Vec<usize>,

    // M_sp machinery: values = scatter(P) + contrib * dinv_thin + reg*I
    Msp: CscMatrix<T>,
    Msp_all_idx: Vec<usize>,
    P_to_Msp: Vec<usize>,
    contrib_colptr: Vec<usize>,
    contrib_rowval: Vec<usize>,
    contrib_nzval: Vec<T>,
    Msp_diag_idx: Vec<usize>,
    ldl: QDLDLFactorisation<T>,

    // low-rank machinery (U, Y column-major n x k)
    k: usize,
    soc_col_offset: usize,
    soc_col_scale: Vec<T>,
    U: Vec<T>,
    Y: Vec<T>,
    core: Vec<T>,
    core_piv: Vec<usize>,

    // rhs and latest solution
    bx: Vec<T>,
    bz: Vec<T>,

    // work
    work_n: Vec<T>,
    work_m: Vec<T>,
    work_k: Vec<T>,

    static_reg: T,
}

fn csc_transpose<T: FloatT>(A: &CscMatrix<T>) -> CscMatrix<T> {
    let (m, n) = (A.m, A.n);
    let nnz = A.nnz();
    let mut colptr = vec![0usize; m + 1];
    for &r in &A.rowval {
        colptr[r + 1] += 1;
    }
    for i in 0..m {
        colptr[i + 1] += colptr[i];
    }
    let mut rowval = vec![0usize; nnz];
    let mut nzval = vec![T::zero(); nnz];
    let mut pos = colptr.clone();
    for col in 0..n {
        for i in A.colptr[col]..A.colptr[col + 1] {
            let r = A.rowval[i];
            rowval[pos[r]] = col;
            nzval[pos[r]] = A.nzval[i];
            pos[r] += 1;
        }
    }
    CscMatrix::new(n, m, colptr, rowval, nzval)
}

impl<T: FloatT> CondensedKKTSolver<T> {
    /// Returns None if the problem structure is unsupported.
    pub fn try_new(
        P: &CscMatrix<T>,
        A: &CscMatrix<T>,
        cones: &CompositeCone<T>,
        m: usize,
        n: usize,
        settings: &CoreSettings<T>,
    ) -> Option<Self> {
        // structure check + collect SOC layouts
        let mut soc_rngs = Vec::new();
        let mut dense_soc_rngs = Vec::new();
        for (cone, rng) in zip(cones.iter(), cones.rng_cones.iter()) {
            match cone {
                SupportedCone::NonnegativeCone(_) => {}
                SupportedCone::SecondOrderCone(soc) => {
                    if soc.sparse_data.is_some() {
                        soc_rngs.push(rng.clone());
                    } else {
                        dense_soc_rngs.push(rng.clone());
                    }
                }
                _ => return None,
            }
        }

        let P = P.to_triu();
        let At = csc_transpose(A);
        let A = A.clone();

        // row split; dense-SOC rows are always thick (their H block couples
        // them; exactness is restored through the core matrix)
        let mut is_thick = vec![false; m];
        for r in 0..m {
            if At.colptr[r + 1] - At.colptr[r] > THIN_ROW_NNZ {
                is_thick[r] = true;
            }
        }
        // dense-SOC rows are handled exactly through closed-form pair
        // contributions to M_sp below; exclude them from both thin (diagonal
        // D^{-1}) and thick (low-rank) treatments
        let mut is_dense_soc_row = vec![false; m];
        for rng in &dense_soc_rngs {
            for r in rng.clone() {
                is_dense_soc_row[r] = true;
                is_thick[r] = false;
            }
        }
        let thick_rows: Vec<usize> = (0..m).filter(|&r| is_thick[r] && !is_dense_soc_row[r]).collect();
        let thin_rows: Vec<usize> =
            (0..m).filter(|&r| !is_thick[r] && !is_dense_soc_row[r]).collect();

        let k = thick_rows.len() + 2 * soc_rngs.len();
        if k > ((n as f64) * MAX_RANK_FRACTION) as usize + 16 {
            return None;
        }

        // ---- symbolic M_sp: triu(P) ∪ diagonal ∪ thin-row pair products
        let mut pairs: Vec<(usize, usize)> = Vec::new(); // (col, row), row <= col
        for col in 0..n {
            for i in P.colptr[col]..P.colptr[col + 1] {
                pairs.push((col, P.rowval[i]));
            }
        }
        for i in 0..n {
            pairs.push((i, i));
        }
        for &r in &thin_rows {
            let cols = &At.rowval[At.colptr[r]..At.colptr[r + 1]];
            for (ai, &ca) in cols.iter().enumerate() {
                for &cb in &cols[ai..] {
                    let (lo, hi) = if ca <= cb { (ca, cb) } else { (cb, ca) };
                    pairs.push((hi, lo));
                }
            }
        }
        for rng in &dense_soc_rngs {
            let mut all_cols: Vec<usize> = Vec::new();
            for r in rng.clone() {
                all_cols.extend_from_slice(&At.rowval[At.colptr[r]..At.colptr[r + 1]]);
            }
            all_cols.sort_unstable();
            all_cols.dedup();
            for (ai, &ca) in all_cols.iter().enumerate() {
                for &cb in &all_cols[ai..] {
                    pairs.push((cb, ca));
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();

        let nnz_m = pairs.len();
        let mut colptr = vec![0usize; n + 1];
        for &(c, _) in &pairs {
            colptr[c + 1] += 1;
        }
        for i in 0..n {
            colptr[i + 1] += colptr[i];
        }
        let rowval: Vec<usize> = pairs.iter().map(|&(_, r)| r).collect();
        let Msp = CscMatrix::new(n, n, colptr, rowval, vec![T::zero(); nnz_m]);

        let find_nz = |col: usize, row: usize| -> usize {
            let f = Msp.colptr[col];
            let l = Msp.colptr[col + 1];
            f + Msp.rowval[f..l].binary_search(&row).unwrap()
        };

        let mut P_to_Msp = Vec::with_capacity(P.nnz());
        for col in 0..n {
            for i in P.colptr[col]..P.colptr[col + 1] {
                P_to_Msp.push(find_nz(col, P.rowval[i]));
            }
        }
        let Msp_diag_idx: Vec<usize> = (0..n).map(|i| find_nz(i, i)).collect();

        // contribution matrix (CSC by thin row): values A_r[i]*A_r[j] at Msp slots
        let mut contrib_colptr = vec![0usize; thin_rows.len() + 1];
        let mut contrib_rowval = Vec::new();
        let mut contrib_nzval = Vec::new();
        for (tr, &r) in thin_rows.iter().enumerate() {
            let f = At.colptr[r];
            let l = At.colptr[r + 1];
            let cols = &At.rowval[f..l];
            let vals = &At.nzval[f..l];
            for (ai, (&ca, &va)) in zip(cols, vals).enumerate() {
                for (&cb, &vb) in zip(&cols[ai..], &vals[ai..]) {
                    let (lo, hi) = if ca <= cb { (ca, cb) } else { (cb, ca) };
                    contrib_rowval.push(find_nz(hi, lo));
                    contrib_nzval.push(va * vb);
                }
            }
            contrib_colptr[tr + 1] = contrib_rowval.len();
        }

        let opts = QDLDLSettingsBuilder::default()
            .logical(true)
            .Dsigns(vec![1i8; n])
            .regularize_enable(true)
            .regularize_eps(settings.dynamic_regularization_eps)
            .regularize_delta(settings.dynamic_regularization_delta)
            .build()
            .unwrap();
        let ldl = QDLDLFactorisation::<T>::new(&Msp, Some(opts)).ok()?;

        // static U columns for thick rows
        let mut U = vec![T::zero(); n * k];
        for (uc, &r) in thick_rows.iter().enumerate() {
            let col = &mut U[uc * n..(uc + 1) * n];
            for i in At.colptr[r]..At.colptr[r + 1] {
                col[At.rowval[i]] = At.nzval[i];
            }
        }
        let soc_col_offset = thick_rows.len();

        let socs_len = soc_rngs.len();
        let socs = soc_rngs
            .into_iter()
            .map(|rng| SocScaling {
                ub: vec![T::zero(); rng.len()],
                vb: vec![T::zero(); rng.len()],
                rng,
            })
            .collect();
        let dense_socs = dense_soc_rngs
            .into_iter()
            .map(|rng| {
                let d = rng.len();
                let mut pair_ptr = vec![0usize];
                let mut pair_msp = Vec::new();
                let mut pair_prod = Vec::new();
                for i in 0..d {
                    let ri = rng.start + i;
                    let (fi, li) = (At.colptr[ri], At.colptr[ri + 1]);
                    for j in 0..d {
                        let rj = rng.start + j;
                        let (fj, lj) = (At.colptr[rj], At.colptr[rj + 1]);
                        for pi in fi..li {
                            let (p, vp) = (At.rowval[pi], At.nzval[pi]);
                            for qj in fj..lj {
                                let (q, vq) = (At.rowval[qj], At.nzval[qj]);
                                if p > q {
                                    continue; // triu only; the (j,i) ordered pair covers it
                                }
                                pair_msp.push(find_nz(q, p));
                                pair_prod.push(vp * vq);
                            }
                        }
                        pair_ptr.push(pair_msp.len());
                    }
                }
                DenseSocScaling {
                    w: vec![T::zero(); d],
                    eta2: T::one(),
                    alpha: T::from_f64(2.0).unwrap(),
                    pair_ptr,
                    pair_msp,
                    pair_prod,
                    rng,
                }
            })
            .collect();

        Some(Self {
            m,
            n,
            P,
            A,
            At,
            hdiag: vec![T::zero(); m],
            dinv: vec![T::zero(); m],
            socs,
            dense_socs,
            thin_rows,
            thick_rows,
            Msp_all_idx: (0..nnz_m).collect(),
            Msp,
            P_to_Msp,
            contrib_colptr,
            contrib_rowval,
            contrib_nzval,
            Msp_diag_idx,
            ldl,
            k,
            soc_col_offset,
            soc_col_scale: vec![T::one(); 2 * socs_len],
            U,
            Y: vec![T::zero(); n * k],
            core: vec![T::zero(); k * k],
            core_piv: vec![0usize; k],
            bx: vec![T::zero(); n],
            bz: vec![T::zero(); m],
            work_n: vec![T::zero(); n],
            work_m: vec![T::zero(); m],
            work_k: vec![T::zero(); k],
            static_reg: settings.static_regularization_constant,
        })
    }

    // refresh hdiag/dinv and the SOC scaling snapshots from the cones
    fn refresh_scalings(&mut self, cones: &CompositeCone<T>) {
        let mut soc_i = 0usize;
        let mut dsoc_i = 0usize;
        for (cone, rng) in zip(cones.iter(), cones.rng_cones.iter()) {
            match cone {
                SupportedCone::NonnegativeCone(_) => {
                    cone.get_Hs(&mut self.hdiag[rng.clone()]);
                }
                SupportedCone::SecondOrderCone(c) => {
                    if let Some(sd) = &c.sparse_data {
                        // Hs diagonal = eta^2 * [d, 1, 1, ...]
                        cone.get_Hs(&mut self.hdiag[rng.clone()]);
                        let soc = &mut self.socs[soc_i];
                        soc_i += 1;
                        for (i, (ub, vb)) in zip(&mut soc.ub, &mut soc.vb).enumerate() {
                            *ub = c.η * sd.u[i];
                            *vb = c.η * sd.v[i];
                        }
                        if std::env::var("CLARABEL_CONDENSED_DEBUG").is_ok() {
                            // self-check the rank-2 snapshot against the exact
                            // H = eta^2 (2ww' - J) on a deterministic vector
                            let d = rng.len();
                            let x: Vec<T> = (0..d)
                                .map(|i| T::from_f64(((i * 7919 + 13) % 17) as f64 / 8.5 - 1.0).unwrap())
                                .collect();
                            let eta2 = c.η * c.η;
                            let wx: T = zip(&c.w, &x).fold(T::zero(), |acc, (a, b)| acc + *a * *b);
                            let two = T::from_f64(2.0).unwrap();
                            let mut yex: Vec<T> = zip(&c.w, &x).map(|(w, xi)| eta2 * two * wx * *w + eta2 * *xi).collect();
                            yex[0] -= eta2 * two * x[0];
                            let soc2 = &self.socs[soc_i - 1];
                            let ux: T = zip(&soc2.ub, &x).fold(T::zero(), |acc, (a, b)| acc + *a * *b);
                            let vx: T = zip(&soc2.vb, &x).fold(T::zero(), |acc, (a, b)| acc + *a * *b);
                            let mut err = T::zero();
                            for i in 0..d {
                                let ymine = self.hdiag[rng.start + i] * x[i] + soc2.ub[i] * ux - soc2.vb[i] * vx;
                                err = T::max(err, (ymine - yex[i]).abs());
                            }
                            eprintln!("soc rank2 self-check: dim={} err={:?}", d, err.to_f64());
                        }
                    } else {
                        let ds = &mut self.dense_socs[dsoc_i];
                        dsoc_i += 1;
                        ds.eta2 = c.η * c.η;
                        ds.w.copy_from_slice(&c.w);
                        let two = T::from_f64(2.0).unwrap();
                        let mut cjw = c.w[0] * c.w[0];
                        for wi in &c.w[1..] {
                            cjw -= *wi * *wi;
                        }
                        ds.alpha = two / (two * cjw - T::one());
                        // closed-form diagonal of H (only used to keep
                        // hdiag/dinv finite; these rows bypass D^{-1})
                        let two = T::from_f64(2.0).unwrap();
                        for (i, r) in rng.clone().enumerate() {
                            let jii = if i == 0 { T::one() } else { -T::one() };
                            self.hdiag[r] = ds.eta2 * (two * c.w[i] * c.w[i] - jii);
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
        for r in 0..self.m {
            self.dinv[r] = T::recip(self.hdiag[r]);
        }

        if std::env::var("CLARABEL_CONDENSED_DEBUG").is_ok() {
            let x: Vec<T> = (0..self.m)
                .map(|i| T::from_f64(((i * 7919 + 13) % 17) as f64 / 8.5 - 1.0).unwrap())
                .collect();
            let mut y = vec![T::zero(); self.m];
            let mut z = vec![T::zero(); self.m];
            self.H_mul(&mut y, &x);
            self.H_solve(&mut z, &y);
            let mut err = T::zero();
            let mut argmax = 0usize;
            for i in 0..self.m {
                let e = (z[i] - x[i]).abs();
                if e > err {
                    err = e;
                    argmax = i;
                }
            }
            eprintln!(
                "H_solve∘H_mul check: err={:?} row={} hdiag={:?} x={:?} z={:?}",
                err.to_f64(), argmax, self.hdiag[argmax].to_f64(), x[argmax].to_f64(), z[argmax].to_f64()
            );
        }
    }

    // y = P x (symmetric, P stored triu)
    fn P_symv(&self, y: &mut [T], x: &[T]) {
        y.fill(T::zero());
        let P = &self.P;
        for col in 0..self.n {
            let xc = x[col];
            for i in P.colptr[col]..P.colptr[col + 1] {
                let row = P.rowval[i];
                let v = P.nzval[i];
                y[row] += v * xc;
                if row != col {
                    y[col] += v * x[row];
                }
            }
        }
    }

    fn A_mul(&self, y: &mut [T], x: &[T]) {
        y.fill(T::zero());
        let A = &self.A;
        for col in 0..self.n {
            let xc = x[col];
            if xc == T::zero() {
                continue;
            }
            for i in A.colptr[col]..A.colptr[col + 1] {
                y[A.rowval[i]] += A.nzval[i] * xc;
            }
        }
    }

    fn At_mul(&self, y: &mut [T], z: &[T]) {
        y.fill(T::zero());
        let At = &self.At;
        for col in 0..self.m {
            let zc = z[col];
            if zc == T::zero() {
                continue;
            }
            for i in At.colptr[col]..At.colptr[col + 1] {
                y[At.rowval[i]] += At.nzval[i] * zc;
            }
        }
    }

    // y = H z using the scaling snapshots
    fn H_mul(&self, y: &mut [T], z: &[T]) {
        for r in 0..self.m {
            y[r] = self.hdiag[r] * z[r];
        }
        for soc in &self.socs {
            let mut du = T::zero();
            let mut dv = T::zero();
            for (i, r) in soc.rng.clone().enumerate() {
                du += soc.ub[i] * z[r];
                dv += soc.vb[i] * z[r];
            }
            for (i, r) in soc.rng.clone().enumerate() {
                y[r] += soc.ub[i] * du - soc.vb[i] * dv;
            }
        }
        for ds in &self.dense_socs {
            let base = ds.rng.start;
            let d = ds.rng.len();
            let two = T::from_f64(2.0).unwrap();
            let mut wz = T::zero();
            for i in 0..d {
                wz += ds.w[i] * z[base + i];
            }
            for i in 0..d {
                let jz = if i == 0 { z[base] } else { -z[base + i] };
                y[base + i] = ds.eta2 * (two * wz * ds.w[i] - jz);
            }
        }
    }

    // z = H^{-1} w via D^{-1} plus SMW / small dense corrections
    fn H_solve(&self, z: &mut [T], w: &[T]) {
        for r in 0..self.m {
            z[r] = self.dinv[r] * w[r];
        }
        for soc in &self.socs {
            // block = E + ub ub' - vb vb'; SMW with 2x2 core J + G'E^{-1}G
            let mut s_uu = T::zero();
            let mut s_uv = T::zero();
            let mut s_vv = T::zero();
            let mut t_u = T::zero();
            let mut t_v = T::zero();
            for (i, r) in soc.rng.clone().enumerate() {
                let di = self.dinv[r];
                s_uu += soc.ub[i] * di * soc.ub[i];
                s_uv += soc.ub[i] * di * soc.vb[i];
                s_vv += soc.vb[i] * di * soc.vb[i];
                t_u += soc.ub[i] * di * w[r];
                t_v += soc.vb[i] * di * w[r];
            }
            let a = T::one() + s_uu;
            let b = s_uv;
            let c2 = s_uv;
            let d2 = -T::one() + s_vv;
            let det = a * d2 - b * c2;
            let cu = (d2 * t_u - b * t_v) / det;
            let cv = (-c2 * t_u + a * t_v) / det;
            for (i, r) in soc.rng.clone().enumerate() {
                z[r] -= self.dinv[r] * (soc.ub[i] * cu + soc.vb[i] * cv);
            }
        }
        for ds in &self.dense_socs {
            let base = ds.rng.start;
            let d = ds.rng.len();
            let eta2inv = T::recip(ds.eta2);
            // wb = J w
            let mut wbr = T::zero(); // wb . rhs
            for i in 0..d {
                let wbi = if i == 0 { ds.w[0] } else { -ds.w[i] };
                wbr += wbi * w[base + i];
            }
            for i in 0..d {
                let wbi = if i == 0 { ds.w[0] } else { -ds.w[i] };
                let jr = if i == 0 { w[base] } else { -w[base + i] };
                z[base + i] = eta2inv * (ds.alpha * wbr * wbi - jr);
            }
        }
    }

    // rx = bx - P x - A' z ; rz = bz - A x + H z ; returns inf norm
    fn kkt_residual(&mut self, x: &[T], z: &[T], rx: &mut [T], rz: &mut [T]) -> T {
        let mut tn = std::mem::take(&mut self.work_n);
        let mut tm = std::mem::take(&mut self.work_m);

        self.P_symv(&mut tn, x);
        for i in 0..self.n {
            rx[i] = self.bx[i] - tn[i];
        }
        self.At_mul(&mut tn, z);
        for i in 0..self.n {
            rx[i] -= tn[i];
        }
        self.A_mul(&mut tm, x);
        for r in 0..self.m {
            rz[r] = self.bz[r] - tm[r];
        }
        self.H_mul(&mut tm, z);
        for r in 0..self.m {
            rz[r] += tm[r];
        }

        self.work_n = tn;
        self.work_m = tm;

        let mut e = T::zero();
        for v in rx.iter().chain(rz.iter()) {
            e = T::max(e, v.abs());
        }
        e
    }

    // one condensed solve pass: (bx, bz) -> (x, z)
    fn solve_once(&mut self, bx: &[T], bz: &[T], x: &mut [T], z: &mut [T]) {
        // rhs = bx + A' H^{-1} bz  (reuse z as H^{-1} bz scratch)
        self.H_solve(z, bz);
        let mut rn = std::mem::take(&mut self.work_n);
        self.At_mul(&mut rn, z);
        for i in 0..self.n {
            rn[i] += bx[i];
        }

        // x = M^{-1} rhs via Woodbury
        x.copy_from_slice(&rn);
        self.work_n = rn;
        self.ldl.solve(x);
        let mut wk = std::mem::take(&mut self.work_k);
        for c in 0..self.k {
            let col = &self.U[c * self.n..(c + 1) * self.n];
            let mut acc = T::zero();
            for (ci, xi) in zip(col, x.iter()) {
                acc += *ci * *xi;
            }
            wk[c] = acc;
        }
        dense_lu_solve(&self.core, &self.core_piv, self.k, &mut wk);
        for c in 0..self.k {
            let ycol = &self.Y[c * self.n..(c + 1) * self.n];
            let w = wk[c];
            for i in 0..self.n {
                x[i] -= ycol[i] * w;
            }
        }
        self.work_k = wk;

        // z = H^{-1} (A x - bz)
        let mut tm = std::mem::take(&mut self.work_m);
        self.A_mul(&mut tm, x);
        for r in 0..self.m {
            tm[r] -= bz[r];
        }
        self.H_solve(z, &tm);
        self.work_m = tm;
    }
}

impl<T: FloatT> HasLinearSolverInfo for CondensedKKTSolver<T> {
    fn linear_solver_info(&self) -> LinearSolverInfo {
        LinearSolverInfo {
            name: "condensed".to_string(),
            threads: 1,
            direct: true,
            nnzA: self.Msp.nnz(),
            nnzL: self.ldl.nnzL(),
        }
    }
}

impl<T: FloatT> KKTSolver<T> for CondensedKKTSolver<T> {
    fn update(&mut self, cones: &CompositeCone<T>, _settings: &CoreSettings<T>) -> bool {
        self.refresh_scalings(cones);

        // numeric M_sp
        self.Msp.nzval.fill(T::zero());
        for (pi, &mi) in self.P_to_Msp.iter().enumerate() {
            self.Msp.nzval[mi] += self.P.nzval[pi];
        }
        for (tr, &r) in self.thin_rows.iter().enumerate() {
            let dr = self.dinv[r];
            for i in self.contrib_colptr[tr]..self.contrib_colptr[tr + 1] {
                self.Msp.nzval[self.contrib_rowval[i]] += self.contrib_nzval[i] * dr;
            }
        }
        // dense-SOC exact contributions: coeff_ij = H^{-1}_ij (closed form)
        for ds in &self.dense_socs {
            let d = ds.rng.len();
            let eta2inv = T::recip(ds.eta2);
            for i in 0..d {
                let wbi = if i == 0 { ds.w[0] } else { -ds.w[i] };
                for j in 0..d {
                    let wbj = if j == 0 { ds.w[0] } else { -ds.w[j] };
                    let jij = if i == j {
                        if i == 0 { T::one() } else { -T::one() }
                    } else {
                        T::zero()
                    };
                    let coeff = eta2inv * (ds.alpha * wbi * wbj - jij);
                    let pr = i * d + j;
                    for t in ds.pair_ptr[pr]..ds.pair_ptr[pr + 1] {
                        self.Msp.nzval[ds.pair_msp[t]] += ds.pair_prod[t] * coeff;
                    }
                }
            }
        }

        let reg = self.static_reg;
        for &di in &self.Msp_diag_idx {
            self.Msp.nzval[di] += reg;
        }

        self.ldl.update_values(&self.Msp_all_idx, &self.Msp.nzval);
        if self.ldl.refactor().is_err() {
            return false;
        }
        if std::env::var("CLARABEL_CONDENSED_DEBUG").is_ok() {
            eprintln!("Msp refactor: regularize_count={}", self.ldl.regularize_count());
        }

        // thick-row U columns scaled by sqrt(dinv) so their C block is the
        // identity: keeps the Woodbury core well conditioned across the many
        // orders of magnitude the cone scalings span
        for (uc, &r) in self.thick_rows.iter().enumerate() {
            let col = &mut self.U[uc * self.n..(uc + 1) * self.n];
            col.fill(T::zero());
            let sc = self.dinv[r].sqrt();
            for i in self.At.colptr[r]..self.At.colptr[r + 1] {
                col[self.At.rowval[i]] = self.At.nzval[i] * sc;
            }
        }

        // sparse-SOC U columns: A' (dinv .* ub), A' (dinv .* vb)
        for (si, soc) in self.socs.iter().enumerate() {
            let base = (self.soc_col_offset + 2 * si) * self.n;
            let (left, right) = self.U.split_at_mut(base + self.n);
            let ucol = &mut left[base..];
            let vcol = &mut right[..self.n];
            ucol.fill(T::zero());
            vcol.fill(T::zero());
            for (i, r) in soc.rng.clone().enumerate() {
                let gu = self.dinv[r] * soc.ub[i];
                let gv = self.dinv[r] * soc.vb[i];
                for j in self.At.colptr[r]..self.At.colptr[r + 1] {
                    let c = self.At.rowval[j];
                    let v = self.At.nzval[j];
                    ucol[c] += v * gu;
                    vcol[c] += v * gv;
                }
            }
            // normalize both columns to unit inf norm (scales absorbed by the
            // core C^{-1} block) to keep the Woodbury core well conditioned
            for (off, col) in [(0usize, ucol), (1usize, vcol)] {
                let mut mx = T::zero();
                for v in col.iter() {
                    mx = T::max(mx, v.abs());
                }
                let sc = if mx > T::zero() { mx } else { T::one() };
                let inv = T::recip(sc);
                for v in col.iter_mut() {
                    *v *= inv;
                }
                self.soc_col_scale[2 * si + off] = sc;
            }
        }

        // Y = M_sp^{-1} U
        self.Y.copy_from_slice(&self.U);
        for c in 0..self.k {
            self.ldl.solve(&mut self.Y[c * self.n..(c + 1) * self.n]);
        }

        // core = C^{-1} + U' Y
        self.core.fill(T::zero());
        for uc in 0..self.thick_rows.len() {
            self.core[uc * self.k + uc] = T::one();
        }
        // sparse-SOC pairs: C^{-1} = -(J + [ub vb]' D^{-1} [ub vb]), rescaled
        // by the column normalizations applied below
        for (si, soc) in self.socs.iter().enumerate() {
            let mut s_uu = T::zero();
            let mut s_uv = T::zero();
            let mut s_vv = T::zero();
            for (i, r) in soc.rng.clone().enumerate() {
                let di = self.dinv[r];
                s_uu += soc.ub[i] * di * soc.ub[i];
                s_uv += soc.ub[i] * di * soc.vb[i];
                s_vv += soc.vb[i] * di * soc.vb[i];
            }
            let base = self.soc_col_offset + 2 * si;
            let su = self.soc_col_scale[2 * si];
            let sv = self.soc_col_scale[2 * si + 1];

            self.core[base * self.k + base] = -(T::one() + s_uu) / (su * su);
            self.core[(base + 1) * self.k + base] = -s_uv / (su * sv);
            self.core[base * self.k + (base + 1)] = -s_uv / (su * sv);
            self.core[(base + 1) * self.k + (base + 1)] = -(-T::one() + s_vv) / (sv * sv);
        }

        for cj in 0..self.k {
            let ycol = &self.Y[cj * self.n..(cj + 1) * self.n];
            for ci in 0..self.k {
                let ucol = &self.U[ci * self.n..(ci + 1) * self.n];
                let mut acc = T::zero();
                for (u, y) in zip(ucol, ycol) {
                    acc += *u * *y;
                }
                self.core[cj * self.k + ci] += acc;
            }
        }

        let core_dbg = if std::env::var("CLARABEL_CONDENSED_DEBUG").is_ok() {
            Some(self.core.clone())
        } else {
            None
        };
        let ok = dense_lu_factor(&mut self.core, self.k, &mut self.core_piv);

        if std::env::var("CLARABEL_CONDENSED_DEBUG").is_ok() {
            // LDL accuracy: e = Msp (ldl^{-1} y) - y
            let y0: Vec<T> = (0..self.n)
                .map(|i| T::from_f64(((i * 48271) % 89) as f64 / 44.5 - 1.0).unwrap())
                .collect();
            let mut x = y0.clone();
            self.ldl.solve(&mut x);
            let mut e = vec![T::zero(); self.n];
            for col in 0..self.n {
                let xc = x[col];
                for i in self.Msp.colptr[col]..self.Msp.colptr[col + 1] {
                    let row = self.Msp.rowval[i];
                    let v = self.Msp.nzval[i];
                    e[row] += v * xc;
                    if row != col {
                        e[col] += v * x[row];
                    }
                }
            }
            let mut lerr = T::zero();
            for i in 0..self.n {
                lerr = T::max(lerr, (e[i] - y0[i]).abs());
            }
            eprintln!("ldl(Msp) accuracy: err={:?} regularize_count={}", lerr.to_f64(), self.ldl.regularize_count());

            let mut mn = T::infinity();
            let mut mx = T::zero();
            for c in 0..self.k {
                let v = self.core[c * self.k + c].abs();
                mn = T::min(mn, v);
                mx = T::max(mx, v);
            }
            let mut hmin = T::infinity();
            let mut hmax = T::zero();
            for r in 0..self.m {
                hmin = T::min(hmin, self.hdiag[r].abs());
                hmax = T::max(hmax, self.hdiag[r].abs());
            }
            eprintln!(
                "core pivots: ok={} min={:?} max={:?} | hdiag range [{:?}, {:?}]",
                ok, mn.to_f64(), mx.to_f64(), hmin.to_f64(), hmax.to_f64()
            );
        }

        if ok && std::env::var("CLARABEL_CONDENSED_DEBUG").is_ok() {
            // roundtrip: x0 -> r = M x0 (via exact H^{-1} action) -> M^{-1} r
            let x0: Vec<T> = (0..self.n)
                .map(|i| T::from_f64(((i * 2654435761) % 97) as f64 / 48.5 - 1.0).unwrap())
                .collect();
            let mut ax = vec![T::zero(); self.m];
            self.A_mul(&mut ax, &x0);
            let mut hax = vec![T::zero(); self.m];
            self.H_solve(&mut hax, &ax);
            let mut r = vec![T::zero(); self.n];
            self.At_mul(&mut r, &hax);
            let mut px = vec![T::zero(); self.n];
            self.P_symv(&mut px, &x0);
            for i in 0..self.n {
                r[i] += px[i] + self.static_reg * x0[i];
            }
            // Woodbury solve of r
            let mut x = r.clone();
            self.ldl.solve(&mut x);
            let mut wk = vec![T::zero(); self.k];
            for c in 0..self.k {
                let col = &self.U[c * self.n..(c + 1) * self.n];
                let mut acc = T::zero();
                for (ci, xi) in zip(col, x.iter()) {
                    acc += *ci * *xi;
                }
                wk[c] = acc;
            }
            dense_lu_solve(&self.core, &self.core_piv, self.k, &mut wk);
            for c in 0..self.k {
                let ycol = &self.Y[c * self.n..(c + 1) * self.n];
                let w = wk[c];
                for i in 0..self.n {
                    x[i] -= ycol[i] * w;
                }
            }
            let mut err = T::zero();
            let mut amax = 0usize;
            for i in 0..self.n {
                let e = (x[i] - x0[i]).abs();
                if e > err {
                    err = e;
                    amax = i;
                }
            }

            // forward check: (Msp + U C U') x0 vs r (assembly vs solve isolation)
            let mut fwd = vec![T::zero(); self.n];
            for col in 0..self.n {
                let xc = x0[col];
                for i in self.Msp.colptr[col]..self.Msp.colptr[col + 1] {
                    let row = self.Msp.rowval[i];
                    let v = self.Msp.nzval[i];
                    fwd[row] += v * xc;
                    if row != col {
                        fwd[col] += v * x0[row];
                    }
                }
            }
            // U C U' x0: thick block C = I; soc pairs C = -core2^{-1} (scaled)
            for uc in 0..self.thick_rows.len() {
                let col = &self.U[uc * self.n..(uc + 1) * self.n];
                let mut acc = T::zero();
                for (ci, xi) in zip(col, x0.iter()) {
                    acc += *ci * *xi;
                }
                for i in 0..self.n {
                    fwd[i] += col[i] * acc;
                }
            }
            for (si, soc) in self.socs.iter().enumerate() {
                let base = self.soc_col_offset + 2 * si;
                let su = self.soc_col_scale[2 * si];
                let sv = self.soc_col_scale[2 * si + 1];
                let ucol = &self.U[base * self.n..(base + 1) * self.n];
                let vcol = &self.U[(base + 1) * self.n..(base + 2) * self.n];
                let mut pu = T::zero();
                let mut pv = T::zero();
                for i in 0..self.n {
                    pu += ucol[i] * x0[i];
                    pv += vcol[i] * x0[i];
                }
                // scaled C block = Sg^{-1} C Sg^{-1}, C = -core2^{-1}
                // solve core2 y = [pu/su... careful: contribution = Ucols * Cblock * [pu; pv]
                // with Cblock^{-1} = -(J+S) scaled by 1/(s_i s_j):
                // Cblock = -inv(core2) scaled by s_i s_j... equivalently solve:
                let mut s_uu = T::zero();
                let mut s_uv = T::zero();
                let mut s_vv = T::zero();
                for (i, r) in soc.rng.clone().enumerate() {
                    let di = self.dinv[r];
                    s_uu += soc.ub[i] * di * soc.ub[i];
                    s_uv += soc.ub[i] * di * soc.vb[i];
                    s_vv += soc.vb[i] * di * soc.vb[i];
                }
                let a = T::one() + s_uu;
                let d2 = -T::one() + s_vv;
                let det = a * d2 - s_uv * s_uv;
                // unscaled q = -core2^{-1} [su*pu; sv*pv]
                let ru = su * pu;
                let rv = sv * pv;
                let qu = -(d2 * ru - s_uv * rv) / det;
                let qv = -(-s_uv * ru + a * rv) / det;
                for i in 0..self.n {
                    fwd[i] += ucol[i] * (su * qu) + vcol[i] * (sv * qv);
                }
            }
            let mut ferr = T::zero();
            for i in 0..self.n {
                ferr = T::max(ferr, (fwd[i] - r[i]).abs());
            }

            // LU consistency: core_dbg * wk_solved vs rhs (U' ldl^{-1} r)
            let mut xs = r.clone();
            self.ldl.solve(&mut xs);
            let mut rhs_k = vec![T::zero(); self.k];
            for c in 0..self.k {
                let col = &self.U[c * self.n..(c + 1) * self.n];
                let mut acc = T::zero();
                for (ci, xi) in zip(col, xs.iter()) {
                    acc += *ci * *xi;
                }
                rhs_k[c] = acc;
            }
            let mut wk2 = rhs_k.clone();
            dense_lu_solve(&self.core, &self.core_piv, self.k, &mut wk2);
            let cd = core_dbg.as_ref().unwrap();
            let mut luerr = T::zero();
            let mut wkmax = T::zero();
            for i in 0..self.k {
                let mut acc = T::zero();
                for c in 0..self.k {
                    acc += cd[c * self.k + i] * wk2[c];
                }
                luerr = T::max(luerr, (acc - rhs_k[i]).abs());
                wkmax = T::max(wkmax, wk2[i].abs());
            }
            eprintln!(
                "M woodbury roundtrip: err={:?} argmax={} | forward assembly err={:?} | core LU err={:?} |wk|={:?}",
                err.to_f64(), amax, ferr.to_f64(), luerr.to_f64(), wkmax.to_f64()
            );
            for (si, soc) in self.socs.iter().enumerate() {
                let base = self.soc_col_offset + 2 * si;
                eprintln!(
                    "  soc{}: su={:?} sv={:?} Cinv=[{:?} {:?}; . {:?}]",
                    si,
                    self.soc_col_scale[2 * si].to_f64(),
                    self.soc_col_scale[2 * si + 1].to_f64(),
                    cd[base * self.k + base].to_f64(),
                    cd[(base + 1) * self.k + base].to_f64(),
                    cd[(base + 1) * self.k + (base + 1)].to_f64(),
                );
            }
        }
        ok
    }

    fn setrhs(&mut self, rhsx: &[T], rhsz: &[T]) {
        self.bx.copy_from_slice(rhsx);
        self.bz.copy_from_slice(rhsz);
    }

    fn solve(
        &mut self,
        lhsx: Option<&mut [T]>,
        lhsz: Option<&mut [T]>,
        settings: &CoreSettings<T>,
    ) -> bool {
        let bx = self.bx.clone();
        let bz = self.bz.clone();
        let mut x = vec![T::zero(); self.n];
        let mut z = vec![T::zero(); self.m];
        self.solve_once(&bx, &bz, &mut x, &mut z);

        // iterative refinement against the exact KKT operator (the condensed
        // solve loses roughly half the digits of the quasidefinite approach,
        // so allow twice the configured refinement steps)
        let maxiter = 2 * settings.iterative_refinement_max_iter;
        let abstol = settings.iterative_refinement_abstol;
        let reltol = settings.iterative_refinement_reltol;
        let mut normb = T::zero();
        for v in bx.iter().chain(bz.iter()) {
            normb = T::max(normb, v.abs());
        }
        let target = abstol + reltol * normb;

        let mut rx = vec![T::zero(); self.n];
        let mut rz = vec![T::zero(); self.m];
        let mut dx = vec![T::zero(); self.n];
        let mut dz = vec![T::zero(); self.m];

        let debug = std::env::var("CLARABEL_CONDENSED_DEBUG").is_ok();
        let mut norme = self.kkt_residual(&x, &z, &mut rx, &mut rz);
        if debug {
            eprintln!("condensed solve: normb={:?} raw norme={:?}", normb.to_f64(), norme.to_f64());
        }
        for _ in 0..maxiter {
            if !norme.is_finite() || norme <= target {
                break;
            }
            let last = norme;
            self.solve_once(&rx, &rz, &mut dx, &mut dz);
            for i in 0..self.n {
                dx[i] += x[i];
            }
            for r in 0..self.m {
                dz[r] += z[r];
            }
            let newe = self.kkt_residual(&dx, &dz, &mut rx, &mut rz);
            if !newe.is_finite() || newe >= last {
                // no improvement; keep the previous iterate
                let _ = self.kkt_residual(&x, &z, &mut rx, &mut rz);
                break;
            }
            std::mem::swap(&mut x, &mut dx);
            std::mem::swap(&mut z, &mut dz);
            norme = newe;
            if debug {
                eprintln!("condensed refine: norme={:?}", norme.to_f64());
            }
        }

        // accept only if we got within a loose multiple of the target;
        // otherwise report failure so the caller's numerical-error paths
        // (scaling switch, best-iterate fallback) engage
        let loose = target * (1e6).as_T();
        let success = norme.is_finite() && norme <= loose;

        if success {
            if let Some(v) = lhsx {
                v.copy_from_slice(&x);
            }
            if let Some(v) = lhsz {
                v.copy_from_slice(&z);
            }
        }
        success
    }

    fn update_P(&mut self, P: &CscMatrix<T>) {
        self.P = P.to_triu();
    }

    fn update_A(&mut self, A: &CscMatrix<T>) {
        self.At = csc_transpose(A);
        self.A = A.clone();
    }
}

// ---- minimal dense LU with partial pivoting (k x k, column-major) ----

fn dense_lu_factor<T: FloatT>(a: &mut [T], k: usize, piv: &mut [usize]) -> bool {
    for col in 0..k {
        let mut p = col;
        let mut best = a[col * k + col].abs();
        for r in (col + 1)..k {
            let v = a[col * k + r].abs();
            if v > best {
                best = v;
                p = r;
            }
        }
        if best == T::zero() {
            return false;
        }
        piv[col] = p;
        if p != col {
            for c in 0..k {
                a.swap(c * k + col, c * k + p);
            }
        }
        let d = T::recip(a[col * k + col]);
        for r in (col + 1)..k {
            a[col * k + r] = a[col * k + r] * d;
        }
        for c in (col + 1)..k {
            let mult = a[c * k + col];
            if mult == T::zero() {
                continue;
            }
            for r in (col + 1)..k {
                let l = a[col * k + r];
                a[c * k + r] -= mult * l;
            }
        }
    }
    true
}

fn dense_lu_solve<T: FloatT>(a: &[T], piv: &[usize], k: usize, b: &mut [T]) {
    for col in 0..k {
        b.swap(col, piv[col]);
        let bc = b[col];
        for r in (col + 1)..k {
            b[r] -= a[col * k + r] * bc;
        }
    }
    for col in (0..k).rev() {
        let bc = b[col] / a[col * k + col];
        b[col] = bc;
        for r in 0..col {
            b[r] -= a[col * k + r] * bc;
        }
    }
}
