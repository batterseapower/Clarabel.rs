#![allow(non_snake_case)]

use super::ldlsolvers::config::LDLConfiguration;
use super::*;
use crate::solver::core::kktsolvers::{HasLinearSolverInfo, KKTSolver, LinearSolverInfo};
use crate::solver::core::{cones::*, CoreSettings};
use std::iter::zip;

// -------------------------------------
// KKTSolver using direct LDL factorisation
// -------------------------------------

// We require Send/Sync here to allow pyo3 builds to share
// solver objects between threads.

pub(crate) type BoxedDirectLDLSolver<T> = Box<dyn DirectLDLSolver<T> + Send + Sync>;

pub struct DirectLDLKKTSolver<T> {
    // problem dimensions
    m: usize,
    n: usize,
    p: usize,

    // Left and right hand sides for solves
    x: Vec<T>,
    b: Vec<T>,

    // internal workspace for IR scheme
    // and static offsetting of KKT
    work1: Vec<T>,
    work2: Vec<T>,

    // KKT mapping from problem data to KKT
    map: LDLDataMap,

    // the expected signs of D in KKT = LDL^T
    dsigns: Vec<i8>,

    // a vector for storing the entries of Hs blocks
    // on the KKT matrix block diagonal
    Hsblocks: Vec<T>,

    // unpermuted KKT matrix
    KKT: CscMatrix<T>,

    // full (untriangular) copy of KKT for parallel refinement residuals, with a map
    // from full nzval index -> triu nzval index for lazy value refresh
    KKTfull: CscMatrix<T>,
    fullmap: Vec<usize>,
    kktfull_stale: bool,

    // triangular storage shape for KKT
    KKTuplo: MatrixTriangle,

    // the direct linear LDL solver
    ldlsolver: BoxedDirectLDLSolver<T>,

    // the diagonal regularizer currently applied
    diagonal_regularizer: T,
}

impl<T> DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    pub fn new(
        P: &CscMatrix<T>,
        A: &CscMatrix<T>,
        cones: &CompositeCone<T>,
        m: usize,
        n: usize,
        settings: &CoreSettings<T>,
    ) -> Self {
        // get a constructor for the LDL solver we should use,
        // and also the matrix shape it requires
        let (kktshape, ldl_ctor) = T::get_ldlsolver_config(settings);

        //construct a KKT matrix of the right shape
        let (KKT, map) = assemble_kkt_matrix(P, A, cones, kktshape);

        //Need this many extra variables for sparse cones
        let p = map.sparse_maps.pdim();

        // LHS/RHS/work for iterative refinement
        let x = vec![T::zero(); n + m + p];
        let b = vec![T::zero(); n + m + p];
        let work1 = vec![T::zero(); n + m + p];
        let work2 = vec![T::zero(); n + m + p];

        // the expected signs of D in LDL
        let mut dsigns = vec![1_i8; n + m + p];
        _fill_signs(&mut dsigns, m, n, &map);

        // updates to the diagonal of KKT will be
        // assigned here before updating matrix entries
        let Hsblocks = allocate_kkt_Hsblocks::<T, T>(cones);

        let diagonal_regularizer = T::zero();

        // now make the LDL linear solver engine
        // final argument is None, since it is only
        // used by the Auto solver type to pass on
        // AMD ordering vectors to its selected solver
        // If using a solver directly and no ordering is
        // provided, the solver finds one for itself
        let ldlsolver = ldl_ctor(&KKT, &dsigns, settings, None);

        let (KKTfull, fullmap) = _build_full_from_triangle(&KKT, kktshape);

        Self {
            m,
            n,
            p,
            x,
            b,
            work1,
            work2,
            map,
            dsigns,
            Hsblocks,
            KKT,
            KKTfull,
            fullmap,
            kktfull_stale: true,
            KKTuplo: kktshape,
            ldlsolver,
            diagonal_regularizer,
        }
    }
}

impl<T> HasLinearSolverInfo for DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    fn linear_solver_info(&self) -> LinearSolverInfo {
        self.ldlsolver.linear_solver_info()
    }
}

impl<T> KKTSolver<T> for DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    fn update(&mut self, cones: &CompositeCone<T>, settings: &CoreSettings<T>) -> bool {
        let map = &self.map;

        // Set the elements the W^tW blocks in the KKT matrix.
        cones.get_Hs(&mut self.Hsblocks);

        let (values, index) = (&mut self.Hsblocks, &map.Hsblocks);
        // change signs to get -W^TW
        values.negate();
        _update_values(&mut self.ldlsolver, &mut self.KKT, index, values);

        let mut sparse_map_iter = map.sparse_maps.iter();
        let ldl = &mut self.ldlsolver;
        let KKT = &mut self.KKT;

        for cone in cones.iter() {
            if cone.is_sparse_expandable() {
                let sc = cone.to_sparse_expansion().unwrap();
                let thismap = sparse_map_iter.next().unwrap();
                sc.csc_update_sparsecone(thismap, ldl, KKT, _update_values, _scale_values);
            }
        }

        self.kktfull_stale = true;
        self.regularize_and_refactor(settings)
    }

    fn setrhs(&mut self, rhsx: &[T], rhsz: &[T]) {
        let (m, n, p) = (self.m, self.n, self.p);

        self.b[0..n].copy_from(rhsx);
        self.b[n..(n + m)].copy_from(rhsz);
        self.b[n + m..(n + m + p)].fill(T::zero());
    }

    fn solve(
        &mut self,
        lhsx: Option<&mut [T]>,
        lhsz: Option<&mut [T]>,
        settings: &CoreSettings<T>,
    ) -> bool {
        self.ldlsolver.solve(&self.KKT, &mut self.x, &mut self.b);

        let is_success = {
            if settings.iterative_refinement_enable {
                self.iterative_refinement(settings)
            } else {
                self.x.is_finite()
            }
        };

        if is_success {
            self.getlhs(lhsx, lhsz);
        }

        is_success
    }

    fn update_P(&mut self, P: &CscMatrix<T>) {
        _update_values(&mut self.ldlsolver, &mut self.KKT, &self.map.P, &P.nzval);
        self.kktfull_stale = true;
    }

    fn update_A(&mut self, A: &CscMatrix<T>) {
        _update_values(&mut self.ldlsolver, &mut self.KKT, &self.map.A, &A.nzval);
        self.kktfull_stale = true;
    }
}

impl<T> DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    // extra helper functions, not required for KKTSolver trait
    fn getlhs(&self, lhsx: Option<&mut [T]>, lhsz: Option<&mut [T]>) {
        let x = &self.x;
        let (m, n) = (self.m, self.n);

        if let Some(v) = lhsx {
            v.copy_from(&x[0..n]);
        }
        if let Some(v) = lhsz {
            v.copy_from(&x[n..(n + m)]);
        }
    }

    fn regularize_and_refactor(&mut self, settings: &CoreSettings<T>) -> bool {
        let map = &self.map;
        let KKT = &mut self.KKT;
        let dsigns = &self.dsigns;
        let diag_kkt = &mut self.work1;
        let diag_shifted = &mut self.work2;

        if settings.static_regularization_enable {
            // hold a copy of the true KKT diagonal
            // diag_kkt .= KKT.nzval[map.diag_full];
            for (d, idx) in zip(&mut *diag_kkt, &map.diag_full) {
                *d = KKT.nzval[*idx];
            }

            let eps = _compute_regularizer(diag_kkt, settings);

            // compute an offset version, accounting for signs
            diag_shifted.copy_from(diag_kkt);

            zip(&mut *diag_shifted, dsigns).for_each(|(shift, &sign)| {
                if sign == 1 {
                    *shift += eps;
                } else {
                    *shift -= eps;
                }
            });

            // overwrite the diagonal of KKT and within the ldlsolver
            _update_values(&mut self.ldlsolver, KKT, &map.diag_full, diag_shifted);

            // remember the value we used.  Not needed,
            // but possibly useful for debugging
            self.diagonal_regularizer = eps;
        }

        //refactor with new data
        let is_success = self.ldlsolver.refactor(KKT);

        if settings.static_regularization_enable {
            // put our internal copy of the KKT matrix back the way
            // it was. Not necessary to fix the ldlsolver copy because
            // this is only needed for our post-factorization IR scheme

            _update_values_KKT(KKT, &map.diag_full, diag_kkt);
        }

        is_success
    }

    fn iterative_refinement(&mut self, settings: &CoreSettings<T>) -> bool {
        if self.kktfull_stale {
            _refresh_full_values(&mut self.KKTfull, &self.fullmap, &self.KKT);
            self.kktfull_stale = false;
        }

        let (x, b) = (&mut self.x, &self.b);
        let (e, dx) = (&mut self.work1, &mut self.work2);

        // iterative refinement params
        let reltol = settings.iterative_refinement_reltol;
        let abstol = settings.iterative_refinement_abstol;
        let maxiter = settings.iterative_refinement_max_iter;
        let stopratio = settings.iterative_refinement_stop_ratio;

        let KKT = &self.KKT;
        let KKTfull = &self.KKTfull;

        let normb = b.norm_inf();

        //compute the initial error
        let mut norme = _get_refine_error(e, b, KKTfull, x);

        if !norme.is_finite() {
            return false;
        }

        for _ in 0..maxiter {
            if norme <= (abstol + reltol * normb) {
                //within tolerance.  Exit
                break;
            }

            let lastnorme = norme;

            //make a refinement
            self.ldlsolver.solve(KKT, dx, e);

            //prospective solution is x + dx.  Use dx space to
            // hold it for a check before applying to x
            dx.axpby(T::one(), x, T::one());

            norme = _get_refine_error(e, b, KKTfull, dx);

            if !norme.is_finite() {
                return false;
            }

            let improved_ratio = lastnorme / norme;
            if improved_ratio < stopratio {
                //insufficient improvement.  Exit
                if improved_ratio > T::one() {
                    std::mem::swap(x, dx);
                }
                break;
            }
            std::mem::swap(x, dx);
        }
        //NB: "success" means only that we had a finite valued result
        true
    }
}

fn _compute_regularizer<T: FloatT>(diag_kkt: &[T], settings: &CoreSettings<T>) -> T {
    let maxdiag = diag_kkt.norm_inf();

    // Compute a new regularizer
    settings.static_regularization_constant + settings.static_regularization_proportional * maxdiag
}

//  computes e = b - Kξ, overwriting the first argument
//  and returning its norm

fn _get_refine_error<T: FloatT>(
    e: &mut [T],
    b: &[T],
    KKTfull: &CscMatrix<T>,
    ξ: &mut [T],
) -> T {
    // KKTfull holds the full (not triangular) symmetric matrix, so each column j of the CSC
    // is also row j: the residual is a conflict-free per-row gather, parallelized over rows.
    use rayon::prelude::*;

    let colptr = &KKTfull.colptr;
    let rowval = &KKTfull.rowval;
    let nzval = &KKTfull.nzval;
    let xi: &[T] = ξ;

    e.par_iter_mut().enumerate().for_each(|(j, ej)| {
        let mut s = T::zero();
        for idx in colptr[j]..colptr[j + 1] {
            s += nzval[idx] * xi[rowval[idx]];
        }
        *ej = b[j] - s;
    });

    e.norm_inf()
}

// Build the full symmetric CSC matrix from triangular data, together with a map from each
// full-matrix nzval index back to the source triangular nzval index (for value refresh).
fn _build_full_from_triangle<T: FloatT>(
    K: &CscMatrix<T>,
    _uplo: MatrixTriangle,
) -> (CscMatrix<T>, Vec<usize>) {
    let n = K.n;
    let mut counts = vec![0usize; n];
    for col in 0..n {
        for idx in K.colptr[col]..K.colptr[col + 1] {
            let row = K.rowval[idx];
            counts[col] += 1;
            if row != col {
                counts[row] += 1;
            }
        }
    }

    let mut colptr = vec![0usize; n + 1];
    for i in 0..n {
        colptr[i + 1] = colptr[i] + counts[i];
    }
    let nnz = colptr[n];

    let mut rowval = vec![0usize; nnz];
    let mut fullmap = vec![0usize; nnz];
    let mut pos = colptr.clone();
    for col in 0..n {
        for idx in K.colptr[col]..K.colptr[col + 1] {
            let row = K.rowval[idx];
            rowval[pos[col]] = row;
            fullmap[pos[col]] = idx;
            pos[col] += 1;
            if row != col {
                rowval[pos[row]] = col;
                fullmap[pos[row]] = idx;
                pos[row] += 1;
            }
        }
    }

    let full = CscMatrix::new(n, n, colptr, rowval, vec![T::zero(); nnz]);
    (full, fullmap)
}

fn _refresh_full_values<T: FloatT>(KKTfull: &mut CscMatrix<T>, fullmap: &[usize], KKT: &CscMatrix<T>) {
    use rayon::prelude::*;

    let src = &KKT.nzval;
    KKTfull
        .nzval
        .par_iter_mut()
        .zip(fullmap.par_iter())
        .for_each(|(v, &i)| *v = src[i]);
}

// update entries of the KKT matrix using the given index into its CSC representation.
// applied to both the unpermuted matrix of the kktsolver and also to the ldlsolver
fn _update_values<T: FloatT>(
    ldlsolver: &mut BoxedDirectLDLSolver<T>,
    KKT: &mut CscMatrix<T>,
    index: &[usize],
    values: &[T],
) {
    //Update values in the KKT matrix K
    _update_values_KKT(KKT, index, values);

    // give the LDL subsolver an opportunity to update the same
    // values if needed.   This latter is useful for QDLDL since
    // it stores its own permuted copy internally
    ldlsolver.update_values(index, values);
}

fn _update_values_KKT<T: FloatT>(KKT: &mut CscMatrix<T>, index: &[usize], values: &[T]) {
    for (idx, v) in zip(index, values) {
        KKT.nzval[*idx] = *v;
    }
}

fn _scale_values<T: FloatT>(
    ldlsolver: &mut BoxedDirectLDLSolver<T>,
    KKT: &mut CscMatrix<T>,
    index: &[usize],
    scale: T,
) {
    //Update values in the KKT matrix K
    _scale_values_KKT(KKT, index, scale);

    // ...and in the LDL subsolver if needed
    ldlsolver.scale_values(index, scale);
}

//scales KKT matrix values
fn _scale_values_KKT<T: FloatT>(KKT: &mut CscMatrix<T>, index: &[usize], scale: T) {
    for idx in index.iter() {
        KKT.nzval[*idx] *= scale;
    }
}

fn _fill_signs(signs: &mut [i8], m: usize, n: usize, map: &LDLDataMap) {
    signs.fill(1);

    //flip expected negative signs of D in LDL
    signs[n..(n + m)].iter_mut().for_each(|x| *x = -*x);

    let mut p = m + n;
    // assign D signs for sparse expansion cones
    for thismap in map.sparse_maps.iter() {
        let thisp = thismap.pdim();
        signs[p..(p + thisp)].copy_from_slice(thismap.Dsigns());
        p += thisp;
    }
}
