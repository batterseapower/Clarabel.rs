#![allow(non_snake_case)]
use crate::algebra::*;
use core::cmp::{max, min};
use derive_builder::Builder;
use std::iter::zip;
use thiserror::Error;

/// Error codes returnable from [`QDLDLFactorisation`](QDLDLFactorisation) factor operations
#[derive(Error, Debug)]
pub enum QDLDLError {
    #[error("Matrix dimension fields are incompatible")]
    /// Matrix dimension fields are incompatible
    IncompatibleDimension,
    #[error("Matrix has a zero column")]
    /// Matrix has a zero column
    EmptyColumn,
    #[error("Matrix is not upper triangular")]
    /// Matrix is not upper triangular
    NotUpperTriangular,
    /// Matrix factorization produced a zero pivot
    #[error("Matrix factorization produced a zero pivot")]
    ZeroPivot,
    #[error("Invalid permutation vector")]
    /// Invalid permutation vector supplied
    InvalidPermutation,
}

#[derive(Builder, Debug, Clone)]
#[allow(missing_docs)]
/// Required settings for [`QDLDLFactorisation`](QDLDLFactorisation)
pub struct QDLDLSettings<T: FloatT> {
    /// "dense scale" parameter for AMD ordering
    #[builder(default = "1.0")]
    pub amd_dense_scale: f64,

    /// optional user-supplied custom permutation vector for the matrix
    #[builder(default = "None", setter(strip_option))]
    pub perm: Option<Vec<usize>>,

    /// Logical factorisation only, no numerical factorisation
    #[builder(default = "false")]
    pub logical: bool,

    /// optional user-supplied signs of the diagonal elements of D in LDL^T
    #[builder(default = "None", setter(strip_option))]
    pub Dsigns: Option<Vec<i8>>,

    /// Enable regularization during factorisation
    #[builder(default = "true")]
    pub regularize_enable: bool,

    /// Regularization epsilon parameter
    #[builder(default = "(1e-12).as_T()")]
    pub regularize_eps: T,

    /// Regularization delta parameter
    #[builder(default = "(1e-7).as_T()")]
    pub regularize_delta: T,
}

impl<T> Default for QDLDLSettings<T>
where
    T: FloatT,
{
    fn default() -> QDLDLSettings<T> {
        QDLDLSettingsBuilder::<T>::default().build().unwrap()
    }
}

/// Performs $LDL^T$ factorization of a symmetric quasidefinite matrix
#[derive(Debug)]
pub struct QDLDLFactorisation<T = f64> {
    /// permutation vector
    pub perm: Vec<usize>,
    /// inverse permutation
    #[allow(dead_code)] //Unused because we call ipermute in solve instead.  Keep anyway.
    iperm: Vec<usize>,
    /// lower triangular factor L in LDL^T
    pub L: CscMatrix<T>,
    /// vector of diagonal elements of D in LDL^T
    pub D: Vec<T>,
    /// vector of reciprocal diagonal elements of D in LDL^T
    pub Dinv: Vec<T>,
    /// internal workspace data
    workspace: QDLDLWorkspace<T>,
    /// true if factorisation is symbolic only
    is_symbolic: bool,
}

impl<T> QDLDLFactorisation<T>
where
    T: FloatT,
{
    /// create a new LDL^T factorisation
    pub fn new(
        Ain: &CscMatrix<T>,
        opts: Option<QDLDLSettings<T>>,
    ) -> Result<QDLDLFactorisation<T>, QDLDLError> {
        //sanity check on structure
        check_structure(Ain)?;
        _qdldl_new(Ain, opts)
    }

    /// returns the number of positive eigenvalues
    pub fn positive_inertia(&self) -> usize {
        self.workspace.positive_inertia
    }
    /// returns the number of regularisation shifts
    /// that were applied during factorisation
    pub fn regularize_count(&self) -> usize {
        self.workspace.regularize_count
    }

    /// Solves Ax = b using LDL factors for A.
    /// Solves in place (x replaces b)
    pub fn solve(&mut self, b: &mut [T]) {
        // bomb if logical factorisation only
        assert!(!self.is_symbolic);

        // bomb if b is the wrong size
        assert_eq!(b.len(), self.D.len());

        // permute b
        let tmp = &mut self.workspace.fwork;
        permute(tmp, b, &self.perm);

        //solve in place with tmp as permuted RHS
        _solve(
            &self.L.colptr,
            &self.L.rowval,
            &self.L.nzval,
            &self.Dinv,
            tmp,
        );

        // inverse permutation to put unpermuted soln in b
        ipermute(b, tmp, &self.perm);
    }

    /// Update a subset of the values of the matrix to be (re)factored.  See [`refactor`](crate::qdldl::QDLDLFactorisation::refactor)
    ///
    pub fn update_values(&mut self, indices: &[usize], values: &[T]) {
        let nzval = &mut self.workspace.triuA.nzval; // post perm internal data
        let AtoPAPt = &self.workspace.AtoPAPt; //mapping from input matrix entries to triuA

        for (i, &idx) in indices.iter().enumerate() {
            nzval[AtoPAPt[idx]] = values[i];
        }
    }

    /// Update a subset of the values of the matrix to be (re)factored.  See [`refactor`](crate::qdldl::QDLDLFactorisation::refactor)
    ///
    pub fn scale_values(&mut self, indices: &[usize], scale: T) {
        let nzval = &mut self.workspace.triuA.nzval; // post perm internal data
        let AtoPAPt = &self.workspace.AtoPAPt; //mapping from input matrix entries to triuA

        for &idx in indices.iter() {
            nzval[AtoPAPt[idx]] *= scale;
        }
    }

    /// Shifts a subset of the values of the matrix to be (re)factored.   The
    /// values are offset by `offset`, with `signs` a vector of +/- 1 values
    /// indicating the direction of shifts. See [`refactor`](crate::qdldl::QDLDLFactorisation::refactor)
    ///
    pub fn offset_values(&mut self, indices: &[usize], offset: T, signs: &[i8]) {
        assert_eq!(indices.len(), signs.len());

        let nzval = &mut self.workspace.triuA.nzval; // post perm internal data
        let AtoPAPt = &self.workspace.AtoPAPt; //mapping from input matrix entries to triuA

        for (&idx, &sign) in zip(indices, signs) {
            match sign.signum() {
                1 => {
                    nzval[AtoPAPt[idx]] += offset;
                }
                -1 => {
                    nzval[AtoPAPt[idx]] -= offset;
                }
                _ => {}
            }
        }
    }

    /// Refactor a matrix after its data has been modified.   See [`update_values`](crate::qdldl::QDLDLFactorisation::update_values),
    /// [`scale_values`](crate::qdldl::QDLDLFactorisation::scale_values) and [`offset_values`](crate::qdldl::QDLDLFactorisation::offset_values)
    ///
    pub fn refactor(&mut self) -> Result<(), QDLDLError> {
        // It never makes sense to call refactor for a logical
        // factorization since it will always be the same.  Calling
        // this function implies that we want a numerical factorization
        self.is_symbolic = false;
        _factor(
            &mut self.L,
            &mut self.D,
            &mut self.Dinv,
            &mut self.workspace,
            self.is_symbolic,
        )
    }

    /// Returns the number of nonzeros in A for A = LDL^T
    pub fn nnzA(&self) -> usize {
        self.workspace.triuA.nnz()
    }

    /// Returns the number of nonzeros in L for A = LDL^T
    pub fn nnzL(&self) -> usize {
        self.L.nnz()
    }
}

fn check_structure<T: FloatT>(A: &CscMatrix<T>) -> Result<(), QDLDLError> {
    if !A.is_square() {
        return Err(QDLDLError::IncompatibleDimension);
    }

    if !A.is_triu() {
        return Err(QDLDLError::NotUpperTriangular);
    }

    //Error if A doesn't have at least one entry in every column
    if !A.colptr.windows(2).all(|c| c[0] < c[1]) {
        return Err(QDLDLError::EmptyColumn);
    }

    Ok(())
}

fn _qdldl_new<T: FloatT>(
    Ain: &CscMatrix<T>,
    opts: Option<QDLDLSettings<T>>,
) -> Result<QDLDLFactorisation<T>, QDLDLError> {
    let n = Ain.nrows();

    //get default values if no options passed at all
    let opts = opts.unwrap_or_default();

    //Use AMD ordering if a user-provided ordering
    //is not supplied.   For no ordering at all, the
    //user would need to pass (0..n).collect() explicitly
    let (perm, iperm);
    if let Some(_perm) = opts.perm {
        iperm = _invperm(&_perm)?;
        perm = _perm;
    } else {
        (perm, iperm, _) = get_amd_ordering(Ain, opts.amd_dense_scale);
    }

    //permute to (another) upper triangular matrix and store the
    //index mapping the input's entries to the permutation's entries
    let (A, AtoPAPt) = permute_symmetric(Ain, &iperm);

    // handle the (possibly permuted) vector of
    // diagonal D signs if one was specified.  Otherwise
    // otherwise all signs are positive
    let mut Dsigns = vec![1_i8; n];
    if let Some(ds) = opts.Dsigns {
        Dsigns = vec![1_i8; n];
        permute(&mut Dsigns, &ds, &perm);
    }

    // allocate workspace
    let mut workspace = QDLDLWorkspace::<T>::new(
        A,
        AtoPAPt,
        Dsigns,
        opts.regularize_enable,
        opts.regularize_eps,
        opts.regularize_delta,
    )?;

    //total nonzeros in factorization
    let sumLnz = workspace.Lnz.iter().sum();

    // allocate space for the L matrix row indices and data
    let mut L = CscMatrix::spalloc((n, n), sumLnz);

    // allocate for D and D inverse in LDL^T
    let mut D = vec![T::zero(); n];
    let mut Dinv = vec![T::zero(); n];

    // factor the matrix into A = LDL^T
    _factor(&mut L, &mut D, &mut Dinv, &mut workspace, opts.logical)?;

    Ok(QDLDLFactorisation {
        perm,
        iperm,
        L,
        D,
        Dinv,
        workspace,
        is_symbolic: opts.logical,
    })
}

#[derive(Debug)]
struct QDLDLWorkspace<T> {
    // internal workspace data
    etree: Vec<usize>,
    Lnz: Vec<usize>,
    iwork: Vec<usize>,
    bwork: Vec<bool>,
    fwork: Vec<T>,

    // number of positive values in D
    positive_inertia: usize,

    // The upper triangular matrix factorisation target
    // This is the post ordering PAPt of the original data
    triuA: CscMatrix<T>,

    // mapping from entries in the triu form
    // of the original input to the post ordering
    // triu form used for the factorization
    // this can be used when modifying entries
    // of the data matrix for refactoring
    AtoPAPt: Vec<usize>,

    //regularization signs and parameters
    Dsigns: Vec<i8>,
    regularize_enable: bool,
    regularize_eps: T,
    regularize_delta: T,

    // number of regularized entries in D
    regularize_count: usize,

    // wavefront schedule for the parallel numeric factorization: nodes 1..n
    // in ascending elimination-tree height order, with per-height offsets
    wf_order: Vec<usize>,
    wf_offsets: Vec<usize>,

    // lazily allocated per-thread scratch for the parallel factorization
    par_scratch: Vec<ParScratch<T>>,

    // static-structure replay cache for numeric refactorization: after the
    // first numeric factorization the sparsity of L is fixed, so refactors
    // can replay each row's elimination sequence directly. pat_ptr indexes
    // rows 1..n into (pat_cols, pat_slots): the column eliminated against and
    // the L slot the row's entry lands in, in the exact processing order of
    // the discovery-based code. li32 is a u32 copy of L.rowval (halving the
    // index traffic of the dominant inner loop).
    pat_ptr: Vec<usize>,
    pat_cols: Vec<u32>,
    pat_slots: Vec<u32>,
    li32: Vec<u32>,
    pat_ready: bool,
}

// per-thread scratch mirroring the y_markers/y_idx/elim_buffer/y_vals split
// of the serial factorization workspace
#[derive(Debug)]
struct ParScratch<T> {
    y_markers: Vec<bool>,
    y_idx: Vec<usize>,
    elim_buffer: Vec<usize>,
    y_vals: Vec<T>,
}

impl<T: FloatT> ParScratch<T> {
    fn new(n: usize) -> Self {
        Self {
            y_markers: vec![QDLDL_UNUSED; n],
            y_idx: vec![0; n],
            elim_buffer: vec![0; n],
            y_vals: vec![T::zero(); n],
        }
    }
}

// raw-pointer wrapper for the shared factorization outputs; safety of the
// shared mutation is argued at the use site in _factor_inner_parallel
#[derive(Clone, Copy)]
struct SendPtr<P>(P);
unsafe impl<P> Send for SendPtr<P> {}
unsafe impl<P> Sync for SendPtr<P> {}

impl<T> QDLDLWorkspace<T>
where
    T: FloatT,
{
    pub fn new(
        triuA: CscMatrix<T>,
        AtoPAPt: Vec<usize>,
        Dsigns: Vec<i8>,
        regularize_enable: bool,
        regularize_eps: T,
        regularize_delta: T,
    ) -> Result<Self, QDLDLError> {
        let mut etree = vec![0; triuA.ncols()];
        let mut Lnz = vec![0; triuA.ncols()]; //nonzeros in each L column
        let mut iwork = vec![0; triuA.ncols() * 3];
        let bwork = vec![false; triuA.ncols()];
        let fwork = vec![T::zero(); triuA.ncols()];

        // compute elimination tree using QDLDL converted code
        _etree(
            triuA.nrows(),
            &triuA.colptr,
            &triuA.rowval,
            &mut iwork,
            &mut Lnz,
            &mut etree,
        )?;

        // wavefront schedule: group nodes by elimination-tree height. Parents
        // always have larger indices than their children, so a single ascending
        // pass computes heights.
        let n = triuA.ncols();
        let mut height = vec![0usize; n];
        for k in 0..n {
            let p = etree[k];
            if p != QDLDL_UNKNOWN {
                height[p] = height[p].max(height[k] + 1);
            }
        }
        let maxh = height.iter().copied().max().unwrap_or(0);
        // counting sort of nodes 1..n by height (node 0 is handled separately
        // by the factorization prologue)
        let mut wf_offsets = vec![0usize; maxh + 2];
        for k in 1..n {
            wf_offsets[height[k] + 1] += 1;
        }
        for h in 0..(maxh + 1) {
            wf_offsets[h + 1] += wf_offsets[h];
        }
        let mut wf_order = vec![0usize; n.saturating_sub(1)];
        let mut pos = wf_offsets.clone();
        for k in 1..n {
            let h = height[k];
            wf_order[pos[h]] = k;
            pos[h] += 1;
        }

        // positive inertia count.
        let positive_inertia = 0;

        // number of regularized entries in D. None to start
        let regularize_count = 0;

        Ok(Self {
            etree,
            Lnz,
            iwork,
            bwork,
            fwork,
            positive_inertia,
            triuA,
            AtoPAPt,
            Dsigns,
            regularize_enable,
            regularize_eps,
            regularize_delta,
            regularize_count,
            wf_order,
            wf_offsets,
            par_scratch: Vec::new(),
            pat_ptr: Vec::new(),
            pat_cols: Vec::new(),
            pat_slots: Vec::new(),
            li32: Vec::new(),
            pat_ready: false,
        })
    }
}

fn _factor<T: FloatT>(
    L: &mut CscMatrix<T>,
    D: &mut [T],
    Dinv: &mut [T],
    workspace: &mut QDLDLWorkspace<T>,
    logical: bool,
) -> Result<(), QDLDLError> {
    if logical {
        L.nzval.fill(T::one());
        D.fill(T::one());
        Dinv.fill(T::one());
    }

    // Use the wavefront-parallel numeric factorization for large problems.
    // The serial path is kept bit-identical for small problems and for
    // logical (symbolic-only) factorization.
    const PARALLEL_MIN_DIM: usize = 20_000;
    if !logical && workspace.triuA.ncols() >= PARALLEL_MIN_DIM && rayon::current_num_threads() > 1 {
        let pos_d_count = _factor_inner_parallel(L, D, Dinv, workspace)?;
        workspace.positive_inertia = pos_d_count;
        return Ok(());
    }

    // factor using QDLDL C style converted code
    let A = &workspace.triuA;

    let pos_d_count = _factor_inner(
        A.n,
        &A.colptr,
        &A.rowval,
        &A.nzval,
        &mut L.colptr,
        &mut L.rowval,
        &mut L.nzval,
        D,
        Dinv,
        &workspace.Lnz,
        &workspace.etree,
        &mut workspace.bwork,
        &mut workspace.iwork,
        &mut workspace.fwork,
        logical,
        &workspace.Dsigns,
        workspace.regularize_enable,
        workspace.regularize_eps,
        workspace.regularize_delta,
        &mut workspace.regularize_count,
    )?;

    workspace.positive_inertia = pos_d_count;

    Ok(())
}

const QDLDL_UNKNOWN: usize = usize::MAX;
const QDLDL_USED: bool = true;
const QDLDL_UNUSED: bool = false;

// Compute the elimination tree for a quasidefinite matrix
// in compressed sparse column form.

fn _etree(
    n: usize,
    Ap: &[usize],
    Ai: &[usize],
    work: &mut [usize],
    Lnz: &mut [usize],
    etree: &mut [usize],
) -> Result<usize, QDLDLError> {
    // zero out Lnz and work.  Set all etree values to unknown
    work.fill(0);
    Lnz.fill(0);
    etree.fill(QDLDL_UNKNOWN);

    // compute the elimination tree
    for j in 0..n {
        work[j] = j;
        for istart in Ai.iter().take(Ap[j + 1]).skip(Ap[j]) {
            let mut i = *istart;

            while work[i] != j {
                if etree[i] == QDLDL_UNKNOWN {
                    etree[i] = j;
                }
                Lnz[i] += 1; // nonzeros in this column
                work[i] = j;
                i = etree[i];
            }
        }
    }

    Ok(0)
}

//allow too_many_arguments since this follows the implementation
//of the C version of QDLDL.
#[allow(clippy::too_many_arguments)]
fn _factor_inner<T: FloatT>(
    n: usize,
    Ap: &[usize],
    Ai: &[usize],
    Ax: &[T],
    Lp: &mut [usize],
    Li: &mut [usize],
    Lx: &mut [T],
    D: &mut [T],
    Dinv: &mut [T],
    Lnz: &[usize],
    etree: &[usize],
    bwork: &mut [bool],
    iwork: &mut [usize],
    fwork: &mut [T],
    logical_factor: bool,
    Dsigns: &[i8],
    regularize_enable: bool,
    regularize_eps: T,
    regularize_delta: T,
    regularize_count: &mut usize,
) -> Result<usize, QDLDLError> {
    *regularize_count = 0;
    let mut positiveValuesInD = 0;

    // partition working memory into pieces
    let y_markers = bwork;
    let (y_idx, iwork) = iwork.split_at_mut(n);
    let (elim_buffer, next_colspace) = iwork.split_at_mut(n);
    let y_vals = fwork;

    //set Lp to cumsum(Lnz), starting from zero
    Lp[0] = 0;
    let mut acc = 0;
    for (Lp, Lnz) in zip(&mut Lp[1..], Lnz) {
        acc += Lnz;
        *Lp = acc;
    }

    //  set all y_idx to be 'unused' initially
    // in each column of L, the next available space
    // to start is just the first space in the column
    y_markers.fill(QDLDL_UNUSED);
    y_vals.fill(T::zero());
    D.fill(T::zero());
    next_colspace.copy_from_slice(&Lp[0..Lp.len() - 1]);

    if !logical_factor {
        // First element of the diagonal D.
        D[0] = Ax[0];
        if regularize_enable {
            let sign = T::from_i8(Dsigns[0]).unwrap();
            if D[0] * sign < regularize_eps {
                D[0] = regularize_delta * sign;
                *regularize_count += 1;
            }
        }

        if D[0].is_zero() {
            return Err(QDLDLError::ZeroPivot);
        }
        if D[0] > T::zero() {
            positiveValuesInD += 1;
        }
        Dinv[0] = T::recip(D[0]);
    }

    // Start from second row (k=1) here. The upper LH corner is trivially 0
    // in L b/c we are only computing the subdiagonal elements
    for k in 1..n {
        // NB : For each k, we compute a solution to
        // y = L(0:(k-1),0:k-1))\b, where b is the kth
        // column of A that sits above the diagonal.
        // The solution y is then the kth row of L,
        // with an implied '1' at the diagonal entry.

        // number of nonzeros in this row of L
        let mut nnz_y = 0; // number of elements in this row

        // This loop determines where nonzeros
        // will go in the kth row of L, but doesn't
        // compute the actual values

        for i in Ap[k]..Ap[k + 1] {
            let bidx = Ai[i]; //we are working on this element of b

            // Initialize D[k] as the element of this column
            // corresponding to the diagonal place.  Don't use
            // this element as part of the elimination step
            // that computes the k^th row of L
            if bidx == k {
                D[k] = Ax[i];
                continue;
            }

            y_vals[bidx] = Ax[i]; // initialise y(bidx) = b(bidx)

            // use the forward elimination tree to figure
            // out which elements must be eliminated after
            // this element of b
            let next_idx = bidx;

            if y_markers[next_idx] == QDLDL_UNUSED {
                //this y term not already visited

                y_markers[next_idx] = QDLDL_USED; //I touched this one
                elim_buffer[0] = next_idx; // It goes at the start of the current list
                let mut nnz_e = 1; //length of unvisited elimination path from here

                let mut next_idx = etree[bidx];

                while next_idx != QDLDL_UNKNOWN && next_idx < k {
                    if y_markers[next_idx] == QDLDL_USED {
                        break;
                    }

                    y_markers[next_idx] = QDLDL_USED; // I touched this one
                    elim_buffer[nnz_e] = next_idx; // It goes in the current list
                    next_idx = etree[next_idx]; // one step further along tree
                    nnz_e += 1; // the list is one longer than before
                }

                // now put the buffered elimination list into
                // my current ordering in reverse order
                while nnz_e != 0 {
                    nnz_e -= 1;
                    y_idx[nnz_y] = elim_buffer[nnz_e];
                    nnz_y += 1;
                }
            }
        }

        // This for loop places nonzeros values in the k^th row
        for i in (0..nnz_y).rev() {
            // which column are we working on?
            let cidx = y_idx[i];

            // loop along the elements in this
            // column of L and subtract to solve to y
            let tmp_idx = next_colspace[cidx];

            // don't compute Lx for logical factorisation
            // this logic is not implemented in the C version
            if !logical_factor {
                let y_vals_cidx = y_vals[cidx];

                let (f, l) = (Lp[cidx], tmp_idx);
                unsafe {
                    //Safety : Here the Lij index comes from the rowval
                    //field of the sparse L factor matrix, and should
                    //always be bounded by the matrix dimension.
                    for (&Lxj, &Lij) in zip(&Lx[f..l], &Li[f..l]) {
                        *(y_vals.get_unchecked_mut(Lij)) -= Lxj * y_vals_cidx;
                    }

                    // Now I have the cidx^th element of y = L\b.
                    // so compute the corresponding element of
                    // this row of L and put it into the right place
                    let Lx_tmp_idx = y_vals_cidx * *Dinv.get_unchecked(cidx);
                    *Lx.get_unchecked_mut(tmp_idx) = Lx_tmp_idx;
                    *D.get_unchecked_mut(k) -= y_vals_cidx * Lx_tmp_idx;
                }
            }

            // record which row it went into
            Li[tmp_idx] = k;
            next_colspace[cidx] += 1;

            // reset the y_vals and indices back to zero and QDLDL_UNUSED
            // once I'm done with them
            y_vals[cidx] = T::zero();
            y_markers[cidx] = QDLDL_UNUSED;
        }

        if !logical_factor {
            // apply dynamic regularization
            if regularize_enable {
                let sign = T::from_i8(Dsigns[k]).unwrap();
                if D[k] * sign < regularize_eps {
                    D[k] = regularize_delta * sign;
                    *regularize_count += 1;
                }
            }

            // Maintain a count of the positive entries
            // in D.  If we hit a zero, we can't factor
            // this matrix, so abort
            if D[k].is_zero() {
                return Err(QDLDLError::ZeroPivot);
            }
            if D[k] > T::zero() {
                positiveValuesInD += 1;
            }

            // compute the inverse of the diagonal
            Dinv[k] = T::recip(D[k]);
        }
    } //end for k

    Ok(positiveValuesInD)
}

// Wavefront-parallel variant of _factor_inner.
//
// Rows are processed in ascending elimination-tree height order; rows of equal
// height are processed in parallel. Safety of the shared mutation relies on
// standard elimination-tree structure properties:
//
//   * the sparsity pattern of row k of L (the columns its triangular solve
//     reads, and the columns it appends one entry to) consists only of
//     DESCENDANTS of k in the elimination tree;
//   * every entry of column j of L is written by a row that is an ANCESTOR of
//     j; ancestors of j form a chain with strictly increasing heights (and
//     strictly increasing row indices), so within one wavefront at most one
//     row touches any given column, appends across wavefronts occur in
//     ascending row order (preserving CSC ordering), and every column read by
//     a row was completed at strictly lower heights;
//   * two rows of equal height have disjoint descendant sets (a common
//     descendant would make one an ancestor of the other, contradicting equal
//     heights).
//
// D[k]/Dinv[k] are written only by row k and read only for descendants
// (strictly lower heights). Each worker uses its own y workspace. Results are
// bitwise identical to the serial factorization: every entry is computed from
// the same inputs with the same inner loop order, independent of scheduling.
#[allow(non_snake_case)]
// Wavefront-parallel variant of _factor_inner with a static-structure replay cache.
//
// Rows are processed in ascending elimination-tree height order; rows of equal
// height are processed in parallel. Safety of the shared mutation relies on
// standard elimination-tree structure properties:
//
//   * the sparsity pattern of row k of L (the columns its triangular solve
//     reads, and the columns it appends one entry to) consists only of
//     DESCENDANTS of k in the elimination tree;
//   * every entry of column j of L is written by a row that is an ANCESTOR of
//     j; ancestors of j form a chain with strictly increasing heights (and
//     strictly increasing row indices), so within one wavefront at most one
//     row touches any given column, appends across wavefronts occur in
//     ascending row order (preserving CSC ordering), and every column read by
//     a row was completed at strictly lower heights;
//   * two rows of equal height have disjoint descendant sets (a common
//     descendant would make one an ancestor of the other, contradicting equal
//     heights).
//
// D[k]/Dinv[k] are written only by row k and read only for descendants
// (strictly lower heights). Each worker uses its own y workspace. Results are
// bitwise identical to the serial factorization: every entry is computed from
// the same inputs with the same inner loop order, independent of scheduling.
//
// The sparsity structure of L is fixed after the first numeric factorization,
// so that first call records, per row, the exact sequence of (column, L slot)
// eliminations the discovery code performs (plus a u32 copy of L.rowval).
// Subsequent refactorizations replay that sequence directly: no elimination
// tree walks, no marker bookkeeping, no index writes, and 32-bit index reads
// in the dominant inner loop.
#[allow(non_snake_case)]
fn _factor_inner_parallel<T: FloatT>(
    L: &mut CscMatrix<T>,
    D: &mut [T],
    Dinv: &mut [T],
    workspace: &mut QDLDLWorkspace<T>,
) -> Result<usize, QDLDLError> {
    use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    let A = &workspace.triuA;
    let n = A.n;
    let (Ap, Ai, Ax) = (&A.colptr, &A.rowval, &A.nzval);
    let Lnz = &workspace.Lnz;
    let etree = &workspace.etree;
    let Dsigns = &workspace.Dsigns;
    let regularize_enable = workspace.regularize_enable;
    let regularize_eps = workspace.regularize_eps;
    let regularize_delta = workspace.regularize_delta;

    let nthreads = rayon::current_num_threads();
    if workspace.par_scratch.len() < nthreads || workspace.par_scratch.first().is_some_and(|w| w.y_vals.len() != n) {
        workspace.par_scratch = (0..nthreads).map(|_| ParScratch::new(n)).collect();
    }

    // prologue, identical to _factor_inner: Lp = cumsum(Lnz) and the first
    // diagonal element
    let Lp = &mut L.colptr;
    Lp[0] = 0;
    let mut acc = 0;
    for (Lp, Lnz) in zip(&mut Lp[1..], Lnz) {
        acc += Lnz;
        *Lp = acc;
    }

    D.fill(T::zero());

    let mut regularize_count_total = 0usize;
    let mut positive_total = 0usize;

    D[0] = Ax[0];
    if regularize_enable {
        let sign = T::from_i8(Dsigns[0]).unwrap();
        if D[0] * sign < regularize_eps {
            D[0] = regularize_delta * sign;
            regularize_count_total += 1;
        }
    }
    if D[0].is_zero() {
        return Err(QDLDLError::ZeroPivot);
    }
    if D[0] > T::zero() {
        positive_total += 1;
    }
    Dinv[0] = T::recip(D[0]);

    if !workspace.pat_ready {
        // First numeric factorization: run the serial discovery-based
        // elimination, recording each row's processing sequence.
        assert!(L.nzval.len() <= u32::MAX as usize && n <= u32::MAX as usize);

        let (y_idx, iwork) = workspace.iwork.split_at_mut(n);
        let (elim_buffer, next_colspace) = iwork.split_at_mut(n);
        let y_markers = &mut workspace.bwork;
        let y_vals = &mut workspace.fwork;
        next_colspace.copy_from_slice(&Lp[0..Lp.len() - 1]);

        let pat_ptr = &mut workspace.pat_ptr;
        let pat_cols = &mut workspace.pat_cols;
        let pat_slots = &mut workspace.pat_slots;
        pat_ptr.clear();
        pat_cols.clear();
        pat_slots.clear();
        pat_ptr.reserve(n + 1);
        pat_ptr.push(0);
        pat_ptr.push(0); // row 0 is empty
        pat_cols.reserve(L.nzval.len());
        pat_slots.reserve(L.nzval.len());

        let (Li, Lx) = (&mut L.rowval, &mut L.nzval);

        for k in 1..n {
            // pattern discovery, identical to _factor_inner
            let mut nnz_y = 0usize;
            let mut dk = T::zero();

            for i in Ap[k]..Ap[k + 1] {
                let bidx = Ai[i];
                if bidx == k {
                    dk = Ax[i];
                    continue;
                }
                y_vals[bidx] = Ax[i];

                let next_idx = bidx;
                if y_markers[next_idx] == QDLDL_UNUSED {
                    y_markers[next_idx] = QDLDL_USED;
                    elim_buffer[0] = next_idx;
                    let mut nnz_e = 1;
                    let mut next_idx = etree[bidx];
                    while next_idx != QDLDL_UNKNOWN && next_idx < k {
                        if y_markers[next_idx] == QDLDL_USED {
                            break;
                        }
                        y_markers[next_idx] = QDLDL_USED;
                        elim_buffer[nnz_e] = next_idx;
                        next_idx = etree[next_idx];
                        nnz_e += 1;
                    }
                    while nnz_e != 0 {
                        nnz_e -= 1;
                        y_idx[nnz_y] = elim_buffer[nnz_e];
                        nnz_y += 1;
                    }
                }
            }

            for i in (0..nnz_y).rev() {
                let cidx = y_idx[i];
                let tmp_idx = next_colspace[cidx];
                pat_cols.push(cidx as u32);
                pat_slots.push(tmp_idx as u32);

                let y_vals_cidx = y_vals[cidx];
                let (f, l) = (Lp[cidx], tmp_idx);
                unsafe {
                    for (&Lxj, &Lij) in zip(&Lx[f..l], &Li[f..l]) {
                        *(y_vals.get_unchecked_mut(Lij)) -= Lxj * y_vals_cidx;
                    }
                    let lx_tmp = y_vals_cidx * *Dinv.get_unchecked(cidx);
                    *Lx.get_unchecked_mut(tmp_idx) = lx_tmp;
                    dk -= y_vals_cidx * lx_tmp;
                }

                Li[tmp_idx] = k;
                next_colspace[cidx] += 1;
                y_vals[cidx] = T::zero();
                y_markers[cidx] = QDLDL_UNUSED;
            }
            pat_ptr.push(pat_cols.len());

            if regularize_enable {
                let sign = T::from_i8(Dsigns[k]).unwrap();
                if dk * sign < regularize_eps {
                    dk = regularize_delta * sign;
                    regularize_count_total += 1;
                }
            }
            if dk.is_zero() {
                return Err(QDLDLError::ZeroPivot);
            }
            if dk > T::zero() {
                positive_total += 1;
            }
            D[k] = dk;
            Dinv[k] = T::recip(dk);
        }

        workspace.li32 = Li.iter().map(|&i| i as u32).collect();
        workspace.pat_ready = true;
        workspace.regularize_count = regularize_count_total;
        return Ok(positive_total);
    }

    // Replay path: the structure of L (rowval, colptr, per-row elimination
    // sequences) is unchanged; only values are recomputed.
    let pat_ptr = &workspace.pat_ptr;
    let pat_cols = &workspace.pat_cols;
    let pat_slots = &workspace.pat_slots;
    let li32 = &workspace.li32;

    let Lp_ptr = SendPtr(Lp.as_ptr());
    let Lx_ptr = SendPtr(L.nzval.as_mut_ptr());
    let D_ptr = SendPtr(D.as_mut_ptr());
    let Dinv_ptr = SendPtr(Dinv.as_mut_ptr());

    let zero_pivot = AtomicBool::new(false);
    let positive_count = AtomicUsize::new(0);
    let regularize_count_atomic = AtomicUsize::new(0);

    let factor_row = |k: usize, scratch: &mut ParScratch<T>| -> bool {
        let y_vals = &mut scratch.y_vals;
        let (Lp, Lx, D, Dinv) = (Lp_ptr, Lx_ptr, D_ptr, Dinv_ptr);

        let mut dk = T::zero();
        for i in Ap[k]..Ap[k + 1] {
            let bidx = Ai[i];
            if bidx == k {
                dk = Ax[i];
            } else {
                y_vals[bidx] = Ax[i];
            }
        }

        unsafe {
            // Safety: every column in this row's cached pattern is a
            // descendant of k, so no other row in this wavefront reads or
            // writes its L values or D entries; all values read were
            // completed at strictly lower wavefronts. The slot for this
            // row's entry in each column is fixed by the static structure.
            for idx in *pat_ptr.get_unchecked(k)..*pat_ptr.get_unchecked(k + 1) {
                let cidx = *pat_cols.get_unchecked(idx) as usize;
                let slot = *pat_slots.get_unchecked(idx) as usize;

                let y_vals_cidx = *y_vals.get_unchecked(cidx);
                let f = *Lp.0.add(cidx);
                const PF_DIST: usize = 24;
                for j in f..slot {
                    #[cfg(target_arch = "x86_64")]
                    if j + PF_DIST < slot {
                        let pf = *li32.get_unchecked(j + PF_DIST) as usize;
                        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                            y_vals.as_ptr().add(pf) as *const i8,
                        );
                    }
                    let lij = *li32.get_unchecked(j) as usize;
                    *(y_vals.get_unchecked_mut(lij)) -= *Lx.0.add(j) * y_vals_cidx;
                }

                let lx_tmp = y_vals_cidx * *Dinv.0.add(cidx);
                *Lx.0.add(slot) = lx_tmp;
                dk -= y_vals_cidx * lx_tmp;

                *y_vals.get_unchecked_mut(cidx) = T::zero();
            }
        }

        if regularize_enable {
            let sign = T::from_i8(Dsigns[k]).unwrap();
            if dk * sign < regularize_eps {
                dk = regularize_delta * sign;
                regularize_count_atomic.fetch_add(1, Ordering::Relaxed);
            }
        }

        if dk.is_zero() {
            zero_pivot.store(true, Ordering::Relaxed);
            return false;
        }
        if dk > T::zero() {
            positive_count.fetch_add(1, Ordering::Relaxed);
        }

        unsafe {
            *D.0.add(k) = dk;
            *Dinv.0.add(k) = T::recip(dk);
        }
        true
    };

    // process the wavefronts; small levels run on the current thread to avoid
    // scheduling overhead, larger ones fan out across a scope
    const PARALLEL_MIN_LEVEL: usize = 32;
    let wf_offsets = &workspace.wf_offsets;
    let wf_order = &workspace.wf_order;

    let mut scratches: Vec<&mut ParScratch<T>> = workspace.par_scratch.iter_mut().collect();

    if std::env::var("CLARABEL_DEBUG_WF").is_ok() {
        let nlv = wf_offsets.len() - 1;
        let mut wide_rows = 0usize;
        let mut wide_work = 0usize;
        let mut total_work = 0usize;
        for h in 0..nlv {
            let level = &wf_order[wf_offsets[h]..wf_offsets[h + 1]];
            let work: usize = level.iter().map(|&k| Lnz[k]).sum();
            total_work += work;
            if level.len() >= PARALLEL_MIN_LEVEL {
                wide_rows += level.len();
                wide_work += work;
            }
        }
        eprintln!(
            "CLARABEL_DEBUG_WF: n={} levels={} wide_rows={} ({:.1}%) wide_Lnz={} of {} ({:.1}%)",
            n, nlv, wide_rows,
            100.0 * wide_rows as f64 / (n - 1) as f64,
            wide_work, total_work,
            100.0 * wide_work as f64 / total_work.max(1) as f64,
        );
    }

    for h in 0..wf_offsets.len() - 1 {
        if zero_pivot.load(Ordering::Relaxed) {
            break;
        }
        let level = &wf_order[wf_offsets[h]..wf_offsets[h + 1]];
        if level.is_empty() {
            continue;
        }

        if level.len() < PARALLEL_MIN_LEVEL {
            let scratch = &mut scratches[0];
            for &k in level {
                if !factor_row(k, scratch) {
                    break;
                }
            }
        } else {
            let chunk = level.len().div_ceil(scratches.len());
            rayon::scope(|s| {
                for (rows, scratch) in zip(level.chunks(chunk), scratches.iter_mut()) {
                    let factor_row = &factor_row;
                    s.spawn(move |_| {
                        for &k in rows {
                            if !factor_row(k, scratch) {
                                break;
                            }
                        }
                    });
                }
            });
        }
    }

    if zero_pivot.load(Ordering::Relaxed) {
        return Err(QDLDLError::ZeroPivot);
    }

    workspace.regularize_count = regularize_count_total + regularize_count_atomic.load(Ordering::Relaxed);
    Ok(positive_total + positive_count.load(Ordering::Relaxed))
}

// Solves (L+I)x = b, with x replacing b (with standard bounds checks)
fn _lsolve_safe<T: FloatT>(Lp: &[usize], Li: &[usize], Lx: &[T], x: &mut [T]) {
    for i in 0..x.len() {
        let xi = x[i];
        let (f, l) = (Lp[i], Lp[i + 1]);
        let Lx = &Lx[f..l];
        let Li = &Li[f..l];
        for (&Lij, &Lxj) in zip(Li, Lx) {
            x[Lij] -= Lxj * xi;
        }
    }
}

// Solves (L+I)'x = b, with x replacing b (with standard bounds checks)
fn _ltsolve_safe<T: FloatT>(Lp: &[usize], Li: &[usize], Lx: &[T], x: &mut [T]) {
    for i in (0..x.len()).rev() {
        let mut s = T::zero();
        let (f, l) = (Lp[i], Lp[i + 1]);
        let Lx = &Lx[f..l];
        let Li = &Li[f..l];
        for (&Lij, &Lxj) in zip(Li, Lx) {
            s += Lxj * x[Lij];
        }
        x[i] -= s;
    }
}

// -------------------------------------
// Versions of L\x and Lᵀ \ x that use unchecked indexing.
//
// Safety : The values in colptr array Lp at the time this
// function is reached should be bounded by the sizes of the
// arrays in Lx and Li.  The length of x should be compatible
// with the row index entries in Li
// -------------------------------------

// Solves (L+I)x = b, with x replacing b.  Unchecked version
fn _lsolve_unsafe<T: FloatT>(Lp: &[usize], Li: &[usize], Lx: &[T], x: &mut [T]) {
    unsafe {
        for i in 0..x.len() {
            let xi = *x.get_unchecked(i);
            let f = *Lp.get_unchecked(i);
            let l = *Lp.get_unchecked(i + 1);
            for (&Lxj, &Lij) in zip(&Lx[f..l], &Li[f..l]) {
                *(x.get_unchecked_mut(Lij)) -= Lxj * xi;
            }
        }
    }
}

// Solves (L+I)'x = b, with x replacing b.  Unchecked version.
fn _ltsolve_unsafe<T: FloatT>(Lp: &[usize], Li: &[usize], Lx: &[T], x: &mut [T]) {
    unsafe {
        for i in (0..x.len()).rev() {
            let mut s = T::zero();
            let f = *Lp.get_unchecked(i);
            let l = *Lp.get_unchecked(i + 1);
            for (&Lxj, &Lij) in zip(&Lx[f..l], &Li[f..l]) {
                s += Lxj * (*x.get_unchecked(Lij));
            }
            *x.get_unchecked_mut(i) -= s;
        }
    }
}

// Solves D(L+I)'x = b, with x replacing b.  Unchecked version.
fn _dltsolve_unsafe<T: FloatT>(Lp: &[usize], Li: &[usize], Lx: &[T], Dinv: &[T], x: &mut [T]) {
    unsafe {
        for i in (0..x.len()).rev() {
            let mut s = T::zero();
            let f = *Lp.get_unchecked(i);
            let l = *Lp.get_unchecked(i + 1);
            for (&Lxj, &Lij) in zip(&Lx[f..l], &Li[f..l]) {
                s += Lxj * (*x.get_unchecked(Lij));
            }

            let xi = x.get_unchecked_mut(i);
            *xi *= *Dinv.get_unchecked(i);
            *xi -= s;
        }
    }
}

// Solves Ax = b where A has given LDL factors, with x replacing b
fn _solve<T: FloatT>(Lp: &[usize], Li: &[usize], Lx: &[T], Dinv: &[T], b: &mut [T]) {
    // We call the `unsafe`d version of the forward and backward substitution
    // functions here, since the matrix data should be well posed and x of
    // compatible dimensions.   For super safety or debugging purposes, there
    // are also `safe` versions implemented above.
    _lsolve_unsafe(Lp, Li, Lx, b);

    // combined D and L^T solve in one pass
    _dltsolve_unsafe(Lp, Li, Lx, Dinv, b);

    // in separate passes for reference
    //zip(b.iter_mut(), Dinv).for_each(|(b, d)| *b *= *d);
    //_ltsolve_unsafe(Lp, Li, Lx, b);
}

// Construct an inverse permutation from a permutation
fn _invperm(p: &[usize]) -> Result<Vec<usize>, QDLDLError> {
    let mut b = vec![0; p.len()];

    for (i, j) in p.iter().enumerate() {
        if *j < p.len() && b[*j] == 0 {
            b[*j] = i;
        } else {
            return Err(QDLDLError::InvalidPermutation);
        }
    }
    Ok(b)
}

// permutation and inverse permutation
// functions that require no allocation
// p must be a valid permutation vector
// in both cases for safety

pub(crate) fn permute<T: Copy>(x: &mut [T], b: &[T], p: &[usize]) {
    debug_assert!(p.is_empty() || *p.iter().max().unwrap() < x.len());
    unsafe {
        zip(p, x).for_each(|(p, x)| *x = *b.get_unchecked(*p));
    }
}

pub(crate) fn ipermute<T: Copy>(x: &mut [T], b: &[T], p: &[usize]) {
    debug_assert!(p.is_empty() || *p.iter().max().unwrap() < x.len());
    unsafe {
        zip(p, b).for_each(|(p, b)| *x.get_unchecked_mut(*p) = *b);
    }
}

// Given a sparse symmetric matrix `A` (with only upper triangular entries), return
// permuted sparse symmetric matrix `P` (also only upper triangular) given the
// inverse permutation vector `iperm`."
pub(crate) fn permute_symmetric<T: FloatT>(
    A: &CscMatrix<T>,
    iperm: &[usize],
) -> (CscMatrix<T>, Vec<usize>) {
    // perform a number of argument checks
    let (_m, n) = A.size();
    let mut P = CscMatrix::<T>::spalloc((n, n), A.nnz());

    // we will record a mapping of entries from A to PAPt
    let mut AtoPAPt = vec![0; A.nnz()];

    _permute_symmetric_inner(
        A,
        &mut AtoPAPt,
        iperm,
        &mut P.rowval,
        &mut P.colptr,
        &mut P.nzval,
    );
    (P, AtoPAPt)
}

// the main function without extra argument checks
// following the book: Timothy Davis - Direct Methods for Sparse Linear Systems

fn _permute_symmetric_inner<T: FloatT>(
    A: &CscMatrix<T>,
    AtoPAPt: &mut [usize],
    iperm: &[usize],
    Pr: &mut [usize],
    Pc: &mut [usize],
    Pv: &mut [T],
) {
    // 1. count number of entries that each column of P will have
    let n = A.nrows();
    let mut num_entries = vec![0; n];
    let Ar = &A.rowval;
    let Ac = &A.colptr;
    let Av = &A.nzval;

    // count the number of upper-triangle entries in columns of P,
    // keeping in mind the row permutation
    for colA in 0..n {
        let colP = iperm[colA];
        // loop over entries of A in column A...
        for rowA in Ar.iter().take(Ac[colA + 1]).skip(Ac[colA]) {
            let rowP = iperm[*rowA];
            // ...and check if entry is upper triangular
            if *rowA <= colA {
                // determine to which column the entry belongs after permutation
                let col_idx = max(rowP, colP);
                num_entries[col_idx] += 1;
            }
        }
    }

    // 2. calculate permuted Pc = P.colptr from number of entries
    // Pc is one longer than num_entries here.
    Pc[0] = 0;
    let mut acc = 0;
    for (Pckp1, ne) in zip(&mut Pc[1..], &num_entries) {
        *Pckp1 = acc + ne;
        acc = *Pckp1;
    }
    // reuse this memory to keep track of free entries in rowval
    num_entries.copy_from_slice(&Pc[0..n]);

    // use alias
    let mut row_starts = num_entries;

    // 3. permute the row entries and position of corresponding nzval
    for colA in 0..n {
        let colP = iperm[colA];
        // loop over rows of A and determine where each row entry of A should be stored
        for rowA_idx in Ac[colA]..Ac[colA + 1] {
            let rowA = Ar[rowA_idx];
            // check if upper triangular
            if rowA <= colA {
                let rowP = iperm[rowA];
                // determine column to store the entry
                let col_idx = max(colP, rowP);

                // find next free location in rowval (this results in unordered columns in the rowval)
                let rowP_idx = row_starts[col_idx];

                // store rowval and nzval
                Pr[rowP_idx] = min(colP, rowP);
                Pv[rowP_idx] = Av[rowA_idx];

                //record this into the mapping vector
                AtoPAPt[rowA_idx] = rowP_idx;

                // increment next free location
                row_starts[col_idx] += 1;
            }
        }
    }
}

pub(crate) fn get_amd_ordering<T: FloatT>(
    A: &CscMatrix<T>,
    amd_dense_scale: f64,
) -> (Vec<usize>, Vec<usize>, amd::Info) {
    // PJG: For interested readers - setting amd_dense_scale to 1.5 seems to work better
    // for KKT systems in QP problems, but this ad hoc method can surely be improved

    // computes a permutation for A using AMD default parameters
    let mut control = amd::Control::default();
    control.dense *= amd_dense_scale; //increase the default value
    let (perm, iperm, info) = amd::order(A.nrows(), &A.colptr, &A.rowval, &control).unwrap();
    (perm, iperm, info)
}

//configure tests of internals
#[path = "test.rs"]
#[cfg(test)]
mod test;
