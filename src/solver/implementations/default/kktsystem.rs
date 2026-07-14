use super::*;
use crate::solver::core::{
    cones::{CompositeCone, Cone},
    kktsolvers::{direct::*, *},
    traits::{KKTSystem, Settings},
    StepDirection,
};

use crate::algebra::*;

// We require Send/Sync here to allow pyo3 builds to share
// solver objects between threads.

type BoxedKKTSolver<T> = Box<dyn KKTSolver<T> + Send + Sync>;

/// Standard-form solver type implementing the [`KKTSystem`](crate::solver::core::traits::KKTSystem) trait
pub struct DefaultKKTSystem<T> {
    kktsolver: BoxedKKTSolver<T>,

    // true while the (approximate) condensed solver is in use.  If it ever
    // degrades we rebuild the exact direct solver and never go back, so the
    // solve can only ever be as accurate as the direct method plus a bounded
    // amount of wasted work.
    is_approximate: bool,

    // solution vector for constant part of KKT solves
    x1: Vec<T>,
    z1: Vec<T>,

    // solution vector for general KKT solves
    x2: Vec<T>,
    z2: Vec<T>,

    // work vectors for assembling/dissambling vectors
    workx: Vec<T>,
    workz: Vec<T>,
    work_conic: Vec<T>,
}

impl<T> DefaultKKTSystem<T>
where
    T: FloatT,
{
    /// Create a new KKT system solver
    pub fn new(
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> Self {
        let (m, n) = (data.m, data.n);

        //here we allow scope for different KKT solvers, e.g.
        //direct vs indirect, different QR based direct methods
        //etc.   For now, we only have direct / LDL based
        let mut is_approximate = false;
        let kktsolver: Box<dyn KKTSolver<T> + Send + Sync> = if settings.direct_kkt_solver {
            // opt-in condensed (normal equations + Woodbury) solver for
            // problems with the right structure; falls back to direct LDL
            let condensed = if std::env::var("CLARABEL_CONDENSED_KKT").is_ok() {
                crate::solver::core::kktsolvers::condensed::CondensedKKTSolver::<T>::try_new(
                    &data.P,
                    &data.A,
                    cones,
                    m,
                    n,
                    settings.core(),
                )
            } else {
                None
            };
            match condensed {
                Some(solver) => {
                    is_approximate = true;
                    Box::new(solver)
                }
                None => Box::new(DirectLDLKKTSolver::<T>::new(
                    &data.P,
                    &data.A,
                    cones,
                    m,
                    n,
                    settings.core(),
                )),
            }
        } else {
            panic!("Indirect and other solve strategies not yet supported.");
        };

        //the LHS constant part of the reduced solve
        let x1 = vec![T::zero(); n];
        let z1 = vec![T::zero(); m];

        //the LHS for other solves
        let x2 = vec![T::zero(); n];
        let z2 = vec![T::zero(); m];

        //workspace compatible with (x,z)
        let workx = vec![T::zero(); n];
        let workz = vec![T::zero(); m];

        //additional conic workspace vector compatible with s and z
        let work_conic = vec![T::zero(); m];

        Self {
            kktsolver,
            is_approximate,
            x1,
            z1,
            x2,
            z2,
            workx,
            workz,
            work_conic,
        }
    }
}

impl<T> HasLinearSolverInfo for DefaultKKTSystem<T>
where
    T: FloatT,
{
    fn linear_solver_info(&self) -> LinearSolverInfo {
        self.kktsolver.linear_solver_info()
    }
}

impl<T> KKTSystem<T> for DefaultKKTSystem<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;
    type C = CompositeCone<T>;
    type SE = DefaultSettings<T>;

    fn update(
        &mut self,
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        // update the linear solver with new cones
        let mut is_success = self.kktsolver.update(cones, settings.core());

        // calculate KKT solution for constant terms.  This solve doubles as a
        // health check on an approximate solver, since it uses the scalings that
        // the step solves in this iteration will use.
        if is_success {
            is_success = self.solve_constant_rhs(data, settings.core());
        }

        if self.is_approximate && (!is_success || self.kktsolver.is_degraded()) {
            is_success = self.revert_to_direct(data, cones, settings)
                && self.solve_constant_rhs(data, settings.core());
        }

        is_success

        //PJG is_success should be a Result in rust
    }

    fn solve(
        &mut self,
        lhs: &mut DefaultVariables<T>,
        rhs: &DefaultVariables<T>,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        cones: &mut CompositeCone<T>,
        step_direction: StepDirection,
        settings: &DefaultSettings<T>,
    ) -> bool {
        // solve for (x1,z1)
        // -----------
        self.workx.copy_from(&rhs.x);

        // compute the vector c in the step equation HₛΔz + Δs = -c,
        // with shortcut in affine case
        let Δs_const_term = &mut self.work_conic;

        match step_direction {
            StepDirection::Affine => {
                Δs_const_term.copy_from(&variables.s);
            }
            StepDirection::Combined => {
                cones.Δs_from_Δz_offset(Δs_const_term, &rhs.s, &mut lhs.z, &variables.z);
            }
        }

        self.workz
            .waxpby(T::one(), Δs_const_term, -T::one(), &rhs.z);

        // ---------------------------------------------------
        // this solves the variable part of reduced KKT system
        let is_success = self.solve_step_rhs(data, cones, settings);
        if !is_success {
            return false;
        }

        let (x1, z1) = (&self.x1, &self.z1);
        let (x2, z2) = (&self.x2, &self.z2); //from constant solve, so not mut
        let workx = &mut self.workx;

        // solve for Δτ.
        // -----------
        // Numerator first
        let ξ = workx;
        ξ.axpby(T::recip(variables.τ), &variables.x, T::zero());

        let two: T = (2.).as_T();
        let tau_num = rhs.τ - rhs.κ / variables.τ
            + data.q.dot(x1)
            + data.b.dot(z1)
            + two * data.P.sym_up().quad_form(ξ, x1);

        // offset ξ for the quadratic form in the denominator
        let ξ_minus_x2 = ξ; //alias to ξ, same as workx
        ξ_minus_x2.axpby(-T::one(), x2, T::one());

        let mut tau_den = variables.κ / variables.τ - data.q.dot(x2) - data.b.dot(z2);
        tau_den +=
            data.P.sym_up().quad_form(ξ_minus_x2, ξ_minus_x2) - data.P.sym_up().quad_form(x2, x2);

        // solve for (Δx,Δz)
        // -----------
        lhs.τ = tau_num / tau_den;
        lhs.x.waxpby(T::one(), x1, lhs.τ, x2);
        lhs.z.waxpby(T::one(), z1, lhs.τ, z2);

        // solve for Δs
        // -------------
        //  compute the linear term HₛΔz, where Hs = WᵀW for symmetric
        //  cones and Hs = μH(z) for asymmetric cones
        cones.mul_Hs(&mut lhs.s, &lhs.z, &mut self.workz);
        lhs.s.axpby(-T::one(), &self.work_conic, -T::one()); // lhs.s = -(lhs.s+Δs_const_term);

        // solve for Δκ
        // --------------
        lhs.κ = -(rhs.κ + variables.κ * lhs.τ) / variables.τ;

        // we don't check the validity of anything
        // after the KKT solve, so just return is_success
        // without further validation
        is_success
    }

    fn solve_initial_point(
        &mut self,
        variables: &mut DefaultVariables<T>,
        data: &DefaultProblemData<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        let mut is_success;

        if data.P.nnz() == 0 {
            // LP initialization
            // solve with [0;b] as a RHS to get (x,-s) initializers
            // zero out any sparse cone variables at end
            self.workx.fill(T::zero());
            self.workz.copy_from(&data.b);
            self.kktsolver.setrhs(&self.workx, &self.workz);
            is_success = self.kktsolver.solve(
                Some(&mut variables.x),
                Some(&mut variables.s),
                settings.core(),
            );
            variables.s.negate();

            if !is_success {
                return is_success;
            }

            // solve with [-q;0] as a RHS to get z initializer
            // zero out any sparse cone variables at end
            self.workx.axpby(-T::one(), &data.q, T::zero());
            self.workz.fill(T::zero());

            self.kktsolver.setrhs(&self.workx, &self.workz);
            is_success = self
                .kktsolver
                .solve(None, Some(&mut variables.z), settings.core());
        } else {
            //QP initialization
            self.workx.scalarop_from(|q| -q, &data.q);
            self.workz.copy_from(&data.b);
            self.kktsolver.setrhs(&self.workx, &self.workz);
            is_success = self.kktsolver.solve(
                Some(&mut variables.x),
                Some(&mut variables.z),
                settings.core(),
            );
            variables.s.scalarop_from(|z| -z, &variables.z);
        }
        is_success
    }
}

impl<T> DefaultKKTSystem<T>
where
    T: FloatT,
{
    // solve the reduced system for (x1,z1) from the rhs already in (workx,workz),
    // reverting to the direct solver and redoing the whole iteration's solves if
    // an approximate solver failed or reports that it has degraded
    fn solve_step_rhs(
        &mut self,
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        let mut x1 = std::mem::take(&mut self.x1);
        let mut z1 = std::mem::take(&mut self.z1);

        self.kktsolver.setrhs(&self.workx, &self.workz);
        let mut is_success = self
            .kktsolver
            .solve(Some(&mut x1), Some(&mut z1), settings.core());

        if self.is_approximate && (!is_success || self.kktsolver.is_degraded()) {
            let (bx, bz) = (self.workx.clone(), self.workz.clone());
            // the constant part was solved with the old solver, so redo it too
            is_success = self.revert_to_direct(data, cones, settings)
                && self.solve_constant_rhs(data, settings.core());
            if is_success {
                self.workx.copy_from_slice(&bx);
                self.workz.copy_from_slice(&bz);
                self.kktsolver.setrhs(&self.workx, &self.workz);
                is_success = self
                    .kktsolver
                    .solve(Some(&mut x1), Some(&mut z1), settings.core());
            }
        }

        self.x1 = x1;
        self.z1 = z1;
        is_success
    }

    // replace an approximate solver with the exact direct one, factoring it with
    // the current cone scalings.  Never reverses: once the condensed form has
    // shown itself inadequate for this problem it stays out of the way.
    fn revert_to_direct(
        &mut self,
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        self.kktsolver = Box::new(DirectLDLKKTSolver::<T>::new(
            &data.P,
            &data.A,
            cones,
            data.m,
            data.n,
            settings.core(),
        ));
        self.is_approximate = false;
        if settings.verbose {
            println!("condensed KKT solver degraded; reverting to direct LDL");
        }
        self.kktsolver.update(cones, settings.core())
    }

    fn solve_constant_rhs(
        &mut self,
        data: &DefaultProblemData<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        self.workx.axpby(-T::one(), &data.q, T::zero()); //workx .= -q
        self.kktsolver.setrhs(&self.workx, &data.b);
        let is_success =
            self.kktsolver
                .solve(Some(&mut self.x2), Some(&mut self.z2), settings.core());

        is_success
    }

    pub(crate) fn update_P(&mut self, P: &CscMatrix<T>) {
        self.kktsolver.update_P(P);
    }

    pub(crate) fn update_A(&mut self, A: &CscMatrix<T>) {
        self.kktsolver.update_A(A);
    }
}
