use super::*;
use crate::algebra::*;
use crate::io::PrintTarget;
use crate::solver::core::ffi::*;
use crate::solver::core::kktsolvers::LinearSolverInfo;
use crate::solver::core::{traits::Info, SolverStatus};
use crate::solver::traits::Variables;
use crate::timers::*;

/// Standard-form solver type implementing the [`Info`](crate::solver::core::traits::Info) and [`InfoPrint`](crate::solver::core::traits::InfoPrint) traits
#[repr(C)]
#[derive(Default, Debug, Clone)]
pub struct DefaultInfo<T> {
    /// interior point path parameter μ
    pub mu: T,
    /// interior point path parameter reduction ratio σ
    pub sigma: T,
    /// step length for the current iteration
    pub step_length: T,
    /// number of iterations
    pub iterations: u32,
    /// primal objective value
    pub cost_primal: T,
    /// dual objective value
    pub cost_dual: T,
    /// primal residual
    pub res_primal: T,
    /// dual residual
    pub res_dual: T,
    /// primal infeasibility residual
    pub res_primal_inf: T,
    /// dual infeasibility residual
    pub res_dual_inf: T,
    /// absolute duality gap
    pub gap_abs: T,
    /// relative duality gap
    pub gap_rel: T,
    /// κ/τ ratio
    pub ktratio: T,

    // previous iterate
    /// primal object value from previous iteration
    pub(crate) prev_cost_primal: T,
    /// dual objective value from previous iteration
    pub(crate) prev_cost_dual: T,
    /// primal residual from previous iteration
    pub(crate) prev_res_primal: T,
    /// dual residual from previous iteration
    pub(crate) prev_res_dual: T,
    /// absolute duality gap from previous iteration
    pub(crate) prev_gap_abs: T,
    /// relative duality gap from previous iteration
    pub(crate) prev_gap_rel: T,

    // best iterate seen so far, by termination-criteria merit.
    // Restored on solver failure so that the reported solution is
    // the best one encountered rather than the point at which the
    // solver gave up.   `best_merit` is ∞ until a checkpoint is taken.
    /// merit of the best iterate (∞ if none yet checkpointed)
    pub(crate) best_merit: T,
    /// primal objective value at the best iterate
    pub(crate) best_cost_primal: T,
    /// dual objective value at the best iterate
    pub(crate) best_cost_dual: T,
    /// primal residual at the best iterate
    pub(crate) best_res_primal: T,
    /// dual residual at the best iterate
    pub(crate) best_res_dual: T,
    /// absolute duality gap at the best iterate
    pub(crate) best_gap_abs: T,
    /// relative duality gap at the best iterate
    pub(crate) best_gap_rel: T,
    /// κ/τ ratio at the best iterate
    pub(crate) best_ktratio: T,
    /// solve time
    pub solve_time: f64,
    /// solver status
    pub status: SolverStatus,

    /// linear solver information
    pub linsolver: LinearSolverInfo,

    // target stream for printing
    pub(crate) stream: PrintTarget,
}

impl<T> DefaultInfo<T>
where
    T: FloatT,
{
    /// creates a new `DefaultInfo` object
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: FloatT> ClarabelFFI<Self> for DefaultInfo<T> {
    type FFI = super::ffi::DefaultInfoFFI<T>;
}

impl<T> Info<T> for DefaultInfo<T>
where
    T: FloatT,
{
    type V = DefaultVariables<T>;
    type R = DefaultResiduals<T>;

    fn reset(&mut self, timers: &mut Timers) {
        self.status = SolverStatus::Unsolved;
        self.iterations = 0;
        self.solve_time = 0f64;
        self.best_merit = T::infinity();

        timers.reset_timer("solve");
    }

    fn post_process(&mut self, residuals: &DefaultResiduals<T>, settings: &DefaultSettings<T>) {
        // if there was an error or we ran out of time
        // or iterations, check for partial convergence

        if self.status.is_errored()
            || matches!(self.status, SolverStatus::MaxIterations)
            || matches!(self.status, SolverStatus::MaxTime)
        {
            self.check_convergence_almost(residuals, settings);
        }
    }

    fn finalize(&mut self, timers: &mut Timers) {
        //final check of timers
        self.solve_time = timers.total_time().as_secs_f64();
    }

    fn update(
        &mut self,
        data: &mut DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        residuals: &DefaultResiduals<T>,
        timers: &Timers,
    ) {
        // optimality termination check should be computed w.r.t
        // the pre-homogenization x and z variables.
        let τinv = T::recip(variables.τ);

        // unscaled linear term norms
        let normb = data.get_normb();
        let normq = data.get_normq();

        // shortcuts for the equilibration matrices
        let d = &data.equilibration.d;
        let e = &data.equilibration.e;
        let dinv = &data.equilibration.dinv;
        let einv = &data.equilibration.einv;
        let cinv = T::recip(data.equilibration.c);

        // primal and dual costs. dot products are invariant w.r.t
        // equilibration, but we still need to back out the overall
        // objective scaling term c

        let xPx_τinvsq_over2 = residuals.dot_xPx * τinv * τinv / (2.).as_T();
        self.cost_primal = (residuals.dot_qx * τinv + xPx_τinvsq_over2) * cinv;
        self.cost_dual = (-residuals.dot_bz * τinv - xPx_τinvsq_over2) * cinv;

        // variables norms, undoing the equilibration.  Do not unscale
        // by τ yet because the infeasibility residuals are ratios of
        // terms that have no affine parts anyway
        let mut normx = variables.x.norm_scaled(d);
        let mut normz = variables.z.norm_scaled(e) * cinv;
        let mut norms = variables.s.norm_scaled(einv);

        // primal and dual infeasibility residuals.
        self.res_primal_inf = (residuals.rx_inf.norm_scaled(dinv) * cinv) / T::max(T::one(), normz);
        self.res_dual_inf = T::max(
            residuals.Px.norm_scaled(dinv) / T::max(T::one(), normx),
            residuals.rz_inf.norm_scaled(einv) / T::max(T::one(), normx + norms),
        );

        // now back out the τ scaling so we can normalize the unscaled primal / dual errors
        normx *= τinv;
        normz *= τinv;
        norms *= τinv;

        // primal and dual relative residuals.
        self.res_primal =
            residuals.rz.norm_scaled(einv) * τinv / T::max(T::one(), normb + normx + norms);
        self.res_dual =
            residuals.rx.norm_scaled(dinv) * τinv * cinv / T::max(T::one(), normq + normx + normz);

        // absolute and relative gaps
        self.gap_abs = T::abs(self.cost_primal - self.cost_dual);
        self.gap_rel = self.gap_abs
            / T::max(
                T::one(),
                T::min(T::abs(self.cost_primal), T::abs(self.cost_dual)),
            );

        // κ/τ ratio (scaled)
        self.ktratio = variables.κ * τinv;

        // solve time so far (includes setup)
        self.solve_time = timers.total_time().as_secs_f64();
    }

    fn check_termination(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &DefaultSettings<T>,
        iter: u32,
    ) -> bool {
        //  optimality or infeasibility
        // ---------------------
        self.check_convergence_full(residuals, settings);

        //  poor progress
        // ----------------------
        if self.status == SolverStatus::Unsolved
            && iter > 1u32
            && (self.res_dual > self.prev_res_dual || self.res_primal > self.prev_res_primal)
        {
            // Poor progress at high tolerance.
            if self.ktratio < T::epsilon() * (100.).as_T()
                && (self.prev_gap_abs < settings.tol_gap_abs
                    || self.prev_gap_rel < settings.tol_gap_rel)
            {
                self.status = SolverStatus::InsufficientProgress;
            }

            // Going backwards. Stop immediately if residuals diverge out of feasibility tolerance.
            #[allow(clippy::collapsible_if)] // nested if for readability
            if self.ktratio < T::one() {
                if (self.res_dual > settings.tol_feas * (100.).as_T()
                    && self.res_dual > self.prev_res_dual * (100.).as_T())
                    || (self.res_primal > settings.tol_feas * (100.).as_T()
                        && self.res_primal > self.prev_res_primal * (100.).as_T())
                {
                    self.status = SolverStatus::InsufficientProgress;
                }
            }
        }

        // time or iteration limits
        // ----------------------
        if self.status == SolverStatus::Unsolved {
            if settings.max_iter == self.iterations {
                self.status = SolverStatus::MaxIterations;
            } else if self.solve_time > settings.time_limit {
                self.status = SolverStatus::MaxTime;
            }
        }

        // return TRUE if we settled on a final status
        self.status != SolverStatus::Unsolved
    }

    fn checkpoint_iterate(
        &mut self,
        variables: &Self::V,
        best_variables: &mut Self::V,
        settings: &DefaultSettings<T>,
    ) {
        // scalars from the previous iteration, used by the
        // poor-progress tests in check_termination
        self.prev_cost_primal = self.cost_primal;
        self.prev_cost_dual = self.cost_dual;
        self.prev_res_primal = self.res_primal;
        self.prev_res_dual = self.res_dual;
        self.prev_gap_abs = self.gap_abs;
        self.prev_gap_rel = self.gap_rel;

        // Additionally checkpoint the iterate itself if it is the best
        // seen so far, so that a failed solve can still report the best
        // point encountered rather than the one at which it gave up.
        // Only iterates on the optimality branch (κ/τ ≤ 1) are candidates:
        // for κ/τ > 1 the iterate is trending towards an infeasibility
        // certificate and its cost/gap values are not meaningful.
        if self.ktratio <= T::one() {
            let merit = self.termination_merit(settings);
            if merit.is_finite() && merit < self.best_merit {
                self.best_merit = merit;
                self.best_cost_primal = self.cost_primal;
                self.best_cost_dual = self.cost_dual;
                self.best_res_primal = self.res_primal;
                self.best_res_dual = self.res_dual;
                self.best_gap_abs = self.gap_abs;
                self.best_gap_rel = self.gap_rel;
                self.best_ktratio = self.ktratio;
                best_variables.copy_from(variables);
            }
        }
    }

    fn reset_to_best_iterate(&mut self, variables: &mut Self::V, best_variables: &Self::V) -> bool {
        // nothing to restore if no iterate was ever checkpointed.   If the
        // current iterate has κ/τ > 1 it is trending towards an
        // infeasibility certificate, which a restore would mask, so keep it.
        if !self.best_merit.is_finite() || self.ktratio > T::one() {
            return false;
        }

        self.cost_primal = self.best_cost_primal;
        self.cost_dual = self.best_cost_dual;
        self.res_primal = self.best_res_primal;
        self.res_dual = self.best_res_dual;
        self.gap_abs = self.best_gap_abs;
        self.gap_rel = self.best_gap_rel;
        self.ktratio = self.best_ktratio;

        variables.copy_from(best_variables);
        true
    }

    fn save_scalars(&mut self, μ: T, α: T, σ: T, iter: u32) {
        self.mu = μ;
        self.step_length = α;
        self.sigma = σ;
        self.iterations = iter;
    }

    fn get_status(&self) -> SolverStatus {
        self.status
    }

    fn set_status(&mut self, status: SolverStatus) {
        self.status = status;
    }
}

// Utility functions for convergence checkiing

impl<T> DefaultInfo<T>
where
    T: FloatT,
{
    fn check_convergence_full(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &DefaultSettings<T>,
    ) {
        // "full" tolerances
        let tol_gap_abs = settings.tol_gap_abs;
        let tol_gap_rel = settings.tol_gap_rel;
        let tol_feas = settings.tol_feas;
        let tol_infeas_abs = settings.tol_infeas_abs;
        let tol_infeas_rel = settings.tol_infeas_rel;
        let tol_ktratio = settings.tol_ktratio;

        let solved_status = SolverStatus::Solved;
        let pinf_status = SolverStatus::PrimalInfeasible;
        let dinf_status = SolverStatus::DualInfeasible;

        self.check_convergence(
            residuals,
            tol_gap_abs,
            tol_gap_rel,
            tol_feas,
            tol_infeas_abs,
            tol_infeas_rel,
            tol_ktratio,
            solved_status,
            pinf_status,
            dinf_status,
        );
    }

    fn check_convergence_almost(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &DefaultSettings<T>,
    ) {
        // "almost" tolerances
        let tol_gap_abs = settings.reduced_tol_gap_abs;
        let tol_gap_rel = settings.reduced_tol_gap_rel;
        let tol_feas = settings.reduced_tol_feas;
        let tol_infeas_abs = settings.reduced_tol_infeas_abs;
        let tol_infeas_rel = settings.reduced_tol_infeas_rel;
        let tol_ktratio = settings.reduced_tol_ktratio;

        let solved_status = SolverStatus::AlmostSolved;
        let pinf_status = SolverStatus::AlmostPrimalInfeasible;
        let dinf_status = SolverStatus::AlmostDualInfeasible;

        self.check_convergence(
            residuals,
            tol_gap_abs,
            tol_gap_rel,
            tol_feas,
            tol_infeas_abs,
            tol_infeas_rel,
            tol_ktratio,
            solved_status,
            pinf_status,
            dinf_status,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn check_convergence(
        &mut self,
        residuals: &DefaultResiduals<T>,
        tol_gap_abs: T,
        tol_gap_rel: T,
        tol_feas: T,
        tol_infeas_abs: T,
        tol_infeas_rel: T,
        tol_ktratio: T,
        solved_status: SolverStatus,
        pinf_status: SolverStatus,
        dinf_status: SolverStatus,
    ) {
        if self.ktratio <= T::one() && self.is_solved(tol_gap_abs, tol_gap_rel, tol_feas) {
            self.status = solved_status;
        //PJG hardcoded factor 1000 here should be fixed
        } else if self.ktratio > tol_ktratio.recip() * (1000.0).as_T() {
            if self.is_primal_infeasible(residuals, tol_infeas_abs, tol_infeas_rel) {
                self.status = pinf_status;
            } else if self.is_dual_infeasible(residuals, tol_infeas_abs, tol_infeas_rel) {
                self.status = dinf_status;
            }
        }
    }

    fn is_solved(&self, tol_gap_abs: T, tol_gap_rel: T, tol_feas: T) -> bool {
        ((self.gap_abs < tol_gap_abs) || (self.gap_rel < tol_gap_rel))
            && (self.res_primal < tol_feas)
            && (self.res_dual < tol_feas)
    }

    // Distance of the current iterate from satisfying the full-accuracy
    // `is_solved` test: each termination quantity normalized by its
    // tolerance, combined exactly as in that test (the duality gap counts
    // via whichever of its absolute/relative forms is closer to passing).
    // An iterate with merit < 1 would terminate as Solved.
    fn termination_merit(&self, settings: &DefaultSettings<T>) -> T {
        let gap = T::min(
            self.gap_abs / settings.tol_gap_abs,
            self.gap_rel / settings.tol_gap_rel,
        );
        let feas = T::max(
            self.res_primal / settings.tol_feas,
            self.res_dual / settings.tol_feas,
        );
        T::max(gap, feas)
    }

    fn is_primal_infeasible(
        &self,
        residuals: &DefaultResiduals<T>,
        tol_infeas_abs: T,
        tol_infeas_rel: T,
    ) -> bool {
        (residuals.dot_bz < -tol_infeas_abs)
            && (self.res_primal_inf < -tol_infeas_rel * residuals.dot_bz)
    }

    fn is_dual_infeasible(
        &self,
        residuals: &DefaultResiduals<T>,
        tol_infeas_abs: T,
        tol_infeas_rel: T,
    ) -> bool {
        (residuals.dot_qx < -tol_infeas_abs)
            && (self.res_dual_inf < -tol_infeas_rel * residuals.dot_qx)
    }
}

// ---------------------------------------------------------------------------
// best-iterate checkpointing tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod test {
    use super::*;
    use crate::solver::core::traits::Info;

    // gives an info with the stated termination quantities and κ/τ,
    // with best_merit initialized as at the start of a solve
    fn info_with(res: f64, gap: f64, ktratio: f64) -> DefaultInfo<f64> {
        let mut info = DefaultInfo::<f64>::default();
        info.best_merit = f64::INFINITY;
        info.res_primal = res;
        info.res_dual = res;
        info.gap_abs = gap;
        info.gap_rel = gap;
        info.ktratio = ktratio;
        info
    }

    fn set_current(info: &mut DefaultInfo<f64>, res: f64, gap: f64, ktratio: f64) {
        info.res_primal = res;
        info.res_dual = res;
        info.gap_abs = gap;
        info.gap_rel = gap;
        info.ktratio = ktratio;
    }

    #[test]
    fn test_best_iterate_checkpoint_and_restore() {
        let settings = DefaultSettings::<f64>::default();
        let mut info = info_with(1e-3, 1e-3, 0.5);
        let mut best = DefaultVariables::<f64>::new(2, 1);
        let mut vars = DefaultVariables::<f64>::new(2, 1);

        // first candidate is checkpointed
        vars.x[0] = 1.0;
        info.cost_primal = 10.0;
        info.checkpoint_iterate(&vars, &mut best, &settings);
        assert_eq!(best.x[0], 1.0);

        // a worse iterate is not
        set_current(&mut info, 1e-2, 1e-2, 0.5);
        vars.x[0] = 2.0;
        info.checkpoint_iterate(&vars, &mut best, &settings);
        assert_eq!(best.x[0], 1.0);

        // a better one is
        set_current(&mut info, 1e-6, 1e-6, 0.5);
        vars.x[0] = 3.0;
        info.cost_primal = 30.0;
        info.checkpoint_iterate(&vars, &mut best, &settings);
        assert_eq!(best.x[0], 3.0);

        // an infeasibility-trending iterate is never a candidate,
        // however good its (meaningless) residuals look
        set_current(&mut info, 1e-9, 1e-9, 2.0);
        vars.x[0] = 4.0;
        info.checkpoint_iterate(&vars, &mut best, &settings);
        assert_eq!(best.x[0], 3.0);

        // restore declines while the current iterate trends infeasible...
        assert!(!info.reset_to_best_iterate(&mut vars, &best));
        assert_eq!(vars.x[0], 4.0);

        // ...and otherwise restores the checkpointed variables and scalars
        set_current(&mut info, 1e-1, 1e-1, 0.5);
        info.cost_primal = 99.0;
        assert!(info.reset_to_best_iterate(&mut vars, &best));
        assert_eq!(vars.x[0], 3.0);
        assert_eq!(info.cost_primal, 30.0);
        assert_eq!(info.res_primal, 1e-6);
    }

    #[test]
    fn test_best_iterate_no_checkpoint_no_restore() {
        let mut info = info_with(1e-3, 1e-3, 0.5);
        info.best_merit = f64::INFINITY; // nothing checkpointed
        let best = DefaultVariables::<f64>::new(2, 1);
        let mut vars = DefaultVariables::<f64>::new(2, 1);
        vars.x[0] = 7.0;
        assert!(!info.reset_to_best_iterate(&mut vars, &best));
        assert_eq!(vars.x[0], 7.0);
    }
}
