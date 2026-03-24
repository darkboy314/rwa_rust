/// Game theory module implementing the three-stage game
/// Stage 1: Upstream player (energy storage provider)
/// Stage 2: Stage One players (renewable energy generators)
/// Stage 3: Stage Two players (electricity offtakers)
use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::brent::BrentOpt;

pub const T: f64 = 10.0; // Lifespan years
pub const C_T: f64 = 100.0; // Transaction cost ($)
pub const K: f64 = 600.0; // Development cost per unit ($/MWh)
pub const F: f64 = 1700.0; // Sales price per unit ($/MWh)
// const Q: f64 = 200.0; // Storage capacity (MWh)
// const R: f64 = 0.2; // Profit sharing ratio

pub struct ResultStruct {
    pub f: f64,
    pub reg1_m: f64,
    pub reg2_m: f64,
    pub oft1_m: f64,
    pub oft2_m: f64,
}

pub struct GameParams {
    /// Expected demand
    pub e_d: f64,
    /// Expected price
    pub e_p: f64,
    /// Expected value
    pub e_v: f64,
    /// Expected demand-price covariance
    pub e_dp: f64,
    /// Expected price-value covariance
    pub e_pv: f64,
    /// Expected operating cost
    pub e_cf: f64,
    /// Std dev of operating cost
    pub sigma_cf: f64,
    /// Std dev of demand
    pub sigma_d: f64,
    /// Std dev of price
    pub sigma_p: f64,
    /// Std dev of value
    pub sigma_v: f64,
    /// Std dev of demand-price
    pub sigma_dp: f64,
    /// Std dev of price-value
    pub sigma_pv: f64,

    /// 2nd-order Expected value of P and V
    pub e_p2: f64,
    pub e_v2: f64,
    pub e_p2v: f64,
    pub e_pv2: f64,
    pub e_p2v2: f64,
}

pub struct StageOnePlayer {
    pub m: f64,
    pub b: f64,
    lbd: f64,
}

pub struct StageTwoPlayer {
    pub m: f64,
    pub b: f64,
    gma: f64,
}

impl StageOnePlayer {
    pub fn new(m: f64, b: f64, lbd: f64) -> Self {
        StageOnePlayer { m, b, lbd }
    }

    /// Profit function for stage one player
    pub fn theta(
        &self,
        m: f64,
        up: &UpstreamPlayer,
        params: &GameParams,
        m2: f64,
        m_others: &[f64],
    ) -> f64 {
        let m_all = m + m_others.iter().sum::<f64>() + 1e-10;

        T * params.e_dp
            + (m / m_all)
                * up.r
                * (up.p2 * m2
                    - T * params.e_cf * up.q
                    - self.lbd * T * (params.sigma_cf * params.sigma_cf) * up.q)
            - up.p1 * m
            - C_T
    }

    /// Constraint function (budget constraint)
    pub fn constraint(&self, up: &UpstreamPlayer, m1: f64) -> f64 {
        self.b - C_T - up.p1 * m1
    }
}

impl StageTwoPlayer {
    pub fn new(m: f64, b: f64, gma: f64) -> Self {
        StageTwoPlayer { m, b, gma }
    }

    /// Profit function for stage two player
    pub fn mu(&self, m: f64, up: &UpstreamPlayer, params: &GameParams, m_others: &[f64]) -> f64 {
        let m_all = m_others.iter().sum::<f64>() + m + 1e-10;
        let c = (m / m_all) * up.q;
        let a = F * params.e_v + c * params.e_p - params.e_pv;
        let b = F.powi(2) * params.e_v2
            + 2.0 * F * c * params.e_pv
            + c.powi(2) * params.e_p2
            + params.e_p2v2
            - 2.0 * F * params.e_p2v
            - 2.0 * c * params.e_pv2;

        T * a - self.gma * T * (b - a.powi(2)) - up.p2 * m - C_T
    }

    /// Constraint function
    pub fn constraint(&self, up: &UpstreamPlayer, m2: f64) -> f64 {
        self.b - C_T - up.p2 * m2
    }
}

pub struct UpstreamPlayer {
    q: f64,
    r: f64,
    p1: f64,
    p2: f64,
    n_re: usize,
    n_of: usize,
}

impl UpstreamPlayer {
    pub fn new(q: f64, r: f64, p1: f64, p2: f64, n_re: usize, n_of: usize) -> Self {
        UpstreamPlayer {
            q,
            r,
            p1,
            p2,
            n_re,
            n_of,
        }
    }

    /// Objective function
    pub fn pi_ess(&self, params: &GameParams, m1: f64, m2: f64) -> f64 {
        let q = self.q;
        let r = self.r;
        let p1 = self.p1;
        let p2 = self.p2;
        p1 * m1 + (1.0 - r) * (p2 * m2 - T * q * params.e_cf)
            - (self.n_re + self.n_of) as f64 * C_T
            - K * q
    }

    /// Constraint functions
    pub fn cons_1(&self, m1: f64) -> f64 {
        let q = self.q;
        let p1 = self.p1;
        p1 * m1 - self.n_re as f64 * C_T - K * q
    }

    pub fn cons_2(&self, m2: f64, e_cf: f64) -> f64 {
        let q = self.q;
        let r = self.r;
        let p2 = self.p2;
        (1.0 - r) * (p2 * m2 - T * e_cf * q) - C_T * self.n_of as f64
    }

    pub fn cons_3(&self, m2: f64) -> f64 {
        let q = self.q;
        let p2 = self.p2;
        F * q - p2 * m2
    }
}

struct OneDCost<F>(F);

impl<F: Fn(f64) -> f64> CostFunction for OneDCost<F> {
    type Param = f64;
    type Output = f64;
    fn cost(&self, p: &f64) -> Result<f64, Error> {
        Ok((self.0)(*p))
    }
}

/// Constrained 1-D minimization using Brent's method (pure Rust, jemalloc-safe).
/// The constraint `c(x) >= 0` is a linear budget constraint of the form
/// `b - C_T - price * x >= 0`, which gives a closed-form upper bound
/// `x <= (b - C_T) / price`. That bound is folded into `ub` here.
pub fn alter_best_response<F>(
    f: F,
    x0: f64,
    lb: f64,
    effective_ub: f64,
    tol: f64,
    max_iter: usize,
) -> f64
where
    F: Fn(f64) -> f64,
{
    if effective_ub <= lb {
        return lb;
    }

    // Clamp initial point into the feasible interval.
    let init = x0.clamp(lb, effective_ub);

    let problem = OneDCost(f);
    let solver = BrentOpt::new(lb, effective_ub).set_tolerance(tol, tol);
    match Executor::new(problem, solver)
        .configure(|s| s.param(init).max_iters(max_iter as u64))
        .run()
    {
        Ok(res) => res.state().best_param.unwrap_or(init),
        Err(_) => init,
    }
}

/// Penalty function for genetic algorithm
pub fn penalty_function(up: &UpstreamPlayer, params: &GameParams, result: &ResultStruct) -> f64 {
    let m1 = result.reg1_m + result.reg2_m;
    let m2 = result.oft1_m + result.oft2_m;

    let a = up.cons_1(m1).min(0.0);
    let b = up.cons_2(m2, params.e_cf).min(0.0);
    let c = up.cons_3(m2).min(0.0);

    -1e100 * (a + b + c)
}

/// Run the multi-stage game simulation
pub fn start_game(
    iter_range: usize,
    e_d: f64,
    e_p: f64,
    e_v: f64,
    e_dp: f64,
    e_pv: f64,
    e_cf: f64,
    sigma_cf: f64,
    sigma_d: f64,
    sigma_p: f64,
    sigma_v: f64,
    sigma_dp: f64,
    sigma_pv: f64,
    e_p2: f64,
    e_v2: f64,
    e_p2v: f64,
    e_pv2: f64,
    e_p2v2: f64,
) -> [f64; 20] {
    let params = GameParams {
        e_d,
        e_p,
        e_v,
        e_dp,
        e_pv,
        e_cf,
        sigma_cf,
        sigma_d,
        sigma_p,
        sigma_v,
        sigma_dp,
        sigma_pv,
        e_p2,
        e_v2,
        e_p2v,
        e_pv2,
        e_p2v2,
    };

    // initialize lower player
    let reg1 = StageOnePlayer::new(500.0, 1e10, 0.4);
    let reg2 = StageOnePlayer::new(500.0, 1e10, 0.6);
    let oft1 = StageTwoPlayer::new(500.0, 1e10, 0.3);
    let oft2 = StageTwoPlayer::new(500.0, 1e10, 0.7);

    // initialize upper player
    let ga = crate::ga::GA::new(600, 500, 0.2);

    let p_range = [(0.0, 10000.0), (0.0, 1.0), (0.0, 10000.0), (0.0, 10000.0)];
    let m_range = [(-50.0, 50.0), (-0.5, 0.5), (-500.0, 500.0), (-500.0, 500.0)];

    let obj_func = |x: &[f64]| {
        let mut reg1_eval = StageOnePlayer::new(reg1.m, reg1.b, reg1.lbd);
        let mut reg2_eval = StageOnePlayer::new(reg2.m, reg2.b, reg2.lbd);
        let mut oft1_eval = StageTwoPlayer::new(oft1.m, oft1.b, oft1.gma);
        let mut oft2_eval = StageTwoPlayer::new(oft2.m, oft2.b, oft2.gma);

        let res = game(
            iter_range,
            x[0],
            x[1],
            x[2],
            x[3],
            &params,
            &mut reg1_eval,
            &mut reg2_eval,
            &mut oft1_eval,
            &mut oft2_eval,
        );

        let up_eval = UpstreamPlayer::new(x[0], x[1], x[2], x[3], 2, 2);
        let p_penalty = penalty_function(&up_eval, &params, &res);

        -res.f + p_penalty
    };

    let (x, ga_result) = ga.run(obj_func, None::<fn(&[f64]) -> f64>, &p_range, &m_range);

    let final_result = (|| {
        let mut reg1_final = StageOnePlayer::new(reg1.m, reg1.b, reg1.lbd);
        let mut reg2_final = StageOnePlayer::new(reg2.m, reg2.b, reg2.lbd);
        let mut oft1_final = StageTwoPlayer::new(oft1.m, oft1.b, oft1.gma);
        let mut oft2_final = StageTwoPlayer::new(oft2.m, oft2.b, oft2.gma);

        game(
            iter_range,
            x[0],
            x[1],
            x[2],
            x[3],
            &params,
            &mut reg1_final,
            &mut reg2_final,
            &mut oft1_final,
            &mut oft2_final,
        )
    })();

    let up_final = UpstreamPlayer::new(x[0], x[1], x[2], x[3], 2, 2);
    let m1_final = final_result.reg1_m + final_result.reg2_m;
    let m2_final = final_result.oft1_m + final_result.oft2_m;

    let cons_1 = up_final.cons_1(m1_final);
    let cons_2 = up_final.cons_2(m2_final, params.e_cf);
    let cons_3 = up_final.cons_3(m2_final);

    let theta1 = reg1.theta(
        final_result.reg1_m,
        &up_final,
        &params,
        m2_final,
        &[final_result.reg2_m],
    );
    let theta2 = reg2.theta(
        final_result.reg2_m,
        &up_final,
        &params,
        m2_final,
        &[final_result.reg1_m],
    );
    let mu1 = oft1.mu(
        final_result.oft1_m,
        &up_final,
        &params,
        &[final_result.oft2_m],
    );
    let mu2 = oft2.mu(
        final_result.oft2_m,
        &up_final,
        &params,
        &[final_result.oft1_m],
    );

    [
        final_result.reg1_m,
        reg1.lbd,
        theta1,
        final_result.reg2_m,
        reg2.lbd,
        theta2,
        final_result.oft1_m,
        oft1.gma,
        mu1,
        final_result.oft2_m,
        oft2.gma,
        mu2,
        x[0],
        x[1],
        x[2],
        x[3],
        cons_1,
        cons_2,
        cons_3,
        -ga_result,
    ]
}

fn linear_budget_upper_bound(price: f64, budget: f64, lb: f64, ub: f64) -> f64 {
    if price <= 1e-12 {
        return ub;
    }

    ((budget - C_T) / price).clamp(lb, ub)
}

fn game(
    iter_range: usize,
    q: f64,
    r: f64,
    p1: f64,
    p2: f64,
    params: &GameParams,
    reg1: &mut StageOnePlayer,
    reg2: &mut StageOnePlayer,
    oft1: &mut StageTwoPlayer,
    oft2: &mut StageTwoPlayer,
) -> ResultStruct {
    let up = UpstreamPlayer {
        q,
        r,
        p1,
        p2,
        n_re: 2,
        n_of: 2,
    };

    const MAX_M: f64 = 1e10;
    const INNER_TOL: f64 = 1e-2;
    const INNER_MAX_ITER: usize = 100;
    const GAME_CONVERGENCE_EPS: f64 = 1e-6;

    // Low level game 2: Stage Two players (oft1, oft2)
    for _ in 0..iter_range {
        let prev_oft1 = oft1.m;
        let prev_oft2 = oft2.m;
        let prev_reg1 = reg1.m;
        let prev_reg2 = reg2.m;

        let oft1_ub = linear_budget_upper_bound(up.p2, oft1.b, 0.0, MAX_M);
        let oft2_ub = linear_budget_upper_bound(up.p2, oft2.b, 0.0, MAX_M);

        // oft1 optimizes against oft2 (fixed)
        let m21_new = alter_best_response(
            |x| -oft1.mu(x, &up, params, &[oft2.m]),
            oft1.m,
            0.0,
            oft1_ub,
            INNER_TOL,
            INNER_MAX_ITER,
        );

        // oft2 optimizes against oft1 (fixed)
        let m22_new = alter_best_response(
            |x| -oft2.mu(x, &up, params, &[oft1.m]),
            oft2.m,
            0.0,
            oft2_ub,
            INNER_TOL,
            INNER_MAX_ITER,
        );

        oft1.m = m21_new;
        oft2.m = m22_new;

        // Low level game 1: Stage One players (reg1, reg2)
        let m2 = oft1.m + oft2.m;
        let reg1_ub = linear_budget_upper_bound(up.p1, reg1.b, 0.0, MAX_M);
        let reg2_ub = linear_budget_upper_bound(up.p1, reg2.b, 0.0, MAX_M);

        // reg1 optimizes against reg2 (fixed)
        let m11_new = alter_best_response(
            |x| -reg1.theta(x, &up, params, m2, &[reg2.m]),
            reg1.m,
            0.0,
            reg1_ub,
            INNER_TOL,
            INNER_MAX_ITER,
        );

        // reg2 optimizes against reg1 (fixed)
        let m12_new = alter_best_response(
            |x| -reg2.theta(x, &up, params, m2, &[reg1.m]),
            reg2.m,
            0.0,
            reg2_ub,
            INNER_TOL,
            INNER_MAX_ITER,
        );

        reg1.m = m11_new;
        reg2.m = m12_new;

        let max_delta = (oft1.m - prev_oft1)
            .abs()
            .max((oft2.m - prev_oft2).abs())
            .max((reg1.m - prev_reg1).abs())
            .max((reg2.m - prev_reg2).abs());

        if max_delta < GAME_CONVERGENCE_EPS {
            break;
        }
    }

    let m1 = reg1.m + reg2.m;
    let m2 = oft1.m + oft2.m;

    let f = up.pi_ess(params, m1, m2);

    ResultStruct {
        f,
        reg1_m: reg1.m,
        reg2_m: reg2.m,
        oft1_m: oft1.m,
        oft2_m: oft2.m,
    }
}
