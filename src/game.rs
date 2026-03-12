/// Game theory module implementing the three-stage game
/// Stage 1: Upstream player (energy storage provider)
/// Stage 2: Stage One players (renewable energy generators)  
/// Stage 3: Stage Two players (electricity offtakers)
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::brent::BrentOpt;

const T: f64 = 10.0; // Lifespan years
const Q: f64 = 600.0; // Storage capacity (MWh)
const C_T: f64 = 100.0; // Transaction cost ($)
const K: f64 = 584000.0; // Development cost per unit ($/MWh) 
const F: f64 = 1700.0; // Sales price per unit ($/MWh)

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
    pub fn theta(&self, up: &UpstreamPlayer, params: &GameParams, m2: f64, m_all: &[f64]) -> f64 {
        let m1 = m_all.iter().sum::<f64>() + 1e-10;

        T * params.e_dp
            + (self.m / m1)
                * up.r
                * (up.p2 * m2
                    - T * params.e_cf * up.q
                    - self.lbd * T * (params.sigma_cf * params.sigma_cf) * up.q)
            - up.p1 * self.m
            - C_T
    }

    /// Constraint function (budget constraint)
    pub fn constraint(&self, up: &UpstreamPlayer, m1: f64) -> f64 {
        self.b - C_T - up.p1 * m1
    }

    /// Market share calculation
    pub fn m1(m_all: &[f64]) -> f64 {
        m_all.iter().sum::<f64>()
    }
}

impl StageTwoPlayer {
    pub fn new(m: f64, b: f64, gma: f64) -> Self {
        StageTwoPlayer { m, b, gma }
    }

    /// Profit function for stage two player
    pub fn mu(&self, up: &UpstreamPlayer, params: &GameParams, m_all: &[f64]) -> f64 {
        let m2 = m_all.iter().sum::<f64>() + 1e-10;

        T * F * params.e_v + T * (self.m / m2) * up.q * params.e_p
            - T * params.e_pv
            - self.gma * T * (F * params.sigma_v).powi(2)
            - self.gma * T * ((self.m / m2) * up.q * params.sigma_p).powi(2)
            + self.gma * T * params.sigma_pv.powi(2)
            - up.p2 * self.m
            - C_T
    }

    /// Constraint function
    pub fn constraint(&self, up: &UpstreamPlayer, m2: f64) -> f64 {
        self.b - C_T - up.p2 * m2
    }

    /// Market share calculation
    pub fn m2(m_all: &[f64]) -> f64 {
        m_all.iter().sum::<f64>()
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
    pub fn pi_ess(&self, params: &GameParams, m1: f64, m2: f64, args: &[f64]) -> f64 {
        let q = args[0];
        let r = args[1];
        let p1 = args[2];
        let p2 = args[3];
        p1 * m1 + (1.0 - r) * (p2 * m2 - T * q * params.e_cf)
            - (self.n_re + self.n_of) as f64 * C_T
            - K * q
    }

    /// Constraint functions
    pub fn cons_1(&self, q: f64, p1: f64, m1: f64) -> f64 {
        p1 * m1 - self.n_re as f64 * C_T - K * q
    }

    pub fn cons_2(&self, q: f64, r: f64, p2: f64, m2: f64, e_cf: f64) -> f64 {
        (1.0 - r) * (p2 * m2 - T * e_cf * q) - C_T * self.n_of as f64
    }

    pub fn cons_3(&self, q: f64, p2: f64, m2: f64) -> f64 {
        F * q - p2 * m2
    }

    pub fn cons_4(&self, r: f64) -> f64 {
        1.0 - r
    }

    pub fn cons_5(&self, q: f64, p2: f64, m2: f64, e_cf: f64) -> f64 {
        p2 * m2 - T * e_cf * q
    }
}

struct ClosureFunc<F>(F);

impl<F> CostFunction for ClosureFunc<F>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    type Param = f64;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
        Ok(self.0(*x))
    }
}

/// Constrained 1‑D optimization using `argmin`.
/// now leverages GoldenSectionSearch with explicit bounds.
pub fn alter_best_response<F, C>(
    f: F,
    constraint: C,
    x0: f64,
    lb: f64,
    ub: f64,
    tol: f64,
    max_iter: usize,
) -> f64
where
    F: Fn(f64) -> f64 + Send + Sync,
    C: Fn(f64) -> f64 + Send + Sync,
{
    let problem = ClosureFunc(f);
    let solver = BrentOpt::new(lb, ub);

    let res = Executor::new(problem, solver)
        .configure(|state| state.max_iters(max_iter.try_into().unwrap()).param(x0))
        .run()
        .expect("Optimization Failed");

    *res.state().get_best_param().unwrap()
}

/// Penalty function for genetic algorithm
pub fn penalty_function(
    up: &UpstreamPlayer,
    params: &GameParams,
    x: &[f64],
    m1: f64,
    m2: f64,
) -> f64 {
    let q = x[0];
    let r = x[1];
    let p1 = x[2];
    let p2 = x[3];

    -1e100
        * (up.cons_1(q, p1, m1).min(0.0)
            + up.cons_2(q, r, p2, m2, params.e_cf).min(0.0)
            + up.cons_3(q, p2, m2).min(0.0)
            + up.cons_4(r).min(0.0)
            + up.cons_5(q, p2, m2, params.e_cf).min(0.0))
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
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
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
    };

    // Initialize players
    let mut reg1 = StageOnePlayer::new(500.0, 100000.0, 0.4);
    let mut reg2 = StageOnePlayer::new(500.0, 100000.0, 0.6);
    let mut oft1 = StageTwoPlayer::new(500.0, 100000.0, 0.3);
    let mut oft2 = StageTwoPlayer::new(500.0, 100000.0, 0.7);

    let up = UpstreamPlayer::new(Q, 0.2, 1000.0, 1000.0, 2, 2);

    // Low level game 2: Stage Two players (oft1, oft2)
    for _ in 0..iter_range {
        // oft1 optimizes against oft2 (fixed)
        let m21_new = alter_best_response(
            |x| -oft1.mu(&up, &params, &[x, oft2.m]), // Negative because we minimize
            |x| oft1.constraint(&up, x),
            oft1.m,
            0.0,
            10000.0,
            1e-6,
            100,
        );

        // oft2 optimizes against oft1 (fixed)
        let m22_new = alter_best_response(
            |x| -oft2.mu(&up, &params, &[oft1.m, x]), // Negative because we minimize
            |x| oft2.constraint(&up, x),
            oft2.m,
            0.0,
            10000.0,
            1e-6,
            100,
        );

        oft1.m = m21_new;
        oft2.m = m22_new;
    }

    // Low level game 1: Stage One players (reg1, reg2)
    for _ in 0..iter_range {
        let m2 = StageTwoPlayer::m2(&[oft1.m, oft2.m]);

        // reg1 optimizes against reg2 (fixed)
        let m11_new = alter_best_response(
            |x| -reg1.theta(&up, &params, m2, &[x, reg2.m]), // Negative because we minimize
            |x| reg1.constraint(&up, x),
            reg1.m,
            0.0,
            10000.0,
            1e-6,
            100,
        );

        // reg2 optimizes against reg1 (fixed)
        let m12_new = alter_best_response(
            |x| -reg2.theta(&up, &params, m2, &[reg1.m, x]), // Negative because we minimize
            |x| reg2.constraint(&up, x),
            reg2.m,
            0.0,
            10000.0,
            1e-6,
            100,
        );

        reg1.m = m11_new;
        reg2.m = m12_new;
    }

    let m1 = StageOnePlayer::m1(&[reg1.m, reg2.m]);
    let m2 = StageTwoPlayer::m2(&[oft1.m, oft2.m]);

    // Upper level game: Use genetic algorithm to optimize upstream player parameters
    let ga = crate::ga::GA::new(500, 500, 0.1);

    let p_range = vec![
        (0.0, 100000.0),
        (0.0, 1.0),
        (0.0, 100000.0),
        (0.0, 100000.0),
    ];
    let m_range = vec![(-5.0, 5.0), (-0.5, 0.5), (-5.0, 5.0), (-5.0, 5.0)];

    let penalty_func = |x: &[f64]| penalty_function(&up, &params, x, m1, m2);
    let obj_func = |x: &[f64]| up.pi_ess(&params, m1, m2, x);

    let (x, result) = ga.run(obj_func, Some(penalty_func), &p_range, &m_range);

    // Check if solution is valid (no NaN)
    if x.iter().any(|&val| val.is_nan()) {
        return (
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        );
    }

    (
        reg1.m, reg2.m, oft1.m, oft2.m, x[0], x[1], x[2], x[3], result,
    )
}
