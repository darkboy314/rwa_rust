/// Game theory module implementing the three-stage game
/// Stage 1: Upstream player (energy storage provider)
/// Stage 2: Stage One players (renewable energy generators)
/// Stage 3: Stage Two players (electricity offtakers)
use cobyla::{Func, RhoBeg, StopTols, minimize};

const T: f64 = 10.0; // Lifespan years
const Q: f64 = 200.0; // Storage capacity (MWh)
const R: f64 = 0.2; // Profit sharing ratio
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
    pub fn theta(
        &self,
        m: f64,
        up: &UpstreamPlayer,
        params: &GameParams,
        m2: f64,
        m_all: &[f64],
    ) -> f64 {
        let m1 = m_all.iter().sum::<f64>() + 1e-10;

        T * params.e_dp
            + (m / m1)
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
    pub fn mu(&self, m: f64, up: &UpstreamPlayer, params: &GameParams, m_all: &[f64]) -> f64 {
        let m2 = m_all.iter().sum::<f64>() + 1e-10;

        T * F * params.e_v + T * (m / m2) * up.q * params.e_p
            - T * params.e_pv
            - self.gma * T * (F * params.sigma_v).powi(2)
            - self.gma * T * ((m / m2) * up.q * params.sigma_p).powi(2)
            + self.gma * T * params.sigma_pv.powi(2)
            - up.p2 * m
            - C_T
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

/// Constrained 1-D optimization using COBYLA.
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
    let objective = |x: &[f64], _data: &mut ()| f(x[0]);
    let inequality = |x: &[f64], _data: &mut ()| constraint(x[0]);
    let constraints = [&inequality as &dyn Func<()>];
    let x_init = [x0];
    let bounds = [(lb, ub)];
    let rho = if lb.is_finite() && ub.is_finite() {
        ((ub - lb).abs() * 0.25).max(tol * 10.0).max(1e-4)
    } else {
        x0.abs().max(1.0).max(tol * 10.0)
    };
    let stop_tol = StopTols {
        xtol_abs: vec![tol.max(1e-9)],
        ftol_rel: tol.max(1e-9),
        ..StopTols::default()
    };

    match minimize(
        objective,
        &x_init,
        &bounds,
        &constraints,
        (),
        max_iter.max(1),
        RhoBeg::All(rho),
        Some(stop_tol),
    ) {
        Ok((_status, x_opt, _y_opt)) => x_opt[0],
        Err((_status, x_opt, _y_opt)) => x_opt[0],
    }
}

/// Penalty function for genetic algorithm
pub fn penalty_function(
    up: &UpstreamPlayer,
    params: &GameParams,
    x: &[f64],
    s1player: &[StageOnePlayer],
    s2player: &[StageTwoPlayer],
) -> f64 {
    let q = x[0];
    let r = x[1];
    let p1 = x[2];
    let p2 = x[3];

    let reg1 = &s1player[0];
    let reg2 = &s1player[1];

    let oft1 = &s2player[0];
    let oft2 = &s2player[1];

    let m1 = reg1.m + reg2.m;
    let m2 = oft1.m + oft2.m;

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
    // initialize lower player
    let mut reg1 = StageOnePlayer::new(500.0, 1e10, 0.4);
    let mut reg2 = StageOnePlayer::new(500.0, 1e10, 0.6);
    let mut oft1 = StageTwoPlayer::new(500.0, 1e10, 0.3);
    let mut oft2 = StageTwoPlayer::new(500.0, 1e10, 0.7);

    let player_list1 = [
        StageOnePlayer::new(reg1.m, reg1.b, reg1.lbd),
        StageOnePlayer::new(reg2.m, reg2.b, reg2.lbd),
    ];
    let player_list2 = [
        StageTwoPlayer::new(oft1.m, oft1.b, oft1.gma),
        StageTwoPlayer::new(oft2.m, oft2.b, oft2.gma),
    ];

    // initialize upper player
    let ga = crate::ga::GA::new(500, 40, 0.2);

    let p_range = [(0.0, 1000.0), (0.0, 1.0), (0.0, 10000000.0), (0.0, 10000000.0)];
    let m_range = [(-5.0, 5.0), (-0.5, 0.5), (-500.0, 500.0), (-500.0, 500.0)];

    let penalty_func = |x: &[f64]| {
        penalty_function(
            &UpstreamPlayer::new(x[0], x[1], x[2], x[3], 2, 2),
            &params,
            x,
            &player_list1,
            &player_list2,
        )
    };
    let obj_func = |x: &[f64]| {
        let mut reg1_eval = StageOnePlayer::new(reg1.m, reg1.b, reg1.lbd);
        let mut reg2_eval = StageOnePlayer::new(reg2.m, reg2.b, reg2.lbd);
        let mut oft1_eval = StageTwoPlayer::new(oft1.m, oft1.b, oft1.gma);
        let mut oft2_eval = StageTwoPlayer::new(oft2.m, oft2.b, oft2.gma);

        game(
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
        )
    };

    let (x, ga_result) = ga.run(obj_func, Some(penalty_func), &p_range, &m_range);

    (
        reg1.m,
        reg2.m,
        oft1.m,
        oft2.m,
        x[0],
        x[1],
        x[2],
        x[3],
        ga_result,
    )
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
) -> f64 {
    let up = UpstreamPlayer {
        q,
        r,
        p1,
        p2,
        n_re: 2,
        n_of: 2,
    };

    // Low level game 2: Stage Two players (oft1, oft2)
    for _ in 0..iter_range {
        // oft1 optimizes against oft2 (fixed)
        let m21_new = alter_best_response(
            |x| -oft1.mu(x, &up, params, &[x, oft2.m]),
            |x| oft1.constraint(&up, x),
            oft1.m,
            0.0,
            f64::INFINITY,
            1e-2,
            20,
        );

        // oft2 optimizes against oft1 (fixed)
        let m22_new = alter_best_response(
            |x| -oft2.mu(x, &up, params, &[x, oft1.m]),
            |x| oft2.constraint(&up, x),
            oft2.m,
            0.0,
            f64::INFINITY,
            1e-2,
            20,
        );

        oft1.m = m21_new;
        oft2.m = m22_new;

        // Low level game 1: Stage One players (reg1, reg2)
        let m2 = oft1.m + oft2.m;

        // reg1 optimizes against reg2 (fixed)
        let m11_new = alter_best_response(
            |x| -reg1.theta(x, &up, params, m2, &[x, reg2.m]),
            |x| reg1.constraint(&up, x),
            reg1.m,
            0.0,
            f64::INFINITY,
            1e-2,
            20,
        );

        // reg2 optimizes against reg1 (fixed)
        let m12_new = alter_best_response(
            |x| -reg2.theta(x, &up, params, m2, &[x, reg1.m]),
            |x| reg2.constraint(&up, x),
            reg2.m,
            0.0,
            f64::INFINITY,
            1e-2,
            20,
        );

        reg1.m = m11_new;
        reg2.m = m12_new;
    }

    let m1 = reg1.m + reg2.m;
    let m2 = oft1.m + oft2.m;

    up.pi_ess(params, m1, m2, &[q, r, p1, p2])
}
