import os
import asyncio
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from scipy.stats import gamma, lognorm
from ga import ga
from datetime import datetime
import game
import matplotlib.pyplot as plt

# for REVB
# k' = 4/7 k
# c_f' = 1.5 * c_f
# sigma_cf' = 2 * sigma_cf


def operation_cost_gamma(num, e, var, seed=None):
    """
    从γ分布抽样生成运营成本

    参数
    ----
    num : int
        需要的样本数量
    e : float
        期望值
    var : float
        方差
    seed : int or None
        随机种子，便于复现。

    返回
    ----
    sample : ndarray, shape(n_samples, 1)
        返回样本列表
    """
    rng = np.random.default_rng(seed)

    theta = var / e
    k = e / theta

    return rng.gamma(k, theta, num)


def sample_bivariate_lognormal(n_samples, mean_log, cov_log, random_state=None):
    """
    从二维对数正态联合分布抽样。

    参数
    ----
    n_samples : int
        需要的样本数量。
    mean_log : array-like, shape (2,)
        对应 ln(Y1), ln(Y2) 的均值向量 μ。
    cov_log : array-like, shape (2, 2)
        对应 ln(Y1), ln(Y2) 的协方差矩阵 Σ。
    random_state : int or None
        随机种子，便于复现。

    返回
    ----
    samples : ndarray, shape (n_samples, 2)
        每一行是一个样本 [y1, y2]。
    """
    rng = np.random.default_rng(random_state)

    # 1. 先在二维正态 N(mean_log, cov_log) 下采样
    x = rng.multivariate_normal(mean=mean_log, cov=cov_log, size=n_samples)

    # 2. 指数变换得到对数正态
    y = np.exp(x)

    return y


def lognormal_params_from_mean_var(mean, var):
    """
    从对数正态分布的均值和方差反推底层正态分布的参数
    μ 和 σ。

    设 X ~ LogNormal(μ, σ^2)，则
        mean = exp(μ + σ^2/2)
        var  = (exp(σ^2) - 1) * exp(2μ + σ^2)

    通过已知 mean 和 var，可以求解：
        σ^2 = ln(1 + var / mean^2)
        μ   = ln(mean) - σ^2 / 2

    参数
    ----
    mean : float
        对数正态分布的期望值 E[X]
    var : float
        对数正态分布的方差 Var[X]

    返回
    ----
    mu : float
        底层正态分布的均值 μ
    sigma : float
        底层正态分布的标准差 σ
    """
    # 先计算方差的对数形式
    sigma2 = np.log(1 + var / (mean**2))
    mu = np.log(mean) - sigma2 / 2
    sigma = np.sqrt(sigma2)
    return mu, sigma


def alter_best_response(func, cons, x0, lb=-np.inf, ub=np.inf, *args):
    """
    在给定 y_fixed 下，求玩家1的带约束最佳相应：
    """
    bounds = Bounds(lb=lb, ub=ub)
    cons = {"type": "ineq", "fun": cons}

    res = minimize(
        fun=func,
        x0=x0,
        args=args,
        method="COBYQA",
        bounds=bounds,
        constraints=cons,
    )

    return res.x[0], -res.fun, res


# @profile
# Main solver of the whole game
def start_game(iter_range, **kwargs):
    # init players
    reg1 = game.StageOneGame(m=500, lbd_w=0.4, b=1e10, **kwargs)
    reg2 = game.StageOneGame(m=500, lbd_w=0.6, b=1e10, **kwargs)

    oft1 = game.StageTwoGame(m=500, gma_j=0.3, b=1e10, **kwargs)
    oft2 = game.StageTwoGame(m=500, gma_j=0.7, b=1e10, **kwargs)

    up = game.UpstreamPlayer(
        game.Game.q, game.Game.r, game.Game.p1, game.Game.p2, **kwargs
    )

    # Low level game 2
    for k in range(iter_range):
        m21_new, m21_fun, m21_res = alter_best_response(
            func=lambda x: -oft1.mu(*x, *x, oft2.m),
            cons=oft1.cons,
            x0=oft1.m,
            lb=0,
            ub=np.inf,
        )

        m22_new, m22_fun, m22_res = alter_best_response(
            func=lambda x: -oft2.mu(*x, *x, oft1.m),
            cons=oft2.cons,
            x0=oft2.m,
            lb=0,
            ub=np.inf,
        )

        oft1.m, oft2.m = m21_new, m22_new

    M2 = game.StageTwoGame.M2(oft1.m, oft2.m)

    # low level game 1
    for k in range(iter_range):
        m11_new, m11_fun, m11_res = alter_best_response(
            func=lambda x: -reg1.theta(x, M2, *x, reg2.m),
            cons=reg1.cons,
            x0=reg1.m,
            lb=0,
            ub=np.inf,
        )

        m12_new, m12_fun, m12_res = alter_best_response(
            func=lambda x: -reg2.theta(x, M2, *x, reg1.m),
            cons=reg2.cons,
            x0=reg2.m,
            lb=0,
            ub=np.inf,
        )

        reg1.m, reg2.m = m11_new, m12_new

    M1 = game.StageOneGame.M1(reg1.m, reg2.m)

    # penalty funcion for theta
    def p_func(*x):
        q = x[0]
        r = x[1]
        p1 = x[2]
        p2 = x[3]

        return -1e100 * (
            np.minimum(0, up.cons_1(q, p1, M1))
            + np.minimum(0, up.cons_2(q, r, p2, M2))
            + np.minimum(0, up.cons_3(q, p2, M2))
            + np.minimum(0, up.cons_4(r))
            + np.minimum(0, up.cons_5(q, p2, M2))
        )

    # upper level game
    x, result = ga.run(
        obj_func=lambda *x: up.pi_ess(*x, M1, M2),
        p_func=p_func,
        pop_size=500,
        gen_n=200,
        p_range=[(0, 10000), (0, 1), (0, 10000), (0, 10000)],
        m_range=[(-5.0, 5.0), (-0.5, 0.5), (-5.0, 5.0), (-5.0, 5.0)],
    )

    return x, reg1.m, reg2.m, oft1.m, oft2.m, result


# Output data to csv file (async version)
async def output(data, header, filename: str = "file.csv"):
    output_path = "./output"
    filepath = output_path + "/" + filename
    df = pd.DataFrame([data])

    # Check if file exists to decide whether to write header
    file_exists = os.path.isfile(filepath)
    # Run the I/O operation in a thread pool to avoid blocking
    await asyncio.to_thread(
        df.to_csv,
        filepath,
        mode="a",  # append mode
        header=(
            header if not file_exists else False
        ),  # only write header if file doesn't exist
        index=False,
    )


def main():
    time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    filename = f"{time}/result.csv"
    os.makedirs(f"./output/{time}/figure-1")
    os.makedirs(f"./output/{time}/figure-2")
    header = [
        "E_D",
        "E_P",
        "E_V",
        "E_DP",
        "E_PV",
        "E_cf",
        "sigma_D",
        "sigma_P",
        "sigma_V",
        "sigma_DP",
        "sigma_PV",
        "sigma_cf",
        "m11",
        "m12",
        "m21",
        "m22",
        "q",
        "r",
        "p1",
        "p2",
        "fun",
    ]

    # Market power demand & supply
    # preset value for generating data by lognormal distribution
    # Unit: MWh and $/MWh
    # Price reference: 140.6 cents/kWh CLP = 1406 $/MWh
    E_D = 30
    E_P = 1400
    E_V = 50

    var_D = 3
    var_P = 0.5
    var_V = 2

    mu_D, sigma_D = lognormal_params_from_mean_var(E_D, var_D)
    mu_P, sigma_P = lognormal_params_from_mean_var(E_P, var_P)
    mu_V, sigma_V = lognormal_params_from_mean_var(E_V, var_V)

    rho_DP = 0.5
    rho_PV = 0.3
    rho_DV = 0.2

    E_cf = 580  # Expected value of operating cost c_f. 2.5% of development cost
    # E_cf *= 1.5  # For REVB E_cf' = 1.5 * E_cf
    var_cf = 10  # variance of operating cost c_f
    # var_cf *= 2  # For REVB Var_cf' = 2 * Var_cf

    mean = np.array([mu_D, mu_P, mu_V])
    cov = np.array(
        [
            [sigma_D**2, rho_DP * sigma_D * sigma_P, rho_DV * sigma_D * sigma_V],
            [rho_DP * sigma_D * sigma_P, sigma_P**2, rho_PV * sigma_P * sigma_V],
            [rho_DV * sigma_D * sigma_V, rho_PV * sigma_P * sigma_V, sigma_V**2],
        ]
    )

    for _ in range(100):
        cf = operation_cost_gamma(60, E_cf, var_cf, 123)
        dpv = sample_bivariate_lognormal(60, mean, cov, 123).T

        mean_d = np.mean(dpv[0])
        mean_p = np.mean(dpv[1])
        mean_v = np.mean(dpv[2])
        mean_dp = np.mean(dpv[0] * dpv[1])
        mean_pv = np.mean(dpv[1] * dpv[2])
        s_d = np.std(dpv[0], ddof=1)
        s_p = np.std(dpv[1], ddof=1)
        s_v = np.std(dpv[2], ddof=1)
        s_dp = np.std(dpv[0] * dpv[1], ddof=1)
        s_pv = np.std(dpv[1] * dpv[2], ddof=1)

        mean_cf = np.mean(cf)
        s_cf = np.std(cf)

        # replace population mean and standard deviation with sample ones
        values = {
            "E_D": mean_d,
            "E_P": mean_p,
            "E_V": mean_v,
            "E_DP": mean_dp,
            "E_PV": mean_pv,
            "E_cf": mean_cf,
            "sigma_D": s_d,
            "sigma_P": s_p,
            "sigma_V": s_v,
            "sigma_DP": s_dp,
            "sigma_PV": s_pv,
            "sigma_cf": s_cf,
        }

        # Figure 1 for Pm
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot()
        ax1.hist(dpv[1], bins=10, alpha=0.7, label="Sample")
        ax1.set_xlabel("x")
        ax1.set_ylabel("Count", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")

        ax2 = ax1.twinx()
        x_lognorm = np.linspace(1300, 1500, 100)
        y_lognorm = lognorm.pdf(x_lognorm, s=sigma_P, scale=np.exp(mu_P))
        ax2.plot(
            x_lognorm,
            y_lognorm,
            label=f"Log-normal PDF (mu={mu_P}, σ={sigma_P})",
            color="C1",
        )
        ax2.set_ylabel("PDF", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")

        plt.title("Figure 1 - Distribution of Pm and sample")
        fig1.savefig(f"./output/{time}/figure-1/{_}.png")
        plt.clf()

        # Figure 2 for cf
        fig2 = plt.figure(2)
        ax3 = fig2.add_subplot()
        ax3.hist(cf, bins=10, alpha=0.7, label="Sample")
        ax3.set_xlabel("x")
        ax3.set_ylabel("Count", color="C0")
        ax3.tick_params(axis="y", labelcolor="C0")

        ax4 = ax3.twinx()
        x_gamma = np.linspace(550, 650, 100)
        y_gamma = gamma.pdf(x_gamma, a=(E_cf**2 / var_cf), scale=(var_cf / E_cf))
        ax4.plot(
            x_gamma,
            y_gamma,
            label=f"Gamma PDF (k={E_cf**2/var_cf}, θ={var_cf/E_cf})",
            color="C1",
        )
        ax4.set_ylabel("PDF", color="C1")
        ax4.tick_params(axis="y", labelcolor="C1")

        plt.title("Figure 2 - Distribution of cf and sample")
        fig2.savefig(f"./output/{time}/figure-2/{_}.png")
        plt.clf()

        x, m11, m12, m21, m22, res = start_game(iter_range=200, **values)
        data = list(values.values()) + [m11, m12, m21, m22] + list(x) + [res]

        asyncio.run(output(data, header, filename=filename))
        print(x, res)


if __name__ == "__main__":
    main()
