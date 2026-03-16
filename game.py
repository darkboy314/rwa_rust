import numpy as np


class Game:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            # Set attributes dynamically
            setattr(self, key, value)

    # static variable
    # q, r, p1, p2 which is init value for search

    T = 10  # Lifespan years

    q = 100  # q: Storage capacity. Unit: MWh. ref: Baotang BESS in Guangdong, Foshan
    r = 0.2  # r: Share of profit amoung generators, constant from 0 to 1
    p1 = 500  # p1: Token price in stage 1. Unit: $
    p2 = 500  # p2: Token price in stage 2. Unit: $

    c_t = 100.0  # Transaction cost. Unit: $
    k = 584000  # Development cost per unit of storage capacity. Unit: $/MWh
    # k *= 4/7  # For REVB k' = 4/7 * k
    f = 1700  # Price that per unit of power offtaker sales to customer. Unit: $/MWh. Ref: 2026 CLP base price + FCC = 1736

    N_RE = 0  # Number of renewable energy generators
    N_OF = 0  # Number of offtakers


# class for emulating two player gaming in stage 1
class StageOneGame(Game):
    M1: float

    def __init__(self, m, lbd_w, b, **kwargs):
        super().__init__(**kwargs)
        self.m = np.float64(m)
        self.lbd_w = np.float64(lbd_w)
        self.b = np.float64(b)
        Game.N_RE += 1

    def __del__(self):
        Game.N_RE -= 1

    def theta(self, m, M2, *m_all):
        T = Game.T
        q = Game.q
        r = Game.r
        p1 = Game.p1
        p2 = Game.p2
        E_cf = self.E_cf
        sigma_cf = self.sigma_cf
        c_t = Game.c_t
        lbd_w = self.lbd_w
        E_DP = self.E_DP
        M1 = self.M1(*m_all) + 1e-10  # Avoid divided by zero
        return (
            T * E_DP
            + (m / M1) * r * (p2 * M2 - T * E_cf * q - lbd_w * T * (sigma_cf**2) * q)
            - p1 * m
            - c_t
        )

    @classmethod
    def M1(cls, *m):
        return np.sum(m)

    def cons(self, m1):
        b = self.b
        p1 = Game.p1
        c_t = Game.c_t
        return b - c_t - p1 * m1


# class for emulating two player gaming in stage 2
class StageTwoGame(Game):
    M2: float

    def __init__(self, m, gma_j, b, **kwargs):
        super().__init__(**kwargs)
        self.m = np.float64(m)
        self.gma_j = np.float64(gma_j)
        self.b = np.float64(b)
        Game.N_OF += 1

    def __del__(self):
        Game.N_OF -= 1

    def mu(self, m, *m_all):
        f = Game.f
        T = Game.T
        q = Game.q
        p2 = Game.p2
        c_t = Game.c_t
        gma_j = self.gma_j
        E_V = self.E_V
        E_P = self.E_P
        E_PV = self.E_PV
        sigma_V = self.sigma_V
        sigma_P = self.sigma_P
        sigma_PV = self.sigma_PV

        M2 = self.M2(*m_all) + 1e-10  # Avoid divided by zero
        return (
            T * f * E_V
            + T * (m / M2) * q * E_P
            - T * E_PV
            - gma_j * T * (f * sigma_V) ** 2
            - gma_j * T * ((m / M2) * q * sigma_P) ** 2
            + gma_j * T * sigma_PV**2
            - p2 * m
            - c_t
        )

    @classmethod
    def M2(cls, *m):
        return np.sum(m)

    def cons(self, m2):
        p2 = Game.p2
        c_t = Game.c_t
        b = self.b
        return b - c_t - p2 * m2


class UpstreamPlayer(Game):
    def __init__(self, q, r, p1, p2, **kwargs):
        super().__init__(**kwargs)
        self.q = q
        self.r = r
        self.p1 = p1
        self.p2 = p2
        return

    def pi_ess(
        self,
        q,
        r,
        p1,
        p2,
        M1,
        M2,
    ):
        E_cf = self.E_cf
        N_RE = Game.N_RE
        N_OF = Game.N_OF
        c_t = Game.c_t
        k = Game.k
        T = Game.T
        return (
            p1 * M1 + (1 - r) * (p2 * M2 - T * q * E_cf) - (N_RE + N_OF) * c_t - k * q
        )

    def cons_1(self, q, p1, M1):
        N_re = Game.N_RE
        c_t = Game.c_t
        k = Game.k
        return p1 * M1 - N_re * c_t - k * q

    def cons_2(
        self,
        q,
        r,
        p2,
        M2,
    ):
        N_of = Game.N_OF
        T = Game.T
        E_cf = self.E_cf
        c_t = Game.c_t
        return (1 - r) * (p2 * M2 - T * E_cf * q) - c_t * N_of

    def cons_3(self, q, p2, M2):
        f = Game.f
        return f * q - p2 * M2

    def cons_4(self, r):
        return 1 - r

    def cons_5(self, q, p2, M2):
        T = Game.T
        E_cf = self.E_cf
        return p2 * M2 - T * E_cf * q
