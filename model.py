"""Unified security-dependability-performance model for 5G MEC task placement.

Analytical ingredients
  * host CTMC with states Up / Compromised / Failed (attacks and faults in one model)
  * failure domains (site power / transport) shared by co-located hosts
  * per-replica reserved compute rate -> M/M/1 response time (exact tail)
  * joint KPI: TSA (Trustworthy Service Availability)
"""
import itertools
import numpy as np

LN10 = np.log(10.0)

# ---------------- isolation (security) classes ---------------------------
# s=1 shared-kernel container, s=2 sandboxed micro-VM, s=3 TEE-backed confidential VM
SEC_PI = {1: 1.00, 2: 0.30, 3: 0.05}      # relative attack-success factor
SEC_ETA = {1: 1.00, 2: 1.12, 3: 1.30}     # compute overhead multiplier
SEC_EPS = {1: 0.0, 2: 0.2e-3, 3: 0.6e-3}  # per-request fixed overhead [s]

# ---------------- infrastructure tiers -----------------------------------
TIERS = {
    #          cap [cyc/s], host MTTF/MTTR [h], domain MTTF/MTTR [h], exposure, P(s=1,2,3)
    'ran': dict(cap=48e9,  mttf=2000., mttr=2.0, d_mttf=4000.,  d_mttr=4.0, expo=1.5, sec_p=(0.5, 0.4, 0.1)),
    'co':  dict(cap=96e9,  mttf=4000., mttr=1.0, d_mttf=5000.,  d_mttr=2.5, expo=1.0, sec_p=(0.0, 0.5, 0.5)),
    'reg': dict(cap=192e9, mttf=8000., mttr=1.0, d_mttf=20000., d_mttr=1.0, expo=0.7, sec_p=(0.0, 0.3, 0.7)),
}
ONE_WAY = {'co': 1.0e-3, 'reg': 3.0e-3}   # gNB -> CO / regional DC [s]

# ---------------- service classes ----------------------------------------
#  D [bit], W [cycles/request], tau [s], r (min isolation), alpha (TSA target), lam [req/s], share
CLASSES = [
    dict(name='IIoT control',       D=16e3,  W=0.5e6, tau=5e-3,  r=2, alpha=0.999, lam=20., share=0.30),
    dict(name='V2X perception',     D=240e3, W=5e6,   tau=15e-3, r=1, alpha=0.999, lam=10., share=0.30),
    dict(name='AR offload',         D=600e3, W=12e6,  tau=25e-3, r=1, alpha=0.99,  lam=10., share=0.25),
    dict(name='e-Health analytics', D=160e3, W=10e6,  tau=30e-3, r=3, alpha=0.999, lam=5.,  share=0.15),
]

DEFAULTS = dict(
    lam_a=0.005,    # attack-attempt rate per unit exposure [1/h]
    gamma0=0.5,     # baseline detection + clean-restore rate [1/h]
    gamma1=40.0,    # detection-rate gain per unit of reserved security capacity [1/h]
    theta=0.2,      # compromised -> crashed (attacker-induced) rate [1/h]
    rho=0.10,       # share of host capacity reserved for security monitoring (IDS/VMI)
    u_max=1.0,      # max reserved share of effective capacity
    t_dl=0.1e-3,    # result download time [s]
    K=3,            # max replicas per task
    Q=(1, 2, 3, 4), # per-replica latency levels: P(late) = 10^-q
)


def host_ctmc(lam_f, mu, beta, gamma, theta):
    """Stationary probabilities (U, C, F) of the 3-state host CTMC.
    U->C beta (successful intrusion), U->F lam_f (fault), C->U gamma (detection +
    clean restore), C->F theta+lam_f (attacker-induced or spontaneous crash),
    F->U mu (repair / re-imaging)."""
    pc = beta / (gamma + theta + lam_f)
    pf = (lam_f + (theta + lam_f) * pc) / mu
    z = 1.0 + pc + pf
    return 1.0 / z, pc / z, pf / z


class Instance:
    """Random 5G-MEC scenario: 8 gNBs on a ring, 4 RAN-edge sites (2 MEHs each),
    one central-office site (2 MEHs) and one regional edge DC (2 MEHs)."""

    def __init__(self, n_users, seed, params=None, layout='full', class_mix=None):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        rng = np.random.default_rng(seed)
        hosts = []
        if layout == 'full':
            n_gnb = 8
            for i in range(4):
                hosts += [('ran', i, 2 * i)] * 2
            hosts += [('co', 4, None)] * 2
            hosts += [('reg', 5, None)] * 2
        elif layout == 'small':
            n_gnb = 4
            for i in range(2):
                hosts += [('ran', i, 2 * i)] * 2
            hosts += [('co', 2, None)]
        else:
            raise ValueError(layout)
        self.n_gnb = n_gnb
        self.M = M = len(hosts)
        self.tier = [h[0] for h in hosts]
        self.dom = np.array([h[1] for h in hosts])
        self.n_dom = int(self.dom.max() + 1)
        self.cap = np.array([TIERS[t]['cap'] for t in self.tier])
        self.s = np.array([rng.choice([1, 2, 3], p=TIERS[t]['sec_p']) for t in self.tier])
        if layout == 'small':
            self.s[:4] = rng.permutation([1, 2, 2, 3])
            self.s[4] = rng.choice([2, 3])
        self.rho = np.full(M, p['rho'])
        self.A_dom = np.zeros(self.n_dom)
        for h in range(M):
            T = TIERS[self.tier[h]]
            self.A_dom[self.dom[h]] = T['d_mttf'] / (T['d_mttf'] + T['d_mttr'])
        d = np.zeros((n_gnb, M))
        for h, (t, dm, g) in enumerate(hosts):
            for b in range(n_gnb):
                if t == 'ran':
                    hop = min(abs(b - g), n_gnb - abs(b - g))
                    d[b, h] = 0.1e-3 if hop == 0 else 0.2e-3 + 0.25e-3 * hop
                else:
                    d[b, h] = ONE_WAY[t]
        self.d = d
        self.eta = np.array([SEC_ETA[x] for x in self.s])
        self.eps = np.array([SEC_EPS[x] for x in self.s])
        # ---------- users / tasks ----------
        self.N = N = n_users
        shares = np.array([c['share'] for c in CLASSES]) if class_mix is None else np.asarray(class_mix, float)
        self.cls = rng.choice(len(CLASSES), size=N, p=shares / shares.sum())
        self.gnb = rng.integers(0, n_gnb, size=N)
        self.R = rng.uniform(60e6, 240e6, size=N)
        self.D = np.array([CLASSES[c]['D'] for c in self.cls])
        self.W = np.array([CLASSES[c]['W'] for c in self.cls])
        self.tau = np.array([CLASSES[c]['tau'] for c in self.cls])
        self.r = np.array([CLASSES[c]['r'] for c in self.cls])
        self.alpha = np.array([CLASSES[c]['alpha'] for c in self.cls])
        self.lam = np.array([CLASSES[c]['lam'] for c in self.cls])
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self):
        """(Re)compute everything that depends on rho / threat parameters."""
        p = self.p
        M = self.M
        self.ceff = self.cap * (1 - self.rho)
        self.pU = np.zeros(M); self.pC = np.zeros(M); self.pF = np.zeros(M)
        for h in range(M):
            T = TIERS[self.tier[h]]
            beta = p['lam_a'] * T['expo'] * SEC_PI[self.s[h]]
            gamma = p['gamma0'] + p['gamma1'] * self.rho[h]
            self.pU[h], self.pC[h], self.pF[h] = host_ctmc(1 / T['mttf'], 1 / T['mttr'], beta, gamma, p['theta'])
        self.Tfix = (self.D / self.R)[:, None] + 2 * self.d[self.gnb, :] + self.eps[None, :] + p['t_dl']
        self.slack = self.tau[:, None] - self.Tfix
        self.policy_ok = self.s[None, :] >= self.r[:, None]
        self.deadline_ok = self.slack > 0
        Q = np.array(p['Q'], float)
        self.Q = Q
        # reserved service rate for P(late) = 10^-q  (M/M/1 sojourn: mu - lam = q ln10 / slack)
        with np.errstate(divide='ignore', invalid='ignore'):
            mu = self.lam[:, None, None] + Q[None, None, :] * LN10 / self.slack[:, :, None]
            f = mu * (self.W[:, None, None] * self.eta[None, :, None])
            self.mu = np.where(self.deadline_ok[:, :, None], mu, np.inf)
            self.share = np.where(self.deadline_ok[:, :, None], f / self.ceff[None, :, None], np.inf)
        self.Fq = 1.0 - 10.0 ** (-Q)

    # ------------------------------------------------------------------
    def _host_terms(self, h, attacks, failures):
        if attacks and failures:
            return self.pC[h], self.pU[h]
        if failures:                 # attack-blind: compromised host counted as up
            return 0.0, self.pU[h] + self.pC[h]
        if attacks:                  # failure-blind
            return self.pC[h], 1.0 - self.pC[h]
        return 0.0, 1.0

    def tsa_parts(self, u, S, q, attacks=True, failures=True):
        """TSA and exact loss split 1-TSA = L_sec + L_dep + L_perf for task u with
        replicas on hosts S at latency levels q (indices into Q)."""
        if len(S) == 0:
            return 0.0, 0.0, 0.0, 1.0
        groups = {}
        for h, k in zip(S, q):
            groups.setdefault(int(self.dom[h]), []).append((h, k))
        X = Y = Y1 = 1.0
        for dmn, items in groups.items():
            A = self.A_dom[dmn] if failures else 1.0
            px = py = py1 = 1.0
            for h, k in items:
                c, pu = self._host_terms(h, attacks, failures)
                g = pu * self.Fq[k]
                px *= 1 - c; py *= 1 - c - g; py1 *= 1 - c - pu
            X *= (1 - A) + A * px
            Y *= (1 - A) + A * py
            Y1 *= (1 - A) + A * py1
        return X - Y, 1 - X, Y1, Y - Y1

    def tsa(self, u, S, q, attacks=True, failures=True):
        return self.tsa_parts(u, S, q, attacks, failures)[0]


# ---------------------------------------------------------------- configs
class ConfigSpace:
    """All replica configurations (distinct hosts, |S|<=K, one latency level each),
    pre-computed once per (M, K, |Q|) for vectorised evaluation."""

    def __init__(self, M, K, nQ):
        H, L = [], []
        for k in range(1, K + 1):
            for S in itertools.combinations(range(M), k):
                for lv in itertools.product(range(nQ), repeat=k):
                    H.append(list(S) + [-1] * (K - k))
                    L.append(list(lv) + [0] * (K - k))
        self.H = np.array(H)
        self.L = np.array(L)
        self.size = (self.H >= 0).sum(1)
        self.K = K


def tsa_vec(inst, cs, idx, attacks=True, failures=True):
    """Vectorised TSA for configurations cs[idx] (TSA does not depend on the task
    once latency levels are fixed; deadline feasibility is handled by the caller)."""
    H = cs.H[idx]; L = cs.L[idx]
    n = len(idx)
    if attacks and failures:
        c_h, pu_h = inst.pC, inst.pU
    elif failures:
        c_h, pu_h = np.zeros(inst.M), inst.pU + inst.pC
    elif attacks:
        c_h, pu_h = inst.pC, 1 - inst.pC
    else:
        c_h, pu_h = np.zeros(inst.M), np.ones(inst.M)
    A = inst.A_dom if failures else np.ones(inst.n_dom)
    px = np.ones((n, inst.n_dom)); py = np.ones((n, inst.n_dom))
    present = np.zeros((n, inst.n_dom), bool)
    rows = np.arange(n)
    for k in range(cs.K):
        hk = H[:, k]
        valid = hk >= 0
        hh = np.where(valid, hk, 0)
        dk = inst.dom[hh]
        c = np.where(valid, c_h[hh], 0.0)
        g = np.where(valid, pu_h[hh] * inst.Fq[L[:, k]], 0.0)
        px[rows, dk] *= 1 - c
        py[rows, dk] *= 1 - c - g
        present[rows, dk] |= valid
    fx = np.where(present, (1 - A) + A * px, 1.0)
    fy = np.where(present, (1 - A) + A * py, 1.0)
    return fx.prod(1) - fy.prod(1)


def evaluate(inst, place):
    """Score a placement under the TRUE joint model.
    place[u] = (hosts, levels); ([], []) means rejected."""
    N = inst.N
    tsa = np.zeros(N); Ls = np.zeros(N); Ld = np.zeros(N); Lp = np.zeros(N)
    cause = np.empty(N, dtype=object)
    U = np.zeros(inst.M)
    reps = np.zeros(N, int)
    for u, (S, q) in enumerate(place):
        reps[u] = len(S)
        for h, k in zip(S, q):
            U[h] += inst.share[u, h, k]
        if len(S) == 0:
            cause[u] = 'rejected'; Lp[u] = 1.0
            continue
        if any(not inst.policy_ok[u, h] for h in S):
            cause[u] = 'policy'
            continue
        t, a, b, c = inst.tsa_parts(u, S, q)
        tsa[u], Ls[u], Ld[u], Lp[u] = t, a, b, c
        cause[u] = 'ok' if t >= inst.alpha[u] else \
            ['security', 'dependability', 'performance'][int(np.argmax([a, b, c]))]
    sat = tsa >= inst.alpha
    return dict(tsa=tsa, sat=sat, Ls=Ls, Ld=Ld, Lp=Lp, cause=cause, U=U, reps=reps)
