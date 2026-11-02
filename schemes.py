"""Placement schemes on a common engine + exact ILP benchmark.

All heuristics share one engine (constrained-first ordering, minimum-reservation
configuration choice, ejection-chain repair). They differ ONLY in the internal
model used to judge a configuration, mirroring the literature classes in the survey:
  PERF      deadline only, single replica                (performance-only works)
  PERF+DEP  deadline + failures, replication, attack-blind (latency/reliability works)
  PERF+SEC  deadline + attacks + isolation policy, 1 replica (security-aware offloading)
  SEQ       isolation policy, then failure-driven replication blind to exposure
  TRIO      joint TSA model (proposed)
Every placement is scored afterwards under the TRUE joint model (model.evaluate).
"""
import time
import numpy as np
from model import ConfigSpace, tsa_vec, evaluate

MODELS = {
    'PERF':     dict(policy=False, attacks=False, failures=False, K=1),
    'PERF+DEP': dict(policy=False, attacks=False, failures=True,  K=None),
    'PERF+SEC': dict(policy=True,  attacks=True,  failures=False, K=1),
    'SEQ':      dict(policy=True,  attacks=False, failures=True,  K=None),
    'TRIO':     dict(policy=True,  attacks=True,  failures=True,  K=None),
}
_CS = {}


def config_space(inst):
    key = (inst.M, inst.p['K'], len(inst.Q))
    if key not in _CS:
        _CS[key] = ConfigSpace(*key)
    return _CS[key]


class Engine:
    def __init__(self, inst, cfg, order='constrained', repair=False, seed=0,
                 fixed_level=None, besteffort_level=1, max_targets=8, margin=1.0, prices=None,
                 defer_be=True):
        self.inst = inst
        self.defer_be = defer_be
        self.prices = np.ones(inst.M) if prices is None else np.asarray(prices, float)
        self.margin = margin
        self.alpha_int = 1.0 - margin * (1.0 - inst.alpha)   # internal design target
        self.cfg = cfg
        self.cs = cs = config_space(inst)
        self.Keff = cfg['K'] or inst.p['K']
        self.order_mode = order
        self.repair_on = repair
        self.rng = np.random.default_rng(seed)
        self.be_level = besteffort_level
        self.max_targets = max_targets
        tc = inst.__dict__.setdefault('_tsacache', {})
        tkey = (cfg['attacks'], cfg['failures'])
        if tkey not in tc:
            tc[tkey] = tsa_vec(inst, cs, np.arange(len(cs.H)), cfg['attacks'], cfg['failures'])
        self.itsa_all = tc[tkey]
        Hc = np.where(cs.H >= 0, cs.H, 0)
        self.Hc = Hc
        self.valid_mask = cs.H >= 0
        base = cs.size <= self.Keff
        self._fixed = fixed_level
        if fixed_level is not None:
            base &= np.all((cs.L == fixed_level) | ~self.valid_mask, axis=1)
        self.base = base
        self.U = np.zeros(inst.M)
        self.place = [([], []) for _ in range(inst.N)]
        self.cfg_of = np.full(inst.N, -1)
        self.on = [set() for _ in range(inst.M)]
        self.n_eval = 0

    # per-task static data -------------------------------------------------
    def task_data(self, u):
        inst, cs = self.inst, self.cs
        key = (self.cfg['policy'], self.Keff, self._fixed)
        cache = inst.__dict__.setdefault('_tdcache', {}).setdefault(key, {})
        if u not in cache:
            ok = self.base.copy()
            pol = inst.policy_ok[u] if self.cfg['policy'] else np.ones(inst.M, bool)
            hostok = pol & inst.deadline_ok[u]
            ok &= np.all(hostok[self.Hc] | ~self.valid_mask, axis=1)
            idx = np.nonzero(ok)[0]
            sh = inst.share[u][self.Hc[idx], cs.L[idx]]          # (n, K)
            sh = np.where(self.valid_mask[idx], sh, 0.0)
            cache[u] = (idx, sh, self.Hc[idx])
        idx, sh, hc = cache[u]
        cost = (sh * self.prices[hc]).sum(1)
        return idx, sh, cost, self.itsa_all[idx]

    def choose(self, u, td, R, exclude=None, must_satisfy=False):
        """Pick a configuration for u given residual capacity R."""
        idx, sh, cost, it = td
        self.n_eval += len(idx)
        Hc = self.Hc[idx]
        fits = np.all(sh <= R[Hc] + 1e-12, axis=1)
        if exclude is not None:
            fits &= ~np.any((Hc == exclude) & self.valid_mask[idx], axis=1)
        good = fits & (it >= self.alpha_int[u])
        if good.any():
            cand = np.nonzero(good)[0]
            j = cand[np.lexsort((-it[cand], cost[cand]))[0]]
            return idx[j]
        if must_satisfy:
            return -1
        be = fits & (self.cs.size[idx] == 1) & (self.cs.L[idx, 0] <= self.be_level)
        if be.any():
            cand = np.nonzero(be)[0]
            j = cand[np.lexsort((cost[cand], -it[cand]))[0]]
            return idx[j]
        return -1

    def _apply(self, u, c, sign):
        if c < 0:
            return
        for k in range(self.cs.K):
            h = self.cs.H[c, k]
            if h >= 0:
                self.U[h] += sign * self.inst.share[u, h, self.cs.L[c, k]]
                if sign > 0:
                    self.on[h].add(u)
                else:
                    self.on[h].discard(u)

    def assign(self, u, c):
        self._apply(u, self.cfg_of[u], -1)
        self.cfg_of[u] = c
        self._apply(u, c, +1)

    def isat(self, u):
        c = self.cfg_of[u]
        return c >= 0 and self.itsa_all[c] >= self.alpha_int[u]

    def run(self):
        inst = self.inst
        umax = inst.p['u_max']
        tds = [self.task_data(u) for u in range(inst.N)]
        if self.order_mode == 'constrained':
            ai = self.alpha_int
            nines = -np.log10(1 - ai)
            n_good = np.array([np.sum(td[3] >= ai[u]) for u, td in enumerate(tds)])
            min_cost = np.array([td[2][td[3] >= ai[u]].min() if np.any(td[3] >= ai[u]) else 9e9
                                 for u, td in enumerate(tds)])
            order = sorted(range(inst.N), key=lambda u: (-nines[u], n_good[u], -min_cost[u], self.rng.random()))
        elif self.order_mode == 'cheapest':
            # order by UNPRICED reservation: prices steer configuration choice, not priority
            ai = self.alpha_int
            min_cost = np.array([td[1].sum(1)[td[3] >= ai[u]].min() if np.any(td[3] >= ai[u]) else 9e9
                                 for u, td in enumerate(tds)])
            order = sorted(range(inst.N), key=lambda u: (min_cost[u], self.rng.random()))
        else:
            order = list(self.rng.permutation(inst.N))
        # ---- Phase 1: minimum-reservation placement of satisfying configurations ----
        for u in order:
            c = self.choose(u, tds[u], umax - self.U, must_satisfy=self.defer_be)
            self.assign(u, c)
        # ---- Phase 1b: best-effort placement of the remaining tasks in leftover capacity ----
        if self.defer_be:
            for u in order:
                if self.cfg_of[u] < 0:
                    self.assign(u, self.choose(u, tds[u], umax - self.U))
        # ---- Phase 2: ejection-chain repair ----
        if self.repair_on:
            for _ in range(3):
                gained = 0
                for u in order:
                    if self.isat(u):
                        continue
                    if self._repair(u, tds):
                        gained += 1
                if gained == 0:
                    break
        cs = self.cs
        for u in range(inst.N):
            c = self.cfg_of[u]
            if c < 0:
                self.place[u] = ([], [])
            else:
                m = cs.H[c] >= 0
                self.place[u] = (list(cs.H[c][m]), list(cs.L[c][m]))
        return self.place

    def _repair(self, u, tds):
        inst, cs = self.inst, self.cs
        umax = inst.p['u_max']
        idx, sh, cost, it = tds[u]
        good = np.nonzero(it >= self.alpha_int[u])[0]
        if len(good) == 0:
            return False
        targets = good[np.argsort(cost[good])][:self.max_targets]
        old_c = self.cfg_of[u]
        for j in targets:
            snap_U = self.U.copy(); snap_cfg = self.cfg_of.copy(); snap_on = [set(x) for x in self.on]
            self.assign(u, idx[j])                   # tentatively over-commit
            ok = True
            hosts = [h for h in cs.H[idx[j]] if h >= 0]
            for h in hosts:
                if self.U[h] <= umax + 1e-12:
                    continue
                victims = [v for v in self.on[h] if v != u and self.isat(v)]
                victims.sort(key=lambda v: -inst.share[v, h, cs.L[self.cfg_of[v]][list(cs.H[self.cfg_of[v]]).index(h)]])
                for v in victims:
                    if self.U[h] <= umax + 1e-12:
                        break
                    cv = self.cfg_of[v]
                    self.assign(v, -1)
                    R = umax - self.U
                    R[R < 0] = 0.0
                    nv = self.choose(v, tds[v], R, exclude=h, must_satisfy=True)
                    if nv < 0:
                        self.assign(v, cv)
                        continue
                    self.assign(v, nv)
                if self.U[h] > umax + 1e-12:
                    ok = False
                    break
            if ok and np.all(self.U <= umax + 1e-9):
                return True
            self.U = snap_U; self.cfg_of = snap_cfg; self.on = snap_on
        return False


def run_scheme(inst, name, seed=0, **kw):
    return Engine(inst, MODELS[name], seed=seed, **kw).run()


def trio(inst, seed=0, rounds=4, xi=2.0, repair=False, return_info=False, model='TRIO',
         orders=('constrained', 'cheapest'), **kw):
    """Algorithm 1: price refinement run for each task ordering; the best placement is kept."""
    if 'order' in kw:
        orders = (kw.pop('order'),)
    best, best_sat, hist = None, -1, []
    for o in orders:
        pl, info = _trio_one(inst, seed=seed, rounds=rounds, xi=xi, repair=repair, model=model, order=o, **kw)
        hist += info['rounds']
        if info['best'] > best_sat:
            best, best_sat = pl, info['best']
    if return_info:
        return best, dict(rounds=hist, best=best_sat)
    return best


def _trio_one(inst, seed=0, rounds=4, xi=2.0, repair=False, model='TRIO', **kw):
    """Price-refined placement (Algorithm 1). With model='TRIO' this is the proposed
    scheme; with a baseline model it gives that baseline the same algorithm.
    Each round re-runs the placement with prices raised on over-demanded hosts
    (demand includes what unsatisfied tasks would need); the best round is kept."""
    cfg = MODELS[model]
    prices = np.ones(inst.M)
    best, best_sat, hist = None, -1, []
    for r in range(rounds):
        eng = Engine(inst, cfg, seed=seed, prices=prices, repair=repair and r == rounds - 1, **kw)
        pl = eng.run()
        nsat = sum(eng.isat(u) for u in range(inst.N))
        hist.append(nsat)
        if nsat > best_sat:
            best, best_sat, best_prices = pl, nsat, prices.copy()
        # demand of unsatisfied tasks at their cheapest satisfying configuration
        demand = eng.U.copy()
        for u in range(inst.N):
            if eng.isat(u):
                continue
            idx, sh, cost, it = eng.task_data(u)
            good = np.nonzero(it >= eng.alpha_int[u])[0]
            if len(good) == 0:
                continue
            j = good[np.argmin(cost[good])]
            for k in range(eng.cs.K):
                h = eng.cs.H[idx[j], k]
                if h >= 0:
                    demand[h] += sh[j, k]
        over = np.maximum(demand / inst.p['u_max'] - 1.0, 0.0)
        prices = prices * np.exp(xi * over)
        prices /= prices.min()
    if repair and rounds > 1:
        eng = Engine(inst, cfg, seed=seed, prices=best_prices, repair=True, **kw)
        pl = eng.run(); nsat = sum(eng.isat(u) for u in range(inst.N))
        if nsat >= best_sat:
            best, best_sat = pl, nsat
    return best, dict(rounds=hist, best=best_sat)


# ---------------------------------------------------------------- exact ILP
def ilp_optimum(inst, kappa=1e-3, prune=False, time_limit=120, K=None):
    """Exact multiple-choice multidimensional-knapsack ILP over all
    policy/deadline-feasible configurations under the TRUE model.
    max sum_u sum_c x_uc (1[TSA_c>=alpha_u] + kappa*TSA_c)
    s.t. sum_c x_uc <= 1 ; sum_{u,c} x_uc share_{u,c,h} <= u_max  (all h)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix
    cs = config_space(inst)
    Kmax = K or inst.p['K']
    tsa_all = tsa_vec(inst, cs, np.arange(len(cs.H)))
    Hc = np.where(cs.H >= 0, cs.H, 0)
    vm = cs.H >= 0
    cols_u, cols_c, vals = [], [], []
    for u in range(inst.N):
        hostok = inst.policy_ok[u] & inst.deadline_ok[u]
        ok = (cs.size <= Kmax) & np.all(hostok[Hc] | ~vm, axis=1)
        idx = np.nonzero(ok)[0]
        if prune and len(idx):
            # keep, per host set, the cheapest satisfying level vector and the best TSA one
            sh = np.where(vm[idx], inst.share[u][Hc[idx], cs.L[idx]], 0).sum(1)
            sat = tsa_all[idx] >= inst.alpha[u]
            keyset = {}
            for j, c in enumerate(idx):
                hs = tuple(cs.H[c][vm[c]])
                k = (hs, bool(sat[j]))
                if k not in keyset or sh[j] < keyset[k][1]:
                    keyset[k] = (c, sh[j])
            idx = np.array(sorted(v[0] for v in keyset.values()))
        for c in idx:
            cols_u.append(u); cols_c.append(c)
            vals.append((1.0 if tsa_all[c] >= inst.alpha[u] else 0.0) + kappa * tsa_all[c])
    nv = len(cols_u)
    cols_u = np.array(cols_u); cols_c = np.array(cols_c); vals = np.array(vals)
    # assignment constraints
    A1 = coo_matrix((np.ones(nv), (cols_u, np.arange(nv))), shape=(inst.N, nv))
    # capacity constraints
    r, cidx, data = [], [], []
    for k in range(cs.K):
        h = cs.H[cols_c, k]
        m = h >= 0
        s = inst.share[cols_u[m], h[m], cs.L[cols_c[m], k]]
        r += list(h[m]); cidx += list(np.nonzero(m)[0]); data += list(s)
    A2 = coo_matrix((data, (r, cidx)), shape=(inst.M, nv))
    cons = [LinearConstraint(A1, 0, 1), LinearConstraint(A2, 0, inst.p['u_max'])]
    t0 = time.time()
    res = milp(-vals, constraints=cons, integrality=np.ones(nv), bounds=Bounds(0, 1),
               options=dict(time_limit=time_limit, mip_rel_gap=1e-6, disp=False))
    dt = time.time() - t0
    place = [([], []) for _ in range(inst.N)]
    if res.x is None:
        return None, dict(status=res.status, time=dt)
    x = np.round(res.x).astype(int)
    for j in np.nonzero(x)[0]:
        c = cols_c[j]; m = cs.H[c] >= 0
        place[cols_u[j]] = (list(cs.H[c][m]), list(cs.L[c][m]))
    info = dict(status=res.status, time=dt, obj=-res.fun, nvar=nv,
                bound=-getattr(res, 'mip_dual_bound', res.fun) if getattr(res, 'mip_dual_bound', None) is not None else None,
                gap=getattr(res, 'mip_gap', None))
    return place, info
