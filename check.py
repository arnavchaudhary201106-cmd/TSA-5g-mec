"""Check the closed-form TSA model against a simulation that does not use its shortcuts.

    python check.py            # one 150-user TRIO placement, about 30 s

Prints the ratio between simulated and analytical TSA loss (1 - TSA) per task and the
late probability of single replicas for different service-time distributions.

* Each replica is simulated as a FCFS single-server queue with Poisson arrivals and
  its reserved service rate (Lindley recursion, vectorised). Service times are
  exponential (model assumption), deterministic or log-normal (robustness).
* Host and failure-domain states are simulated as continuous-time trajectories of
  the CTMCs over a long horizon; requests arrive at random instants and see the
  current states (no stationarity shortcut).
"""
import numpy as np

from model import Instance, evaluate, TIERS, SEC_PI
from schemes import trio


def lindley_sojourn(lam, mu, n, rng, dist='exp', cv=1.0):
    """Sojourn times of n customers in a FCFS queue: Poisson(lam) arrivals,
    mean service 1/mu with the given distribution."""
    ia = rng.exponential(1.0 / lam, n)
    if dist == 'exp':
        s = rng.exponential(1.0 / mu, n)
    elif dist == 'det':
        s = np.full(n, 1.0 / mu)
    elif dist == 'lognorm':
        sig2 = np.log(1 + cv ** 2)
        s = rng.lognormal(np.log(1.0 / mu) - sig2 / 2, np.sqrt(sig2), n)
    else:
        raise ValueError(dist)
    # W_{k+1} = max(0, W_k + S_k - A_{k+1});  W_k = Z_k - min_{j<=k} Z_j
    z = np.concatenate(([0.0], np.cumsum(s[:-1] - ia[1:])))
    w = z - np.minimum.accumulate(z)
    soj = w + s
    return soj[n // 20:]          # drop warm-up


def ctmc_trajectory(rates, horizon, rng, start=0):
    """Simulate a CTMC given rates[i] = {j: rate}. Returns (jump_times, states)."""
    t, s = 0.0, start
    T, S = [0.0], [s]
    while t < horizon:
        out = rates[s]
        tot = sum(out.values())
        t += rng.exponential(1.0 / tot)
        r = rng.random() * tot
        acc = 0.0
        for j, q in out.items():
            acc += q
            if r <= acc:
                s = j
                break
        T.append(t); S.append(s)
    return np.array(T), np.array(S)


def state_at(traj, times):
    T, S = traj
    return S[np.searchsorted(T, times, side='right') - 1]


def simulate_tsa(inst, place, horizon_h=2e5, n_req=200000, n_q=400000, seed=0, dist='exp', cv=1.0):
    """Monte Carlo estimate of TSA for every placed task.
    horizon_h: simulated hours of host/domain dynamics; n_req: sampled requests per task."""
    rng = np.random.default_rng(seed)
    p = inst.p
    # --- domain and host state trajectories (time in hours) ---
    dom_traj = {}
    for dmn in range(inst.n_dom):
        h0 = int(np.nonzero(inst.dom == dmn)[0][0])
        T = TIERS[inst.tier[h0]]
        dom_traj[dmn] = ctmc_trajectory({0: {1: 1 / T['d_mttf']}, 1: {0: 1 / T['d_mttr']}}, horizon_h, rng)
    host_traj = {}
    for h in range(inst.M):
        T = TIERS[inst.tier[h]]
        lf, mu = 1 / T['mttf'], 1 / T['mttr']
        beta = p['lam_a'] * T['expo'] * SEC_PI[inst.s[h]]
        gamma = p['gamma0'] + p['gamma1'] * inst.rho[h]
        # 0=U, 1=C, 2=F
        rates = {0: {1: beta, 2: lf}, 1: {0: gamma, 2: p['theta'] + lf}, 2: {0: mu}}
        host_traj[h] = ctmc_trajectory(rates, horizon_h, rng)
    est = np.full(inst.N, np.nan)
    for u, (S, q) in enumerate(place):
        if not S or any(not inst.policy_ok[u, h] for h in S):
            continue
        times = rng.uniform(0, horizon_h, n_req)
        any_comp = np.zeros(n_req, bool)
        any_good = np.zeros(n_req, bool)
        for h, k in zip(S, q):
            reach = state_at(dom_traj[int(inst.dom[h])], times) == 0
            st = state_at(host_traj[h], times)
            soj = lindley_sojourn(inst.lam[u], inst.mu[u, h, k], n_q, rng, dist, cv)
            rt = inst.Tfix[u, h] + rng.choice(soj, n_req)
            any_comp |= reach & (st == 1)
            any_good |= reach & (st == 0) & (rt <= inst.tau[u])
        est[u] = np.mean(~any_comp & any_good)
    return est


def replica_late_prob(inst, u, h, k, n=400000, seed=0, dist='exp', cv=1.0):
    rng = np.random.default_rng(seed)
    soj = lindley_sojourn(inst.lam[u], inst.mu[u, h, k], n, rng, dist, cv)
    return np.mean(soj > inst.slack[u, h])


if __name__ == '__main__':
    inst = Instance(150, seed=7)
    place = trio(inst)
    ev = evaluate(inst, place)
    sim = simulate_tsa(inst, place, horizon_h=2e5, n_req=200000, n_q=200000, seed=11)
    ok = ~np.isnan(sim)
    ratio = (1 - sim[ok]) / (1 - ev['tsa'][ok])
    q25, q50, q75 = np.percentile(ratio, [25, 50, 75])
    print(f'tasks checked: {ok.sum()}')
    print(f'simulated / analytical (1 - TSA): median {q50:.3f}, IQR {q25:.2f}-{q75:.2f}')
    pairs = [(u, h, k) for u, (S, q) in enumerate(place) for h, k in zip(S, q)][:40]
    for dist, cv, name in [('exp', 1.0, 'exponential (model)'), ('det', 0.0, 'deterministic'),
                           ('lognorm', 0.5, 'log-normal, cv 0.5'), ('lognorm', 1.5, 'log-normal, cv 1.5')]:
        r = [replica_late_prob(inst, u, h, k, n=200000, seed=u, dist=dist, cv=cv) / 10 ** -inst.Q[k]
             for (u, h, k) in pairs]
        print(f'late probability / design value, {name:22s}: mean {np.mean(r):.3f}')
