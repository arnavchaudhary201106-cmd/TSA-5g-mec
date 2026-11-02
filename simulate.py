"""Run the experiments and store raw results as JSON in raw/ (one file per experiment).

    python simulate.py                 # everything (about 1 h on 2 cores)
    python simulate.py load attack     # only the experiments used in the paper
    python simulate.py --quick load    # 2 seeds per point, for a fast smoke test

Experiments: load (E1), attack (E2), reservation (E3), ablation (E4), optimality (E5), runtime.
Afterwards run export_dataset.py to turn raw/*.json into the E*.csv files.
"""
import json, sys, time, os
import numpy as np
from multiprocessing import Pool
from model import Instance, evaluate, CLASSES
from schemes import MODELS, trio, ilp_optimum, Engine

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')
os.makedirs(OUT, exist_ok=True)
MARGINS = [1.0, 0.5, 0.2]


def summarize(inst, ev, extra=None):
    ok = np.array([c not in ('policy', 'rejected') for c in ev['cause']])
    exposure = []
    d = dict(
        sat=float(ev['sat'].mean()),
        cls_sat=[float(ev['sat'][inst.cls == c].mean()) if np.any(inst.cls == c) else None for c in range(len(CLASSES))],
        reps=float(ev['reps'][ev['reps'] > 0].mean()) if np.any(ev['reps'] > 0) else 0.0,
        Ls=float(ev['Ls'][ok].mean()) if ok.any() else None,
        Ld=float(ev['Ld'][ok].mean()) if ok.any() else None,
        Lp=float(ev['Lp'][ok].mean()) if ok.any() else None,
        causes={k: int(np.sum(ev['cause'] == k)) for k in ['ok', 'policy', 'rejected', 'security', 'dependability', 'performance']},
        util=float(ev['U'].mean()),
        tsa_mean=float(ev['tsa'].mean()),
    )
    if extra:
        d.update(extra)
    return d


def run_all_schemes(N, seed, params=None, margins=MARGINS, schemes=None):
    inst = Instance(N, seed=seed, params=params)
    out = []
    for name in (schemes or MODELS):
        for m in ([1.0] if name == 'TRIO' else margins):
            t = time.time()
            pl = trio(inst, model=name, margin=m)
            dt = time.time() - t
            ev = evaluate(inst, pl)
            expo = [sum(inst.pC[h] for h in S) for (S, q) in pl if S]
            out.append(summarize(inst, ev, dict(scheme=name, margin=m, N=N, seed=seed, time=dt,
                                                exposure=float(np.mean(expo)) if expo else 0.0,
                                                params=params or {})))
    return out


def job(args):
    kind = args[0]
    if kind == 'main':
        _, N, seed = args
        return run_all_schemes(N, seed)
    if kind == 'threat':
        _, lam_a, seed = args
        return run_all_schemes(200, seed, params=dict(lam_a=lam_a))
    if kind == 'rho':
        _, rho, N, seed = args
        return run_all_schemes(N, seed, params=dict(rho=rho), schemes=['TRIO', 'SEQ'])
    if kind == 'ablation':
        _, N, seed = args
        inst = Instance(N, seed=seed)
        res = []
        variants = {
            'TRIO': dict(),
            'most-constrained-first order only': dict(orders=('constrained',), rounds=8),
            'cheapest-first order only': dict(orders=('cheapest',), rounds=8),
            'no price refinement': dict(rounds=1),
            'single replica (K=1)': dict(K1=True),
            'fixed latency level (q=3)': dict(fixed_level=2),
        }
        for vname, kw in variants.items():
            kw = dict(kw)
            t = time.time()
            if kw.pop('K1', False):
                import schemes as S
                cfg = dict(S.MODELS['TRIO']); cfg['K'] = 1
                S.MODELS['_TRIO_K1'] = cfg
                pl = trio(inst, model='_TRIO_K1', **kw)
            else:
                pl = trio(inst, **kw)
            dt = time.time() - t
            ev = evaluate(inst, pl)
            res.append(summarize(inst, ev, dict(variant=vname, N=N, seed=seed, time=dt)))
        return res
    if kind == 'ilp':
        _, N, seed, layout, prune, tl = args
        inst = Instance(N, seed=seed, layout=layout)
        t = time.time(); pl = trio(inst); tt = time.time() - t
        ev = evaluate(inst, pl)
        pl2, info = ilp_optimum(inst, prune=prune, time_limit=tl)
        ev2 = evaluate(inst, pl2)
        return [dict(N=N, seed=seed, layout=layout, prune=prune,
                     trio_sat=int(ev['sat'].sum()), trio_time=tt,
                     ilp_sat=int(ev2['sat'].sum()), ilp_time=info['time'], ilp_status=int(info['status']),
                     ilp_obj=info.get('obj'), ilp_bound=info.get('bound'), ilp_gap=info.get('gap'),
                     nvar=info.get('nvar'))]
    if kind == 'runtime':
        _, N, seed = args
        inst = Instance(N, seed=seed)
        t = time.time(); pl = trio(inst); dt = time.time() - t
        return [dict(N=N, seed=seed, time=dt, sat=float(evaluate(inst, pl)['sat'].mean()))]
    raise ValueError(kind)


def run(name, jobs, procs=2):
    t = time.time()
    res = []
    with Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            res += r
            if (i + 1) % 5 == 0 or i + 1 == len(jobs):
                print(f'[{name}] {i + 1}/{len(jobs)} jobs  {time.time() - t:.0f}s', flush=True)
                json.dump(res, open(os.path.join(OUT, f'{name}.json'), 'w'))
    json.dump(res, open(os.path.join(OUT, f'{name}.json'), 'w'))


if __name__ == '__main__':
    args = sys.argv[1:]
    quick = '--quick' in args
    which = [a for a in args if not a.startswith('--')] or ['load', 'attack', 'reservation', 'ablation', 'optimality', 'runtime']
    S15, S10, S5 = (range(2),) * 3 if quick else (range(15), range(10), range(5))
    if 'load' in which:
        run('main', [('main', N, s) for s in S15 for N in [50, 100, 150, 200, 250, 300]])
    if 'attack' in which:
        run('threat', [('threat', la, s) for s in S10 for la in [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04]])
    if 'reservation' in which:
        run('rho', [('rho', r, N, s) for s in S10 for N in [150, 250] for r in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]])
    if 'ablation' in which:
        run('ablation', [('ablation', N, s) for s in S10 for N in [200, 300]])
    if 'optimality' in which:
        jobs = [('ilp', N, s, 'small', False, 120) for s in S5 for N in [40, 60, 90, 120]]
        jobs += [('ilp', N, s, 'full', True, 60) for s in S5 for N in [100, 150, 200, 250, 300]]
        run('ilp', jobs, procs=2)
    if 'runtime' in which:
        run('runtime', [('runtime', N, s) for s in range(3) for N in [100, 200, 400, 800]], procs=1)
