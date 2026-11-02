"""Turn the raw simulation output (raw/*.json) into the CSV files of this repository.

    python export_dataset.py

Writes E1_load.csv, E2_attack_rate.csv, E3_security_reservation.csv, E4_ablation.csv,
E5_optimality.csv, scenarios_hosts.csv, scenarios_tasks.csv and results.json.
"""
import csv
import json
import math
import os
from collections import defaultdict

from model import Instance, TIERS, CLASSES

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'raw')
CLASS_COLS = ['iiot_share', 'v2x_share', 'ar_share', 'ehealth_share']
CAUSES = ['ok', 'security', 'dependability', 'performance', 'policy', 'rejected']
SCHEMES = ['TRIO', 'SEQ', 'PERF+SEC', 'PERF+DEP', 'PERF']


def load(name):
    p = os.path.join(RAW, name + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def f4(x):
    return '' if x is None else f'{x:.4f}'


def outcome_cols(r):
    cols = [f4(r['sat'])] + [f4(x) for x in r['cls_sat']]
    cols += [f"{r['reps']:.3f}", f"{r.get('exposure', 0.0):.3e}", f4(r['util']), f"{r['time']:.3f}"]
    cols += [r['causes'][k] for k in CAUSES]
    return cols


OUTCOME_HDR = (['met_target_share'] + CLASS_COLS +
               ['replicas_per_task', 'exposure', 'mean_utilisation', 'runtime_s'] +
               ['n_met', 'n_security_loss', 'n_dependability_loss', 'n_performance_loss',
                'n_policy_violation', 'n_rejected'])


def write(name, header, rows):
    with open(os.path.join(HERE, name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'{name}: {len(rows)} rows')


def mean(xs):
    return sum(xs) / len(xs)


def pct(xs):
    """Mean of shares in percent, rounded like the paper (numpy mean, one decimal)."""
    import numpy as np
    return float(f'{100 * np.mean(xs):.1f}')


def best_margin(rows, key):
    """Per key value and scheme, keep the safety margin with the highest mean (TRIO uses 1)."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(key(r), r['scheme'])][r['margin']].append(r['sat'])
    out = {}
    for (k, s), d in by.items():
        m = max(d, key=lambda mm: mean(d[mm]))
        out[(k, s)] = dict(margin=m, mean=pct(d[m]), runs=len(d[m]))
    return out


summary = {}

# ---------------------------------------------------------------- E1 load
main = load('main')
if main:
    write('E1_load.csv', ['scheme', 'safety_margin', 'users', 'seed'] + OUTCOME_HDR,
          [[r['scheme'], r['margin'], r['N'], r['seed']] + outcome_cols(r) for r in main])
    b = best_margin(main, lambda r: r['N'])
    summary['E1_load'] = {str(N): {s: b[(N, s)] for s in SCHEMES if (N, s) in b}
                          for N in sorted({r['N'] for r in main})}

# ---------------------------------------------------------------- E2 attack rate
thr = load('threat')
if thr:
    write('E2_attack_rate.csv', ['scheme', 'safety_margin', 'users', 'seed', 'attack_rate_per_h'] + OUTCOME_HDR,
          [[r['scheme'], r['margin'], r['N'], r['seed'], r['params']['lam_a']] + outcome_cols(r) for r in thr])
    b = best_margin(thr, lambda r: r['params']['lam_a'])
    summary['E2_attack_rate'] = {f'{la:g}': {s: b[(la, s)] for s in SCHEMES if (la, s) in b}
                                 for la in sorted({r['params']['lam_a'] for r in thr})}

# ---------------------------------------------------------------- E3 security reservation
rho = load('rho')
if rho:
    write('E3_security_reservation.csv',
          ['scheme', 'safety_margin', 'users', 'seed', 'reserved_share_rho'] + OUTCOME_HDR,
          [[r['scheme'], r['margin'], r['N'], r['seed'], r['params']['rho']] + outcome_cols(r) for r in rho])
    b = best_margin(rho, lambda r: (r['N'], r['params']['rho']))
    summary['E3_security_reservation'] = {
        f'{N} users': {f'rho={x:g}': b[((N, x), 'TRIO')]['mean']
                       for x in sorted({r['params']['rho'] for r in rho}) if ((N, x), 'TRIO') in b}
        for N in sorted({r['N'] for r in rho})}

# ---------------------------------------------------------------- E4 ablation
abl = load('ablation')
if abl:
    write('E4_ablation.csv',
          ['variant', 'users', 'seed', 'met_target_share'] + CLASS_COLS + ['replicas_per_task', 'runtime_s'],
          [[r['variant'], r['N'], r['seed'], f4(r['sat'])] + [f4(x) for x in r['cls_sat']] +
           [f"{r['reps']:.3f}", f"{r['time']:.3f}"] for r in abl])
    by = defaultdict(list)
    for r in abl:
        by[(r['variant'], r['N'])].append(r['sat'])
    summary["E4_ablation"] = {f"{v} ({N} users)": pct(x) for (v, N), x in by.items()}

# ---------------------------------------------------------------- E5 optimality
ilp = load('ilp')
if ilp:
    rows = []
    for r in ilp:
        hosts = 5 if r['layout'] == 'small' else 12
        bound = min(r['N'], math.floor(r['ilp_bound'] + 1e-6)) if r.get('ilp_bound') is not None else ''
        rows.append([hosts, r['N'], r['seed'], r['trio_sat'], r['ilp_sat'], bound,
                     'optimal' if r['ilp_status'] == 0 else 'time limit', 'full' if not r['prune'] else 'restricted',
                     f"{r['trio_time']:.3f}", f"{r['ilp_time']:.1f}", r.get('nvar', '')])
    write('E5_optimality.csv',
          ['hosts', 'users', 'seed', 'trio_tasks_met', 'ilp_tasks_met', 'ilp_upper_bound', 'ilp_status',
           'ilp_configuration_set', 'trio_runtime_s', 'ilp_runtime_s', 'ilp_variables'], rows)
    by = defaultdict(list)
    for r in rows:
        by[(r[0], r[1])].append((r[4] - r[3]) / max(r[4], 1))
    summary['E5_optimality_gap_percent'] = {f'{h} hosts, {N} users': pct(g)
                                            for (h, N), g in sorted(by.items())}

# ---------------------------------------------------------------- runtime
rt = load('runtime')
if rt:
    by = defaultdict(list)
    for r in rt:
        by[r['N']].append(r['time'])
    summary['trio_runtime_s'] = {str(N): round(mean(t), 2) for N, t in sorted(by.items())}

# ---------------------------------------------------------------- scenarios
hosts_rows, task_rows = [], []
for N in [50, 100, 150, 200, 250, 300]:
    for s in range(15):
        inst = Instance(N, seed=s)
        for h in range(inst.M):
            T = TIERS[inst.tier[h]]
            hosts_rows.append([N, s, h, inst.tier[h], int(inst.dom[h]), int(inst.s[h]), f'{inst.cap[h]:.0f}',
                               T['mttf'], T['mttr'], T['d_mttf'], T['d_mttr'], T['expo'],
                               f'{inst.pU[h]:.6f}', f'{inst.pC[h]:.6e}', f'{inst.pF[h]:.6e}'])
        for u in range(inst.N):
            c = int(inst.cls[u])
            task_rows.append([N, s, u, CLASSES[c]['name'], int(inst.gnb[u]), f'{inst.R[u] / 1e6:.2f}',
                              f'{inst.D[u] / 1e3:.0f}', f'{inst.W[u]:.0f}', f'{inst.lam[u]:g}',
                              f'{inst.tau[u] * 1e3:g}', int(inst.r[u]), inst.alpha[u]])
write('scenarios_hosts.csv',
      ['users', 'seed', 'host_id', 'tier', 'site_id', 'isolation_class', 'capacity_cycles_per_s', 'host_mttf_h',
       'host_mttr_h', 'site_mttf_h', 'site_mttr_h', 'exposure', 'p_up', 'p_compromised', 'p_failed'], hosts_rows)
write('scenarios_tasks.csv',
      ['users', 'seed', 'task_id', 'task_class', 'gnb', 'uplink_mbps', 'data_kbit', 'cycles_per_request',
       'request_rate_per_s', 'deadline_ms', 'min_isolation_class', 'tsa_target'], task_rows)

json.dump(summary, open(os.path.join(HERE, 'results.json'), 'w'), indent=2)
print('results.json written')
