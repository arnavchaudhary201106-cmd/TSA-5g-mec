"""Draw the two result figures of the paper from E1_load.csv and E2_attack_rate.csv.

    python plot.py      # writes figures/load.pdf and figures/attack_rate.pdf

For every point, each baseline is shown at the safety margin that gave it the best mean.
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')
SCHEMES = ['TRIO', 'SEQ', 'PERF+SEC', 'PERF+DEP', 'PERF']
MARKERS = ['o', 's', '^', 'D', 'v']


def best_curves(path, xcol):
    runs = defaultdict(list)
    for r in csv.DictReader(open(path)):
        runs[(r['scheme'], float(r[xcol]), r['safety_margin'])].append(float(r['met_target_share']))
    xs = sorted({k[1] for k in runs})
    curves = {}
    for s in SCHEMES:
        ys = []
        for x in xs:
            means = [sum(v) / len(v) for (sc, xx, m), v in runs.items() if sc == s and xx == x]
            ys.append(100 * max(means))
        curves[s] = ys
    return xs, curves


def draw(xs, curves, xlabel, name, categorical=False):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    pos = list(range(len(xs))) if categorical else xs
    for s, m in zip(SCHEMES, MARKERS):
        ax.plot(pos, curves[s], marker=m, label=s)
    if categorical:
        ax.set_xticks(pos)
        ax.set_xticklabels([f'{x:g}' for x in xs])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Tasks meeting target (%)')
    ax.grid(True)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    print('wrote figures/' + name)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    xs, c = best_curves(os.path.join(HERE, 'E1_load.csv'), 'users')
    draw(xs, c, 'Number of users', 'load.pdf')
    xs, c = best_curves(os.path.join(HERE, 'E2_attack_rate.csv'), 'attack_rate_per_h')
    draw(xs, c, 'Attack rate $\\lambda_a$ (1/h)', 'attack_rate.pdf', categorical=True)
