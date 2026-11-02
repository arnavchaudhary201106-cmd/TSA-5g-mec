# TSA-5G-MEC

Code and data for the paper *Trustworthy Service Availability: Joint Security, Dependability
and Performance for 5G Multi-Access Edge Computing* (Arnav Chaudhary, Manipal University Jaipur).

TSA is the probability that a request is answered on time by a MEC host that is up and not
compromised. TRIO uses it to decide how many replicas of a service to run and where. Everything
here is simulated: no external dataset is used, and all scenarios come from `model.py` with fixed
random seeds, so every number can be regenerated.

## Files

| File | What it is |
|---|---|
| `model.py` | System model: host Markov chain (up / compromised / failed), failure domains, M/M/1 latency levels, closed-form TSA |
| `schemes.py` | Placement engine, TRIO, the four baselines (PERF, PERF+DEP, PERF+SEC, SEQ) and the ILP used as a reference |
| `simulate.py` | Runs the experiments and writes raw JSON to `raw/` |
| `export_dataset.py` | Turns `raw/*.json` into the CSV files below and `results.json` |
| `check.py` | Compares the closed-form TSA with a trajectory-level simulation |
| `plot.py` | Draws the two result figures from `E1_load.csv` and `E2_attack_rate.csv` |
| `E1_load.csv` | Every scheme, margin and seed for 50–300 users (paper: Fig. 2, Table 3) |
| `E2_attack_rate.csv` | 200 users, attack rate 0–0.04 per hour (paper: Fig. 3) |
| `E3_security_reservation.csv` | Share of host capacity reserved for monitoring, 0–30 % |
| `E4_ablation.csv` | TRIO with one component removed at a time |
| `E5_optimality.csv` | TRIO against the ILP on 5-host and 12-host systems |
| `scenarios_hosts.csv` | The 12 MEC hosts of each scenario (1,080 rows) |
| `scenarios_tasks.csv` | One task per user in each scenario (15,750 rows) |
| `results.json` | Summary of all experiments (means, best margin per baseline, runtimes) |

A scenario is identified by `users` and `seed`. Isolation classes: 1 = shared-kernel container,
2 = sandboxed micro-VM, 3 = TEE-backed VM. In the result files, `met_target_share` is the share of
tasks whose TSA reached its target, and the `n_*` columns count why the other tasks missed it.
Baselines were run with safety margins 1, 0.5 and 0.2; the paper reports the best one for each point.

## Running it

```
pip install -r requirements.txt
python simulate.py load attack     # experiments used in the paper (about 25 min on 2 cores)
python simulate.py                 # all experiments (about 1 h)
python export_dataset.py
python plot.py
python check.py
```

`python simulate.py --quick load` uses 2 seeds per point instead of 15 and finishes in a few minutes.

## Main results (200 users)

| Scheme | Tasks meeting target |
|---|---|
| TRIO | 95.8 % |
| SEQ | 71.4 % |
| PERF+SEC | 65.5 % |
| PERF+DEP | 47.5 % |
| PERF | 40.5 % |
