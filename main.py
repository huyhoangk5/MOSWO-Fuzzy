"""
Run MOSWO with 4-input 6-output Fuzzy Adaptive Parameter Control against ZDT and DTLZ benchmarks.
Setup follows Section 4.1 in paper: pop_size=100, ZDT 10000 FEs, DTLZ 20000 FEs.
Evaluation metrics: IGD (Inverted Generational Distance), HV (Hypervolume), Spread.
"""

import sys
import numpy as np
from scipy.spatial.distance import cdist

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from MOSWO import MOSWO
from benchmarks import make_benchmarks
from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV


def compute_spread(PopObj, Optimum):
    if len(PopObj) < 2:
        return 1.0
    N, M = PopObj.shape
    if M == 2:
        PopObj = PopObj[np.argsort(PopObj[:, 0])]
        Optimum = Optimum[np.argsort(Optimum[:, 0])]
        d = np.linalg.norm(PopObj[:-1] - PopObj[1:], axis=1)
        d_mean = np.mean(d)
        d_f = np.linalg.norm(PopObj[0] - Optimum[0])
        d_l = np.linalg.norm(PopObj[-1] - Optimum[-1])
        if d_mean == 0:
            return 1.0
        return (d_f + d_l + np.sum(np.abs(d - d_mean))) / (d_f + d_l + (N - 1) * d_mean)
    else:
        extreme_idx = np.argmax(Optimum, axis=0)
        Extreme = Optimum[extreme_idx]
        d1 = np.sum(np.min(cdist(Extreme, PopObj), axis=1))
        d2 = cdist(PopObj, PopObj)
        np.fill_diagonal(d2, np.inf)
        d = np.min(d2, axis=1)
        d_mean = np.mean(d)
        if d_mean == 0:
            return 1.0
        return (d1 + np.sum(np.abs(d - d_mean))) / (d1 + N * d_mean)


def compute_hv(PopObj, Optimum):
    fmin = np.min(Optimum, axis=0)
    fmax = np.max(Optimum, axis=0)
    denom = fmax - fmin
    denom[denom == 0] = 1.0
    norm_obj = (PopObj - fmin) / denom
    ref_point = np.ones(PopObj.shape[1]) * 1.1
    valid = np.all(norm_obj <= 1.1, axis=1)
    if not np.any(valid):
        return 0.0
    try:
        return float(HV(ref_point=ref_point)(norm_obj[valid]))
    except Exception:
        return 0.0


def run_benchmark(pop_size: int = 100, runs: int = 5, dtlz_obj: int = 3, seed: int = 42):
    """Run benchmark suite and print evaluation metrics."""
    print(f"\n{'='*75}")
    print(f"  MOSWO (Fuzzy Adaptive Controller) — {runs} runs")
    print(f"{'='*75}\n")
    print(f"{'Problem':<8} | {'Obj':<4} | {'Mean Time':<10} | {'IGD (mean)':<12} | "
          f"{'HV (mean)':<12} | {'Spread (mean)':<12}")
    print("-" * 75)

    all_benchmarks = make_benchmarks(n_obj=dtlz_obj)

    for problem in all_benchmarks:
        durations, igd_list, hv_list, spread_list = [], [], [], []

        # FE budgeting: ZDT ~ 10,000 FEs, DTLZ ~ 20,000 FEs
        if problem.name == "ZDT4":
            max_iter = 135
            n_min_val = 50
        elif problem.name.startswith("ZDT"):
            max_iter = 150
            n_min_val = None
        else:
            max_iter = 310
            n_min_val = None

        problem_key = problem.name.lower()
        try:
            pymoo_prob = (get_problem(problem_key, n_obj=problem.n_obj)
                          if problem.name.startswith("DTLZ")
                          else get_problem(problem_key))
            true_pareto = pymoo_prob.pareto_front()
            if true_pareto is None:
                raise ValueError
        except Exception:
            if problem.n_obj == 2:
                t = np.linspace(0, 1, 500)
                true_pareto = np.column_stack((t, 1.0 - np.sqrt(t)))
            else:
                true_pareto = np.random.uniform(0, 1, (500, problem.n_obj))

        igd_indicator = IGD(true_pareto)

        for run_index in range(runs):
            optimizer = MOSWO(
                obj_func=problem.function,
                n_dim=problem.n_dim,
                lower_bound=problem.lower_bound,
                upper_bound=problem.upper_bound,
                pop_size=pop_size,
                max_iter=max_iter,
                n_obj=problem.n_obj,
                archive_size=pop_size,
                CR=0.8,
                use_fuzzy=True,
                random_state=seed + run_index * 1000,
                n_min=n_min_val,
            )
            archive_pop, pareto_objs, elapsed = optimizer.optimize()
            durations.append(elapsed)

            fronts = optimizer.non_dominated_sort(pareto_objs)
            pareto_objs_nd = pareto_objs[fronts[0]]

            try:
                igd_val = float(igd_indicator(pareto_objs_nd))
            except Exception:
                igd_val = float("inf")

            hv_val = compute_hv(pareto_objs_nd, true_pareto)
            spread_val = compute_spread(pareto_objs_nd, true_pareto)

            igd_list.append(igd_val)
            hv_list.append(hv_val)
            spread_list.append(spread_val)

        m_igd = float(np.mean(igd_list))
        m_hv = float(np.mean(hv_list))
        m_spread = float(np.mean(spread_list))
        m_time = float(np.mean(durations))

        print(f"{problem.name:<8} | {problem.n_obj:<4} | {m_time:<9.2f}s | "
              f"{m_igd:<12.4e} | {m_hv:<12.4f} | {m_spread:<12.4f}", flush=True)

    print("-" * 75)


def main():
    pop_size = 100
    runs = 30
    dtlz_objectives = 3
    seed = 42

    run_benchmark(pop_size=pop_size, runs=runs, dtlz_obj=dtlz_objectives, seed=seed)


if __name__ == "__main__":
    main()
