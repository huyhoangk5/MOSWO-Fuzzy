"""
Ablation Study Runner for MOSWO (Công việc 1 trong Kế hoạch Nghiên cứu).

Đánh giá 7 biến thể theo bảng phân loại của giảng viên hướng dẫn:
  1. MOSWO          : Nguyên bản (Sequential ND-sort & CD, tham số cố định)
  2. MOSWO-V        : Chỉ chuyển đổi sang Vector hóa
  3. MOSWO-A        : Các tham số thích ứng tất định theo thời gian (không dùng mờ)
  4. MOSWO-F        : Bộ điều khiển mờ 4-in 6-out (điều khiển bước nhảy & lai ghép/đột biến)
  5. MOSWO-F+Cauchy : MOSWO-F kết hợp đột biến nhảy bước lớn Cauchy
  6. MOSWO-F+Reinit : MOSWO-F kết hợp tái khởi tạo một phần thích ứng (Partial Reinit + OBL)
  7. MOSWO-Full     : Phiên bản đầy đủ v10 (Tích hợp toàn bộ mô hình)
"""

import sys
import argparse
import pathlib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import ranksums

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from MOSWO import MOSWO
from benchmarks import make_benchmarks
from main import compute_hv, compute_spread
from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD


VARIANTS = [
    ("MOSWO",          "Nguyên bản"),
    ("MOSWO-V",        "Chỉ chuyển đổi thành vector"),
    ("MOSWO-A",        "Tham số thích ứng không mờ"),
    ("MOSWO-F",        "Bộ điều khiển mờ"),
    ("MOSWO-F+Cauchy", "Bộ mờ + Đột biến Cauchy"),
    ("MOSWO-F+Reinit", "Bộ mờ + Khởi tạo lại thích ứng"),
    ("MOSWO-Full",     "Phiên bản đầy đủ v10 (Toàn bộ mô hình)"),
]


def get_true_pareto(problem):
    problem_key = problem.name.lower()
    try:
        pymoo_prob = (
            get_problem(problem_key, n_obj=problem.n_obj)
            if problem.name.startswith("DTLZ")
            else get_problem(problem_key)
        )
        true_pareto = pymoo_prob.pareto_front()
        if true_pareto is None:
            raise ValueError
        return true_pareto
    except Exception:
        if problem.n_obj == 2:
            t = np.linspace(0, 1, 500)
            return np.column_stack((t, 1.0 - np.sqrt(t)))
        return np.random.uniform(0, 1, (500, problem.n_obj))


def get_fe_budget(problem_name: str):
    if problem_name == "ZDT4":
        return 135, 50
    elif problem_name.startswith("ZDT"):
        return 150, None
    else:
        return 310, None


def wilcoxon_sign(ref_vals, other_vals, lower_is_better=True, alpha=0.05):
    """Compare MOSWO-Full (ref_vals) vs another variant (other_vals).
    Returns '+' if MOSWO-Full is significantly better,
            '-' if MOSWO-Full is significantly worse,
            '≈' if no statistically significant difference.
    """
    ref_arr = np.asarray(ref_vals, dtype=float)
    oth_arr = np.asarray(other_vals, dtype=float)
    if np.allclose(ref_arr, oth_arr):
        return "≈"
    try:
        _, p_val = ranksums(ref_arr, oth_arr)
    except Exception:
        return "≈"
    if p_val >= alpha:
        return "≈"
    ref_mean = float(np.mean(ref_arr))
    oth_mean = float(np.mean(oth_arr))
    if lower_is_better:
        return "+" if ref_mean < oth_mean else "-"
    else:
        return "+" if ref_mean > oth_mean else "-"


def run_ablation(pop_size: int = 100, runs: int = 10, dtlz_obj: int = 3,
                 seed: int = 42, problem_filter=None, report_path: pathlib.Path = None):
    all_benchmarks = make_benchmarks(n_obj=dtlz_obj)
    if problem_filter:
        filter_set = {p.upper() for p in problem_filter}
        all_benchmarks = [b for b in all_benchmarks if b.name.upper() in filter_set]

    variant_keys = [v[0] for v in VARIANTS]
    results = {v_key: {} for v_key in variant_keys}

    print(f"\n{'='*95}", flush=True)
    print(f"  THỰC NGHIỆM CẮT BỎ (ABLATION STUDY) — 7 BIẾN THỂ MOSWO ({runs} runs độc lập)", flush=True)
    print(f"{'='*95}", flush=True)

    for problem in all_benchmarks:
        max_iter, n_min_val = get_fe_budget(problem.name)
        true_pareto = get_true_pareto(problem)
        igd_indicator = IGD(true_pareto)

        print(f"\n>>> Bài toán: {problem.name} (M={problem.n_obj}, D={problem.n_dim}, max_iter={max_iter})", flush=True)
        print(f"{'Biến thể':<16} | {'Time (s)':<10} | {'IGD (Mean ± Std)':<23} | "
              f"{'HV (Mean ± Std)':<23} | {'Spread (Mean ± Std)':<23}", flush=True)
        print("-" * 105, flush=True)

        for v_key, _ in VARIANTS:
            durations, igd_list, hv_list, spread_list = [], [], [], []

            for run_idx in range(runs):
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
                    n_min=n_min_val,
                    random_state=seed + run_idx * 1000,
                    variant=v_key,
                )
                _, pareto_objs, elapsed = optimizer.optimize()
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

            entry = {
                "igd_mean": float(np.mean(igd_list)),
                "igd_std": float(np.std(igd_list)),
                "igd_raw": igd_list,
                "hv_mean": float(np.mean(hv_list)),
                "hv_std": float(np.std(hv_list)),
                "hv_raw": hv_list,
                "spread_mean": float(np.mean(spread_list)),
                "spread_std": float(np.std(spread_list)),
                "spread_raw": spread_list,
                "time_mean": float(np.mean(durations)),
            }
            results[v_key][problem.name] = entry

            print(
                f"{v_key:<16} | {entry['time_mean']:<9.2f}s | "
                f"{entry['igd_mean']:.4e} ± {entry['igd_std']:.2e} | "
                f"{entry['hv_mean']:.4f} ± {entry['hv_std']:.4f}   | "
                f"{entry['spread_mean']:.4f} ± {entry['spread_std']:.4f}",
                flush=True,
            )

    if report_path is not None:
        write_markdown_report(results, all_benchmarks, runs, report_path)
        print(f"\n[OK] Báo cáo Ablation Study đã được xuất ra: {report_path}", flush=True)

    return results


def write_markdown_report(results: dict, benchmarks_list: list, runs: int, report_path: pathlib.Path):
    variant_keys = [v[0] for v in VARIANTS]
    prob_names = [b.name for b in benchmarks_list]

    lines = [
        "# BÁO CÁO KẾT QUẢ THỰC NGHIỆM CẮT BỎ (ABLATION STUDY)",
        "",
        f"**Số lần chạy độc lập (Independent Runs):** `{runs}`  ",
        "**Ngân sách đánh giá hàm mục tiêu (FEs):** ZDT = 10,000 FEs (`pop_size=100`), DTLZ = 20,000 FEs (`pop_size=100`, `M=3`)  ",
        "**Kiểm định thống kê:** Wilcoxon rank-sum test ($\\alpha = 0.05$) so sánh `MOSWO-Full` với từng biến thể rút gọn (`+`: `MOSWO-Full` tốt hơn có ý nghĩa, `≈`: tương đương, `-`: kém hơn).",
        "",
        "---",
        "",
        "## 1. BẢNG ĐỊNH NGHĨA CÁC PHIÊN BẢN TRONG ABLATION STUDY",
        "",
        "| Phiên bản | Ý nghĩa | Vector hóa | Tham số thích ứng | Điều khiển mờ (4-in 6-out) | Đột biến Cauchy | Khởi tạo lại thích ứng (Reinit + OBL) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        "| `MOSWO` | Nguyên bản | ✗ | ✗ | ✗ | ✗ | ✗ |",
        "| `MOSWO-V` | Chỉ được chuyển đổi thành vector | ✓ | ✗ | ✗ | ✗ | ✗ |",
        "| `MOSWO-A` | Các tham số thích ứng không bị mờ | ✓ | ✓ | ✗ | ✗ | ✗ |",
        "| `MOSWO-F` | Bộ điều khiển mờ | ✓ | ✓ | ✓ | ✗ | ✗ |",
        "| `MOSWO-F+Cauchy` | Bộ điều khiển mờ + Đột biến Cauchy | ✓ | ✓ | ✓ | ✓ | ✗ |",
        "| `MOSWO-F+Reinit` | Bộ điều khiển mờ + Khởi tạo lại thích ứng hơn | ✓ | ✓ | ✓ | ✗ | ✓ |",
        "| `MOSWO-Full` | Phiên bản đầy đủ v10 (Toàn bộ mô hình) | ✓ | ✓ | ✓ | ✓ | ✓ |",
        "",
        "---",
        "",
    ]

    # Section 2: Runtime comparison (MOSWO vs MOSWO-V vs MOSWO-Full)
    lines.extend([
        "## 2. SO SÁNH THỜI GIAN THỰC THI TRUNG BÌNH (GIÂY / RUN)",
        "",
        "| Bài toán | " + " | ".join(f"`{v}`" for v in variant_keys) + " | Tăng tốc (`MOSWO` $\\rightarrow$ `MOSWO-V`) |",
        "| :--- | " + " | ".join(":---:" for _ in variant_keys) + " | :---: |",
    ])
    for pname in prob_names:
        t_orig = results["MOSWO"][pname]["time_mean"]
        t_vec = results["MOSWO-V"][pname]["time_mean"]
        speedup = t_orig / max(t_vec, 1e-6)
        row_cells = [f"{results[v][pname]['time_mean']:.2f}s" for v in variant_keys]
        lines.append(f"| **{pname}** | " + " | ".join(row_cells) + f" | **{speedup:.2f}x** |")
    lines.extend(["", "---", ""])

    # Sections 3, 4, 5: IGD, HV, Spread tables
    metrics_info = [
        ("igd", "3. KẾT QUẢ ĐỘ ĐO IGD (INVERTED GENERATIONAL DISTANCE — CÀNG NHỎ CÀNG TỐT)", True, ".4e"),
        ("hv", "4. KẾT QUẢ ĐỘ ĐO HV (HYPERVOLUME — CÀNG LỚN CÀNG TỐT)", False, ".4f"),
        ("spread", "5. KẾT QUẢ ĐỘ ĐO SPREAD (ĐỘ PHÂN BỐ — CÀNG NHỎ CÀNG TỐT)", True, ".4f"),
    ]

    for m_prefix, title, lower_better, fmt in metrics_info:
        lines.extend([
            f"## {title}",
            "",
            "| Bài toán | " + " | ".join(f"`{v}`" for v in variant_keys) + " |",
            "| :--- | " + " | ".join(":---:" for _ in variant_keys) + " |",
        ])

        ranks_per_variant = {v: [] for v in variant_keys}
        w_counts = {v: {"+": 0, "≈": 0, "-": 0} for v in variant_keys[:-1]}

        for pname in prob_names:
            means = [results[v][pname][f"{m_prefix}_mean"] for v in variant_keys]
            if lower_better:
                best_idx = int(np.argmin(means))
                order = np.argsort(means)
            else:
                best_idx = int(np.argmax(means))
                order = np.argsort(-np.asarray(means))

            ranks = np.empty(len(variant_keys), dtype=float)
            for r_pos, idx in enumerate(order):
                ranks[idx] = r_pos + 1
            for idx, v in enumerate(variant_keys):
                ranks_per_variant[v].append(ranks[idx])

            ref_raw = results["MOSWO-Full"][pname][f"{m_prefix}_raw"]
            row_cells = []
            for idx, v in enumerate(variant_keys):
                m_val = results[v][pname][f"{m_prefix}_mean"]
                s_val = results[v][pname][f"{m_prefix}_std"]
                cell_str = f"{m_val:{fmt}} ± {s_val:.2e}"
                if v != "MOSWO-Full":
                    sgn = wilcoxon_sign(ref_raw, results[v][pname][f"{m_prefix}_raw"], lower_is_better=lower_better)
                    w_counts[v][sgn] += 1
                    cell_str += f" ({sgn})"
                if idx == best_idx:
                    cell_str = f"**{cell_str}**"
                row_cells.append(cell_str)

            lines.append(f"| **{pname}** | " + " | ".join(row_cells) + " |")

        # Summary rows: Avg Rank and Wilcoxon (+/≈/-)
        rank_cells = [f"**{np.mean(ranks_per_variant[v]):.2f}**" for v in variant_keys]
        lines.append("| **Hạng TB (Mean Rank)** | " + " | ".join(rank_cells) + " |")

        w_cells = [
            f"**{w_counts[v]['+']} / {w_counts[v]['≈']} / {w_counts[v]['-']}**"
            for v in variant_keys[:-1]
        ] + ["*Baseline (Full)*"]
        lines.append("| **Wilcoxon (+ / ≈ / -)** | " + " | ".join(w_cells) + " |")
        lines.extend(["", "---", ""])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run MOSWO Ablation Study (7 variants)")
    parser.add_argument("--runs", type=int, default=10, help="Number of independent runs per problem")
    parser.add_argument("--pop-size", type=int, default=100, help="Population size")
    parser.add_argument("--dtlz-obj", type=int, default=3, help="Number of objectives for DTLZ problems")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--problems", nargs="*", default=None, help="Optional subset of problems (e.g. ZDT1 DTLZ3)")
    parser.add_argument(
        "--output",
        type=str,
        default=str(pathlib.Path(__file__).parent / "ABLATION_RESULTS.md"),
        help="Output Markdown report path",
    )
    args = parser.parse_args()

    run_ablation(
        pop_size=args.pop_size,
        runs=args.runs,
        dtlz_obj=args.dtlz_obj,
        seed=args.seed,
        problem_filter=args.problems,
        report_path=pathlib.Path(args.output),
    )


if __name__ == "__main__":
    main()

