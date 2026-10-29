"""
MOSWO: Multiobjective Spider-Wasp Optimizer with 4-Input 6-Output Fuzzy Adaptive Parameter Control.

Based on the Spider-Wasp Optimizer framework (Nguyen et al., Biomimetics 2025)
enhanced with dynamic adaptive parameter control:
  - 4 Inputs:
      1. iter_ratio    : t / T
      2. diversity     : objective space diversity
      3. g_excess      : distance of archive from ideal Pareto front
      4. dec_diversity : decision space diversity
  - 6 Outputs:
      1. mu_scale       : exploration step multiplier
      2. c_factor       : exploitation speed multiplier
      3. reinit_strength: partial reinitialization fraction for stuck g-dimensions
      4. cr_rate        : adaptive crossover rate
      5. eta_m          : adaptive polynomial mutation distribution index
      6. cauchy_prob    : adaptive Cauchy mutation probability
"""

import time
import numpy as np
from scipy.spatial.distance import cdist
from fuzzy_controller_skfuzzy import FuzzyMOSWOController


_SHARED_FUZZY_CONTROLLER = None


def _get_fuzzy_controller():
    global _SHARED_FUZZY_CONTROLLER
    if _SHARED_FUZZY_CONTROLLER is None:
        _SHARED_FUZZY_CONTROLLER = FuzzyMOSWOController()
    return _SHARED_FUZZY_CONTROLLER


class MOSWO:
    VARIANT_CONFIGS = {
        "MOSWO": {
            "use_vectorized": False,
            "use_adaptive": False,
            "use_fuzzy": False,
            "use_cauchy": False,
            "use_reinit": False,
            "use_obl": False,
        },
        "MOSWO-V": {
            "use_vectorized": True,
            "use_adaptive": False,
            "use_fuzzy": False,
            "use_cauchy": False,
            "use_reinit": False,
            "use_obl": False,
        },
        "MOSWO-A": {
            "use_vectorized": True,
            "use_adaptive": True,
            "use_fuzzy": False,
            "use_cauchy": False,
            "use_reinit": False,
            "use_obl": False,
        },
        "MOSWO-F": {
            "use_vectorized": True,
            "use_adaptive": False,
            "use_fuzzy": True,
            "use_cauchy": False,
            "use_reinit": False,
            "use_obl": False,
        },
        "MOSWO-F+Cauchy": {
            "use_vectorized": True,
            "use_adaptive": False,
            "use_fuzzy": True,
            "use_cauchy": True,
            "use_reinit": False,
            "use_obl": False,
        },
        "MOSWO-F+Reinit": {
            "use_vectorized": True,
            "use_adaptive": False,
            "use_fuzzy": True,
            "use_cauchy": False,
            "use_reinit": True,
            "use_obl": True,
        },
        "MOSWO-Full": {
            "use_vectorized": True,
            "use_adaptive": False,
            "use_fuzzy": True,
            "use_cauchy": True,
            "use_reinit": True,
            "use_obl": True,
        },
    }

    def __init__(
        self,
        obj_func,
        n_dim: int,
        lower_bound,
        upper_bound,
        pop_size: int = 100,
        max_iter: int = 100,
        n_obj: int = 2,
        archive_size: int = 100,
        CR: float = 0.8,
        use_vectorized: bool = True,
        use_adaptive: bool = False,
        use_fuzzy: bool = True,
        use_cauchy: bool = True,
        use_reinit: bool = True,
        use_obl: bool = True,
        obl_period: int = 5,
        obl_frac: float = 0.25,
        stagnant_limit: int = 2,
        cauchy_scale: float = 0.05,
        g_excess_threshold: float = 0.22,
        use_arch_div: bool = False,
        active_rules_mask=None,
        n_min: int = None,
        random_state: int = None,
        variant: str = None,
    ):
        if variant is not None:
            if variant not in self.VARIANT_CONFIGS:
                raise ValueError(f"Unknown variant '{variant}'. Valid options: {list(self.VARIANT_CONFIGS.keys())}")
            cfg = self.VARIANT_CONFIGS[variant]
            use_vectorized = cfg["use_vectorized"]
            use_adaptive   = cfg["use_adaptive"]
            use_fuzzy      = cfg["use_fuzzy"]
            use_cauchy     = cfg["use_cauchy"]
            use_reinit     = cfg["use_reinit"]
            use_obl        = cfg["use_obl"]

        self.obj_func           = obj_func
        self.n_dim              = n_dim
        self.lb                 = np.broadcast_to(np.asarray(lower_bound, dtype=float), (n_dim,)).copy()
        self.ub                 = np.broadcast_to(np.asarray(upper_bound, dtype=float), (n_dim,)).copy()
        self.pop_size           = pop_size
        self.max_iter           = max_iter
        self.n_obj              = n_obj
        self.archive_size       = archive_size
        self.CR                 = CR
        self.use_vectorized     = use_vectorized
        self.use_adaptive       = use_adaptive
        self.use_fuzzy          = use_fuzzy
        self.use_cauchy         = use_cauchy
        self.use_reinit         = use_reinit
        self.use_obl            = use_obl
        self.obl_period         = obl_period
        self.obl_frac           = obl_frac
        self.stagnant_limit     = stagnant_limit
        self.cauchy_scale       = cauchy_scale
        self.g_excess_threshold = g_excess_threshold
        self.use_arch_div       = use_arch_div
        self.active_rules_mask  = active_rules_mask
        self.n_min              = n_min
        self.rng                = np.random.default_rng(random_state)
        aux_seed                = (random_state + 777777) if random_state is not None else None
        self.aux_rng            = np.random.default_rng(aux_seed)

        if np.any(self.lb >= self.ub):
            raise ValueError("Lower bounds must be strictly less than upper bounds.")

        self._fuzzy = _get_fuzzy_controller() if use_fuzzy else None
        if self._fuzzy is not None:
            if active_rules_mask is not None:
                self._fuzzy.active_rules_mask = np.asarray(active_rules_mask, dtype=bool)
            else:
                self._fuzzy.active_rules_mask = np.ones(len(self._fuzzy._rule_table), dtype=bool)

        # Number of front-position variables to preserve during partial reinit
        # The first (n_obj - 1) dimensions control front distribution,
        # the remaining dimensions control convergence (g-function).
        self._n_front_vars = max(1, self.n_obj - 1)

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)

    def _rand_init(self) -> np.ndarray:
        return self.rng.uniform(self.lb, self.ub, (self.pop_size, self.n_dim))

    def _evaluate(self, pop: np.ndarray) -> np.ndarray:
        return np.array([self.obj_func(ind) for ind in pop], dtype=float)

    @staticmethod
    def dominates(a: np.ndarray, b: np.ndarray) -> bool:
        return np.all(a <= b) and np.any(a < b)

    # ------------------------------------------------------------------ #
    # Fuzzy metrics computation
    # ------------------------------------------------------------------ #

    def _compute_diversity(self, objs: np.ndarray) -> float:
        """Scale-invariant nearest-neighbour diversity in normalized objective space in [0, 1]."""
        n_pts = len(objs)
        if n_pts < 2:
            return 0.0
        fmin = objs.min(axis=0)
        fmax = objs.max(axis=0)
        denom = fmax - fmin
        denom[denom == 0] = 1.0
        norm = (objs - fmin) / denom
        d = cdist(norm, norm)
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        # Expected spacing of n_pts uniform points on an (M-1)-D Pareto manifold
        expected_nn = (n_pts ** (-1.0 / max(1, self.n_obj - 1))) * np.sqrt(self.n_obj - 1)
        raw = (np.mean(nn) / max(expected_nn, 1e-6)) * 0.65
        return float(np.clip(raw, 0.0, 1.0))

    def _compute_g_excess(self, objs: np.ndarray) -> float:
        """Estimate distance of current archive from true Pareto front in [0, 1]."""
        if len(objs) < 1:
            return 0.0
        mean_sum = float(np.mean(objs.sum(axis=1)))
        scale = 10.0
        return float(np.clip(np.log1p(mean_sum) / np.log1p(scale), 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Non-dominated sorting & Crowding distance
    # ------------------------------------------------------------------ #

    def _non_dominated_sort_loop(self, objs: np.ndarray):
        """Sequential O(M * N^2) non-dominated sorting using standard Python loops (original MOSWO)."""
        objs_list = np.asarray(objs, dtype=float).tolist()
        n = len(objs_list)
        if n == 0:
            return []
        m_obj = self.n_obj
        domination_sets = [[] for _ in range(n)]
        domination_counts = [0] * n
        fronts = [[]]

        for p in range(n):
            fp = objs_list[p]
            for q in range(p + 1, n):
                fq = objs_list[q]
                p_le_q = True
                q_le_p = True
                p_lt_q = False
                q_lt_p = False
                for k in range(m_obj):
                    if fp[k] > fq[k]:
                        p_le_q = False
                        q_lt_p = True
                    elif fp[k] < fq[k]:
                        q_le_p = False
                        p_lt_q = True
                if p_le_q and p_lt_q:
                    domination_sets[p].append(q)
                    domination_counts[q] += 1
                elif q_le_p and q_lt_p:
                    domination_sets[q].append(p)
                    domination_counts[p] += 1
            if domination_counts[p] == 0:
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in domination_sets[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        next_front.append(q)
            i += 1
            if len(next_front) > 0:
                fronts.append(sorted(next_front))
            else:
                break
        return fronts

    def non_dominated_sort(self, objs: np.ndarray):
        """Non-dominated sorting (vectorized or sequential based on configuration)."""
        if not self.use_vectorized:
            return self._non_dominated_sort_loop(objs)
        objs = np.asarray(objs, dtype=float)
        n = len(objs)
        if n == 0:
            return []
        a = objs[:, None, :]
        b = objs[None, :, :]
        d = np.all(a <= b, axis=2) & np.any(a < b, axis=2)
        counts = d.sum(axis=0)
        remaining = np.ones(n, dtype=bool)
        fronts = []
        current = np.flatnonzero(counts == 0)
        while current.size:
            fronts.append(current.tolist())
            remaining[current] = False
            counts = counts - d[current].sum(axis=0)
            current = np.flatnonzero((counts == 0) & remaining)
        return fronts

    def _calculate_crowding_distance_loop(self, objs: np.ndarray, front):
        """Sequential crowding distance calculation (original MOSWO)."""
        front = list(front)
        l = len(front)
        if l <= 2:
            return {idx: float("inf") for idx in front}
        distance = {idx: 0.0 for idx in front}
        for m in range(self.n_obj):
            sorted_front = sorted(front, key=lambda idx: float(objs[idx, m]))
            distance[sorted_front[0]] = float("inf")
            distance[sorted_front[-1]] = float("inf")
            f_min = float(objs[sorted_front[0], m])
            f_max = float(objs[sorted_front[-1], m])
            span = f_max - f_min
            if span == 0.0:
                continue
            for i in range(1, l - 1):
                prev_f = float(objs[sorted_front[i - 1], m])
                next_f = float(objs[sorted_front[i + 1], m])
                distance[sorted_front[i]] += (next_f - prev_f) / span
        return distance

    def calculate_crowding_distance(self, objs: np.ndarray, front):
        """Crowding distance calculation for a given front."""
        if not self.use_vectorized:
            return self._calculate_crowding_distance_loop(objs, front)
        front = list(front)
        l = len(front)
        if l <= 2:
            return {idx: float("inf") for idx in front}
        f = np.asarray(objs, dtype=float)[front]
        dist = np.zeros(l)
        for m in range(self.n_obj):
            order = np.argsort(f[:, m], kind="stable")
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            span = f[order[-1], m] - f[order[0], m]
            if span == 0:
                continue
            dist[order[1:-1]] += (f[order[2:], m] - f[order[:-2], m]) / span
        return dict(zip(front, dist.tolist()))

    @staticmethod
    def _worst_indices(fronts, n: int):
        """Pick n individuals starting from the worst front."""
        pool = []
        for fr in reversed(fronts):
            pool.extend(fr)
            if len(pool) >= n:
                break
        return pool[:n]

    # ------------------------------------------------------------------ #
    # Archive management
    # ------------------------------------------------------------------ #

    def _remove_duplicates(self, pop: np.ndarray, objs: np.ndarray, tol: float = 1e-8):
        rounded = np.round(objs / tol).astype(np.int64)
        _, unique = np.unique(rounded, axis=0, return_index=True)
        return pop[unique], objs[unique]

    def _update_archive(self, archive_pop: np.ndarray, archive_objs: np.ndarray,
                        new_pop: np.ndarray, new_objs: np.ndarray):
        if len(archive_pop) == 0:
            combined_pop = new_pop.copy()
            combined_objs = new_objs.copy()
        else:
            combined_pop = np.vstack((archive_pop, new_pop))
            combined_objs = np.vstack((archive_objs, new_objs))

        combined_pop, combined_objs = self._remove_duplicates(combined_pop, combined_objs)
        fronts = self.non_dominated_sort(combined_objs)
        nd_pop = combined_pop[fronts[0]]
        nd_objs = combined_objs[fronts[0]]

        if len(nd_pop) > self.archive_size:
            cd = self.calculate_crowding_distance(nd_objs, list(range(len(nd_objs))))
            sorted_idx = sorted(range(len(nd_objs)), key=lambda i: cd[i], reverse=True)
            nd_pop = nd_pop[sorted_idx[:self.archive_size]]
            nd_objs = nd_objs[sorted_idx[:self.archive_size]]

        return nd_pop, nd_objs

    def _select_leader(self, archive_pop: np.ndarray, cd_map: dict):
        n = len(archive_pop)
        if n <= 2:
            return archive_pop[self.rng.integers(n)]
        i1, i2 = self.rng.choice(n, 2, replace=False)
        return archive_pop[i1 if cd_map[i1] >= cd_map[i2] else i2]

    # ------------------------------------------------------------------ #
    # Mutation & Diversity Operators
    # ------------------------------------------------------------------ #

    def _polynomial_mutation(self, x: np.ndarray, eta_m: float = 20.0, prob: float = None) -> np.ndarray:
        if prob is None:
            prob = 1.0 / self.n_dim
        y = x.copy()
        for j in range(self.n_dim):
            if self.rng.random() < prob:
                yl, yu = self.lb[j], self.ub[j]
                d1 = (y[j] - yl) / (yu - yl) if (yu - yl) > 0 else 0
                d2 = (yu - y[j]) / (yu - yl) if (yu - yl) > 0 else 0
                u = self.rng.random()
                if u < 0.5:
                    xy = 1.0 - d1
                    val = 2 * u + (1 - 2 * u) * (xy ** (eta_m + 1))
                    dq = val ** (1 / (eta_m + 1)) - 1.0
                else:
                    xy = 1.0 - d2
                    val = 2 * (1 - u) + 2 * (u - 0.5) * (xy ** (eta_m + 1))
                    dq = 1.0 - val ** (1 / (eta_m + 1))
                y[j] += dq * (yu - yl)
        return self._clip(y)

    def _environmental_selection(self, combined_pop: np.ndarray, combined_objs: np.ndarray, target_size: int):
        combined_pop, combined_objs = self._remove_duplicates(combined_pop, combined_objs)
        if len(combined_pop) <= target_size:
            return combined_pop, combined_objs
        fronts = self.non_dominated_sort(combined_objs)
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= target_size:
                selected.extend(front)
            else:
                needed = target_size - len(selected)
                cd = self.calculate_crowding_distance(combined_objs, front)
                sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                selected.extend(sorted_front[:needed])
                break
        return combined_pop[selected], combined_objs[selected]

    def _partial_reinit(self, pop: np.ndarray, objs: np.ndarray, n_reinit: int,
                        archive_pop: np.ndarray = None, archive_objs: np.ndarray = None):
        """Coordinate/Leader-guided elitist partial reinitialization on g-controlling dimensions."""
        if n_reinit <= 0 or self.n_dim <= self._n_front_vars:
            return pop, objs

        fronts = self.non_dominated_sort(objs)
        worst_pool = []
        for fr in reversed(fronts):
            worst_pool.extend(fr)
            if len(worst_pool) >= n_reinit:
                break
        reinit_idx = worst_pool[:n_reinit]

        g_start = self._n_front_vars
        n_g = self.n_dim - g_start
        center_g = 0.5 * (self.lb[g_start:] + self.ub[g_start:])
        span_g = self.ub[g_start:] - self.lb[g_start:]

        best_arch = None
        if archive_pop is not None and len(archive_pop) > 0:
            best_arch_idx = int(np.argmin(archive_objs.sum(axis=1)))
            best_arch = archive_pop[best_arch_idx]

        for idx in reinit_idx:
            x_trial = pop[idx].copy()
            k_dims = min(n_g, int(self.aux_rng.integers(1, 3)))
            dims = self.aux_rng.choice(np.arange(g_start, self.n_dim), size=k_dims, replace=False)
            if best_arch is not None and self.aux_rng.random() < 0.70:
                leader = best_arch if self.aux_rng.random() < 0.6 else archive_pop[self.aux_rng.integers(len(archive_pop))]
                for d in dims:
                    x_trial[d] = leader[d] + self.aux_rng.normal(0.0, 0.05) * (self.ub[d] - self.lb[d])
            else:
                for d in dims:
                    if self.aux_rng.random() < 0.6:
                        x_trial[d] = center_g[d - g_start] + self.aux_rng.normal(0.0, 0.08) * span_g[d - g_start]
                    else:
                        x_trial[d] = self.aux_rng.uniform(self.lb[d], self.ub[d])
            x_trial = self._clip(x_trial)
            f_trial = self.obj_func(x_trial)
            if np.sum(f_trial) < np.sum(objs[idx]) or self.dominates(f_trial, objs[idx]):
                pop[idx] = x_trial
                objs[idx] = f_trial

        return pop, objs

    def _obl_escape(self, pop: np.ndarray, objs: np.ndarray, n_candidates: int,
                    candidates=None, archive_pop: np.ndarray = None, archive_objs: np.ndarray = None) -> tuple:
        """Opposition-Based Learning (quasi-opposition & leader-reflected) on g-controlling dimensions."""
        if n_candidates <= 0 or self.n_dim <= self._n_front_vars:
            return pop, objs

        g_start = self._n_front_vars
        if candidates is None:
            candidates = self._worst_indices(self.non_dominated_sort(objs), n_candidates)

        best_arch = None
        if archive_pop is not None and len(archive_pop) > 0:
            best_arch = archive_pop[int(np.argmin(archive_objs.sum(axis=1)))]

        for idx in candidates:
            x_orig = pop[idx].copy()
            x_opp = x_orig.copy()
            d = int(self.aux_rng.integers(g_start, self.n_dim))
            mid_d = 0.5 * (self.lb[d] + self.ub[d])
            opp_d = self.lb[d] + self.ub[d] - x_orig[d]
            if best_arch is not None and self.aux_rng.random() < 0.5:
                # Quasi-reflection toward best archive g-coordinate and center
                piv = 0.5 * (best_arch[d] + mid_d)
                x_opp[d] = self.aux_rng.uniform(min(piv, opp_d), max(piv, opp_d))
            else:
                x_opp[d] = self.aux_rng.uniform(min(mid_d, opp_d), max(mid_d, opp_d))
            x_opp = self._clip(x_opp)

            f_opp = self.obj_func(x_opp)
            if np.sum(f_opp) < np.sum(objs[idx]) or self.dominates(f_opp, objs[idx]):
                pop[idx] = x_opp
                objs[idx] = f_opp

        return pop, objs

    def _cauchy_escape(self, pop: np.ndarray, objs: np.ndarray, archive_pop: np.ndarray,
                       archive_objs: np.ndarray, cauchy_prob_val: float, progress: float) -> tuple:
        """Elitist coordinate-wise heavy-tailed Cauchy perturbation on g-controlling dimensions."""
        if self.n_dim <= self._n_front_vars or cauchy_prob_val <= 0.05:
            return pop, objs

        g_start = self._n_front_vars
        n_pop = len(pop)
        n_trials = max(2, int(cauchy_prob_val * n_pop))
        candidates = self._worst_indices(self.non_dominated_sort(objs), n_trials)

        best_arch = None
        if archive_pop is not None and len(archive_pop) > 0:
            best_arch = archive_pop[int(np.argmin(archive_objs.sum(axis=1)))]

        for idx in candidates:
            x_trial = pop[idx].copy()
            d = int(self.aux_rng.integers(g_start, self.n_dim))
            span_d = self.ub[d] - self.lb[d]
            scale_c = span_d * self.cauchy_scale * (1.0 - 0.6 * progress)
            step = float(np.clip(self.aux_rng.standard_cauchy(), -10.0, 10.0)) * scale_c
            if best_arch is not None and self.aux_rng.random() < 0.5:
                x_trial[d] = best_arch[d] + step * 0.5
            else:
                x_trial[d] = x_trial[d] + step
            x_trial = self._clip(x_trial)
            f_trial = self.obj_func(x_trial)
            if np.sum(f_trial) < np.sum(objs[idx]) or self.dominates(f_trial, objs[idx]):
                pop[idx] = x_trial
                objs[idx] = f_trial

        return pop, objs

    def _archive_diversification_nd(self, archive_pop: np.ndarray, archive_objs: np.ndarray, n_seeds: int):
        """Inject seeds covering all front-position dimensions to maintain diversity."""
        n_fp = self._n_front_vars
        seeds = self.rng.uniform(self.lb, self.ub, (n_seeds, self.n_dim))

        if n_fp == 1:
            seeds[:, 0] = np.linspace(self.lb[0], self.ub[0], n_seeds + 2)[1:-1]
        else:
            for d in range(n_fp):
                offsets = (np.arange(n_seeds) + 0.5) / n_seeds
                offsets = (offsets + d / (n_fp + 1)) % 1.0
                seeds[:, d] = self.lb[d] + offsets * (self.ub[d] - self.lb[d])

        diverse_objs = self._evaluate(seeds)
        return self._update_archive(archive_pop, archive_objs, seeds, diverse_objs)

    # ------------------------------------------------------------------ #
    # Main optimization loop
    # ------------------------------------------------------------------ #

    def optimize(self):
        start_time = time.perf_counter()

        pop = self._rand_init()
        objs = self._evaluate(pop)

        archive_pop = np.empty((0, self.n_dim))
        archive_objs = np.empty((0, self.n_obj))
        archive_pop, archive_objs = self._update_archive(archive_pop, archive_objs, pop, objs)

        n_min = self.n_min if self.n_min is not None else max(20, self.pop_size // 5)
        reduction_period = max(1, self.max_iter // 10)
        span_bounds = self.ub - self.lb
        span_norm = np.linalg.norm(span_bounds)
        velocities = np.zeros((self.pop_size, self.n_dim))

        prev_archive_mean = np.mean(archive_objs)
        stagnant_gens = 0
        reinit_cooldown = 0

        # Periodic calculation of fuzzy inputs
        G_EXCESS_PERIOD = 5
        diversity_val = 0.5
        g_excess_val = 0.5

        for t in range(1, self.max_iter + 1):
            N = len(pop)
            progress = t / self.max_iter
            w = 0.9 - 0.5 * progress

            if len(velocities) != N:
                velocities = np.resize(velocities, (N, self.n_dim))

            # Stagnation counter
            curr_mean = np.mean(archive_objs)
            if abs(curr_mean - prev_archive_mean) < 1e-4:
                stagnant_gens += 1
            else:
                stagnant_gens = 0
            prev_archive_mean = curr_mean

            # Decision-space diversity (Antecedent 4)
            dec_div = float(np.mean(np.std(pop, axis=0) / (span_bounds + 1e-12)) * np.sqrt(12))
            dec_div = float(np.clip(dec_div, 0.0, 1.0))

            # Compute objective-space diversity & g_excess (Antecedents 2 & 3)
            if t % G_EXCESS_PERIOD == 1 or t == 1:
                diversity_val = self._compute_diversity(archive_objs)
                g_excess_val = self._compute_g_excess(archive_objs)

            # Parameter adaptation: Fuzzy (4-in 6-out) vs Deterministic Adaptive vs Fixed
            if self.use_fuzzy and self._fuzzy is not None:
                try:
                    (mu_scale_val, c_factor_val, reinit_str_val,
                     cr_val, eta_m_val, cauchy_prob_val) = self._fuzzy.compute(
                        progress, diversity_val, g_excess_val, dec_div
                    )
                except Exception:
                    mu_scale_val = 2.0 - 1.5 * progress
                    c_factor_val = 0.5 + 0.5 * progress
                    reinit_str_val = 0.0
                    cr_val = self.CR
                    eta_m_val = 5.0 + 15.0 * (progress ** 2)
                    cauchy_prob_val = 0.0
                if not self.use_cauchy:
                    cauchy_prob_val = 0.0
                if not self.use_reinit:
                    reinit_str_val = 0.0
            elif self.use_adaptive:
                # Non-fuzzy time-varying adaptive schedules (MOSWO-A)
                mu_scale_val = 2.0 - 1.5 * progress
                c_factor_val = 0.5 + 0.5 * progress
                reinit_str_val = 0.0
                cr_val = 0.9 - 0.3 * progress
                eta_m_val = 5.0 + 15.0 * (progress ** 2)
                cauchy_prob_val = 0.0
            else:
                # Original fixed parameters (MOSWO / MOSWO-V)
                mu_scale_val = 1.0
                c_factor_val = 1.0
                reinit_str_val = 0.0
                cr_val = self.CR
                eta_m_val = 20.0
                cauchy_prob_val = 0.0

            C = c_factor_val * 2.0 * (1.0 - progress)

            # Pre-compute crowding distance for archive
            n_arch = len(archive_pop)
            cd_map = self.calculate_crowding_distance(archive_objs, list(range(n_arch)))

            # Algorithm steps 1-3
            new_pop = pop.copy()

            for i in range(N):
                SW_star = self._select_leader(archive_pop, cd_map)
                SW_a = self._select_leader(archive_pop, cd_map)

                # Step 1: SEARCH PHASE (Eq 1 / Eq 2)
                if self.rng.random() > 0.5:
                    a_idx, b_idx = self.rng.choice(N, 2, replace=False)
                    r1 = self.rng.random()
                    rn = self.rng.normal(0, 1)
                    mu1 = rn * r1 * mu_scale_val
                    new_pop[i] = new_pop[i] + mu1 * (pop[a_idx] - pop[b_idx])
                else:
                    ell = self.rng.uniform(-2, -1)
                    B = 1.0 / (1.0 + np.exp(ell))
                    mu2 = B * np.cos(2 * np.pi * ell) * mu_scale_val
                    r2 = self.rng.random(self.n_dim)
                    c_idx = self.rng.integers(N)
                    new_pop[i] = pop[c_idx] + mu2 * (self.lb + r2 * span_bounds)

                new_pop[i] = self._clip(new_pop[i])

                # Step 2: FOLLOWING & ESCAPING PHASE (Eq 3 / Eq 4)
                r5 = self.rng.random(self.n_dim)
                dist_vec = SW_a - new_pop[i]
                k_dist = np.linalg.norm(dist_vec) / (span_norm + 1e-12)

                velocities[i] = w * velocities[i] + C * r5 * dist_vec

                if k_dist > 0.5:
                    ell = self.rng.uniform(-2, -1)
                    B = 1.0 / (1.0 + np.exp(ell))
                    mu2 = B * np.cos(2 * np.pi * ell) * mu_scale_val
                    r2 = self.rng.random(self.n_dim)
                    new_pop[i] = (new_pop[i]
                                   + mu2 * (self.lb + r2 * span_bounds)
                                   + 0.1 * velocities[i])
                else:
                    if self.rng.random() > 0.5:
                        new_pop[i] = new_pop[i] + C * (2 * r5 * dist_vec)
                    else:
                        scale_factor = k_dist * (1.0 - progress)
                        vc = 1.0 + self.rng.normal(0, scale_factor + 0.05, self.n_dim)
                        new_pop[i] = new_pop[i] * vc + (1.0 - progress) * velocities[i]

                new_pop[i] = self._clip(new_pop[i])

                # Step 3: NESTING PHASE (Eq 5)
                ell = self.rng.uniform(-2, -1)
                new_pop[i] = SW_star + np.cos(2 * np.pi * ell) * (SW_star - new_pop[i])
                new_pop[i] = self._clip(new_pop[i])

                # CROSSOVER MATING (Adaptive cr_rate from fuzzy controller)
                if self.rng.random() < cr_val:
                    mate = self._select_leader(archive_pop, cd_map)
                    j_rand = self.rng.integers(self.n_dim)
                    mask = self.rng.random(self.n_dim) < cr_val
                    mask[j_rand] = True
                    new_pop[i][mask] = mate[mask]

                # POLYNOMIAL MUTATION (Adaptive eta_m from fuzzy controller)
                new_pop[i] = self._polynomial_mutation(new_pop[i], eta_m=eta_m_val, prob=1.0 / self.n_dim)

            # Step 4: EVALUATION
            new_objs = self._evaluate(new_pop)

            # Elitist Cauchy Escape (controlled by fuzzy cauchy_prob_val or stagnation above g_excess_threshold)
            if (self.use_fuzzy
                    and self.use_cauchy
                    and (cauchy_prob_val > 0.04 or stagnant_gens >= self.stagnant_limit)
                    and g_excess_val > self.g_excess_threshold
                    and progress < 0.92):
                eff_cp = max(cauchy_prob_val, 0.10)
                new_pop, new_objs = self._cauchy_escape(
                    new_pop, new_objs, archive_pop, archive_objs, eff_cp, progress
                )

            # Elitist Partial Reinitialization & OBL Escape (controlled by fuzzy reinit_str_val)
            if g_excess_val < self.g_excess_threshold:
                stagnant_threshold = 9999
            elif g_excess_val < 0.55:
                stagnant_threshold = self.stagnant_limit + 1
            else:
                stagnant_threshold = self.stagnant_limit

            if reinit_cooldown > 0:
                reinit_cooldown -= 1

            if (self.use_fuzzy
                    and self.use_reinit
                    and (reinit_str_val > 0.15 or g_excess_val > self.g_excess_threshold)
                    and stagnant_gens >= stagnant_threshold
                    and progress < 0.90
                    and reinit_cooldown == 0):
                eff_str = max(reinit_str_val, 0.35)
                n_reinit = max(1, int(eff_str * 0.25 * N))
                new_pop, new_objs = self._partial_reinit(
                    new_pop, new_objs, n_reinit, archive_pop=archive_pop, archive_objs=archive_objs
                )
                if self.use_arch_div:
                    n_seeds = max(4, self.n_obj * n_reinit // 2)
                    archive_pop, archive_objs = self._archive_diversification_nd(
                        archive_pop, archive_objs, n_seeds
                    )
                stagnant_gens = 0
                reinit_cooldown = 3

            if (self.use_fuzzy
                    and self.use_reinit
                    and self.use_obl
                    and g_excess_val > self.g_excess_threshold
                    and t % self.obl_period == 0
                    and progress < 0.92):
                n_obl = max(1, int(self.obl_frac * N))
                cand = self._worst_indices(self.non_dominated_sort(new_objs), n_obl)
                new_pop, new_objs = self._obl_escape(
                    new_pop, new_objs, n_obl, candidates=cand,
                    archive_pop=archive_pop, archive_objs=archive_objs
                )

            archive_pop, archive_objs = self._update_archive(archive_pop, archive_objs, new_pop, new_objs)

            # Environmental selection with population size reduction
            combined_pop = np.vstack((pop, new_pop))
            combined_objs = np.vstack((objs, new_objs))

            current_target = N
            if t % reduction_period == 0 and t < self.max_iter:
                ratio = (self.max_iter - t) / self.max_iter
                current_target = max(n_min, int(n_min + (self.pop_size - n_min) * ratio))

            pop, objs = self._environmental_selection(combined_pop, combined_objs, current_target)

        return archive_pop, archive_objs, time.perf_counter() - start_time
