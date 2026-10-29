"""
FuzzyMOSWOController — 4-input, 6-output Mamdani fuzzy system for MOSWO.

Inputs:
  - iter_ratio   : normalised iteration progress t / T [0, 1]
  - diversity    : normalised population diversity in objective space [0, 1]
  - g_excess     : normalised distance of archive from ideal point [0, 1]
  - dec_diversity: normalised standard deviation of decision variables [0, 1]

Outputs:
  - mu_scale       : step size multiplier [0.1, 2.0]
  - c_factor       : follower pull factor [0.1, 1.0]
  - reinit_strength: partial reinitialisation fraction [0.0, 1.0]
  - cr_rate        : crossover rate [0.4, 0.98]
  - eta_m          : polynomial mutation distribution index [2.0, 25.0]
  - cauchy_prob    : Cauchy mutation jump probability [0.0, 0.25]
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyMOSWOController:
    """4-input, 6-output Mamdani Fuzzy Parameter Controller for MOSWO."""

    def __init__(self):
        u01   = np.linspace(0.0, 1.0, 101)
        u_mu  = np.linspace(0.1, 2.0, 101)
        u_c   = np.linspace(0.1, 1.0, 101)
        u_ri  = np.linspace(0.0, 1.0, 101)
        u_cr  = np.linspace(0.4, 0.98, 101)
        u_eta = np.linspace(2.0, 25.0, 101)
        u_cp  = np.linspace(0.0, 0.25, 101)

        # Antecedents (Inputs)
        self.iter_ratio    = ctrl.Antecedent(u01, 'iter_ratio')
        self.diversity     = ctrl.Antecedent(u01, 'diversity')
        self.g_excess      = ctrl.Antecedent(u01, 'g_excess')
        self.dec_diversity = ctrl.Antecedent(u01, 'dec_diversity')

        # Consequents (Outputs)
        self.mu_scale        = ctrl.Consequent(u_mu,  'mu_scale')
        self.c_factor        = ctrl.Consequent(u_c,   'c_factor')
        self.reinit_strength = ctrl.Consequent(u_ri,  'reinit_strength')
        self.cr_rate         = ctrl.Consequent(u_cr,  'cr_rate')
        self.eta_m           = ctrl.Consequent(u_eta, 'eta_m')
        self.cauchy_prob     = ctrl.Consequent(u_cp,  'cauchy_prob')

        # Membership functions for Antecedents
        self.iter_ratio['low']    = fuzz.trimf(u01, [0.0, 0.0, 0.4])
        self.iter_ratio['medium'] = fuzz.trimf(u01, [0.2, 0.5, 0.8])
        self.iter_ratio['high']   = fuzz.trimf(u01, [0.6, 1.0, 1.0])

        self.diversity['low']    = fuzz.trimf(u01, [0.0, 0.0, 0.35])
        self.diversity['medium'] = fuzz.trimf(u01, [0.2, 0.5, 0.8])
        self.diversity['high']   = fuzz.trimf(u01, [0.6, 1.0, 1.0])

        self.g_excess['low']    = fuzz.trapmf(u01, [0.0, 0.0, 0.20, 0.35])
        self.g_excess['medium'] = fuzz.trimf( u01, [0.25, 0.50, 0.75])
        self.g_excess['high']   = fuzz.trapmf(u01, [0.60, 0.75, 1.0, 1.0])

        self.dec_diversity['low']    = fuzz.trimf(u01, [0.0, 0.0, 0.35])
        self.dec_diversity['medium'] = fuzz.trimf(u01, [0.2, 0.5, 0.8])
        self.dec_diversity['high']   = fuzz.trimf(u01, [0.6, 1.0, 1.0])

        # Membership functions for Consequents
        self.mu_scale['small']  = fuzz.trimf(u_mu, [0.1, 0.1, 0.8])
        self.mu_scale['medium'] = fuzz.trimf(u_mu, [0.5, 1.0, 1.5])
        self.mu_scale['large']  = fuzz.trimf(u_mu, [1.2, 2.0, 2.0])

        self.c_factor['small']  = fuzz.trimf(u_c, [0.1, 0.1, 0.45])
        self.c_factor['medium'] = fuzz.trimf(u_c, [0.3, 0.55, 0.8])
        self.c_factor['high']   = fuzz.trimf(u_c, [0.6, 1.0, 1.0])

        self.reinit_strength['none']   = fuzz.trapmf(u_ri, [0.0, 0.0, 0.10, 0.20])
        self.reinit_strength['mild']   = fuzz.trimf( u_ri, [0.15, 0.35, 0.55])
        self.reinit_strength['strong'] = fuzz.trapmf(u_ri, [0.55, 0.75, 1.0, 1.0])

        self.cr_rate['low']    = fuzz.trimf(u_cr, [0.4, 0.5, 0.7])
        self.cr_rate['medium'] = fuzz.trimf(u_cr, [0.65, 0.80, 0.90])
        self.cr_rate['high']   = fuzz.trimf(u_cr, [0.85, 0.95, 0.98])

        self.eta_m['low']    = fuzz.trimf(u_eta, [2.0, 3.0, 8.0])
        self.eta_m['medium'] = fuzz.trimf(u_eta, [6.0, 12.0, 18.0])
        self.eta_m['high']   = fuzz.trimf(u_eta, [15.0, 22.0, 25.0])

        self.cauchy_prob['none'] = fuzz.trapmf(u_cp, [0.0, 0.0, 0.02, 0.05])
        self.cauchy_prob['mild'] = fuzz.trimf( u_cp, [0.03, 0.08, 0.15])
        self.cauchy_prob['high'] = fuzz.trapmf(u_cp, [0.10, 0.18, 0.25, 0.25])

        ir = self.iter_ratio
        dv = self.diversity
        gx = self.g_excess
        dd = self.dec_diversity

        ms = self.mu_scale
        cf = self.c_factor
        ri = self.reinit_strength
        cr = self.cr_rate
        eta = self.eta_m
        cp = self.cauchy_prob

        # Rule base
        rules = [
            # gx = LOW (Near True PF: fine exploitation, standard CR, high eta_m, zero Cauchy)
            ctrl.Rule(gx['low'] & ir['low'],
                      (ms['medium'], cf['medium'], ri['none'], cr['medium'], eta['medium'], cp['none'])),
            ctrl.Rule(gx['low'] & ir['medium'],
                      (ms['small'], cf['high'], ri['none'], cr['medium'], eta['high'], cp['none'])),
            ctrl.Rule(gx['low'] & ir['high'],
                      (ms['small'], cf['high'], ri['none'], cr['medium'], eta['high'], cp['none'])),

            # gx = MEDIUM (Mid-run normal exploration)
            ctrl.Rule(gx['medium'] & dv['high'],
                      (ms['medium'], cf['medium'], ri['none'], cr['medium'], eta['medium'], cp['none'])),
            ctrl.Rule(gx['medium'] & dv['medium'],
                      (ms['medium'], cf['medium'], ri['none'], cr['medium'], eta['medium'], cp['none'])),
            ctrl.Rule(gx['medium'] & dv['low'],
                      (ms['large'], cf['small'], ri['mild'], cr['medium'], eta['medium'], cp['mild'])),

            # gx = HIGH (Stuck in local basin)
            ctrl.Rule(gx['high'] & dd['low'],
                      (ms['large'], cf['small'], ri['strong'], cr['high'], eta['low'], cp['high'])),
            ctrl.Rule(gx['high'] & dd['medium'],
                      (ms['large'], cf['small'], ri['mild'], cr['high'], eta['low'], cp['mild'])),
            ctrl.Rule(gx['high'] & dd['high'],
                      (ms['medium'], cf['medium'], ri['none'], cr['medium'], eta['medium'], cp['none'])),
        ]

        self._in_vars  = [self.iter_ratio, self.diversity, self.g_excess, self.dec_diversity]
        self._out_vars = [self.mu_scale, self.c_factor, self.reinit_strength, self.cr_rate, self.eta_m, self.cauchy_prob]
        in_index  = {v.label: k for k, v in enumerate(self._in_vars)}
        out_index = {v.label: k for k, v in enumerate(self._out_vars)}
        self._in_terms  = [list(v.terms.keys()) for v in self._in_vars]
        self._out_terms = [list(v.terms.keys()) for v in self._out_vars]
        self._in_univ   = [v.universe for v in self._in_vars]
        self._out_univ  = [v.universe for v in self._out_vars]
        self._in_mfs    = [{name: v[name].mf for name in self._in_terms[k]} for k, v in enumerate(self._in_vars)]
        self._out_mfs   = [{name: v[name].mf for name in self._out_terms[k]} for k, v in enumerate(self._out_vars)]

        self._rule_table = []
        self.rule_labels = [
            "R1: gx=LOW  & ir=LOW    -> Exploit-Balanced (ms=med, cf=med, ri=none, cr=med, eta=med, cp=none)",
            "R2: gx=LOW  & ir=MEDIUM -> Exploit-Fine     (ms=sml, cf=high, ri=none, cr=med, eta=high, cp=none)",
            "R3: gx=LOW  & ir=HIGH   -> Exploit-Strong   (ms=sml, cf=high, ri=none, cr=med, eta=high, cp=none)",
            "R4: gx=MED  & dv=HIGH   -> Normal-Diverse   (ms=med, cf=med, ri=none, cr=med, eta=med, cp=none)",
            "R5: gx=MED  & dv=MEDIUM -> Normal-Balanced  (ms=med, cf=med, ri=none, cr=med, eta=med, cp=none)",
            "R6: gx=MED  & dv=LOW    -> Escape-Mild      (ms=lrg, cf=sml, ri=mild, cr=med, eta=med, cp=mild)",
            "R7: gx=HIGH & dd=LOW    -> Escape-Strong    (ms=lrg, cf=sml, ri=strg, cr=high, eta=low, cp=high)",
            "R8: gx=HIGH & dd=MEDIUM -> Escape-Moderate  (ms=lrg, cf=sml, ri=mild, cr=high, eta=low, cp=mild)",
            "R9: gx=HIGH & dd=HIGH   -> Early-Search     (ms=med, cf=med, ri=none, cr=med, eta=med, cp=none)",
        ]
        for r in rules:
            ants = [(in_index[t.parent.label], t.label) for t in r.antecedent_terms]
            cons = [(out_index[wt.term.parent.label], wt.term.label) for wt in r.consequent]
            self._rule_table.append((ants, cons))

        self.reset_stats()

    def reset_stats(self):
        """Reset rule activation statistics for sensitivity & pruning analysis."""
        self.rule_fire_counts = np.zeros(len(self._rule_table), dtype=int)
        self.rule_fire_strength_sum = np.zeros(len(self._rule_table), dtype=float)
        self.total_evals = 0
        self.active_rules_mask = np.ones(len(self._rule_table), dtype=bool)

    def get_rule_stats(self):
        """Return firing frequency (%) and mean firing strength per rule."""
        denom = max(1, self.total_evals)
        stats = []
        for idx, label in enumerate(self.rule_labels):
            stats.append({
                "rule_id": f"R{idx + 1}",
                "description": label,
                "fire_count": int(self.rule_fire_counts[idx]),
                "fire_freq_pct": float(self.rule_fire_counts[idx] / denom * 100.0),
                "mean_strength": float(self.rule_fire_strength_sum[idx] / denom),
                "active": bool(self.active_rules_mask[idx]),
            })
        return stats

    def compute(self, iter_ratio: float, diversity: float, g_excess: float, dec_diversity: float):
        """
        Evaluate fuzzy system for given inputs.

        Parameters
        ----------
        iter_ratio    : t / T in [0, 1]
        diversity     : objective-space diversity in [0, 1]
        g_excess      : normalized distance from true PF in [0, 1]
        dec_diversity : decision-space diversity in [0, 1]

        Returns
        -------
        tuple of (mu_scale, c_factor, reinit_strength, cr_rate, eta_m, cauchy_prob)
        """
        inputs = [
            float(np.clip(iter_ratio, 0.0, 1.0)),
            float(np.clip(diversity, 0.0, 1.0)),
            float(np.clip(g_excess, 0.0, 1.0)),
            float(np.clip(dec_diversity, 0.0, 1.0)),
        ]
        self.total_evals += 1

        # 1. Fuzzification
        mu_in = []
        for k in range(len(self._in_vars)):
            x = inputs[k]
            univ = self._in_univ[k]
            mfs = self._in_mfs[k]
            mu_in.append({name: float(np.interp(x, univ, mfs[name]))
                          for name in self._in_terms[k]})

        # 2. Rule evaluation and aggregation
        act = [{name: 0.0 for name in self._out_terms[k]} for k in range(len(self._out_vars))]
        for r_idx, (ants, cons) in enumerate(self._rule_table):
            if not self.active_rules_mask[r_idx]:
                continue
            a = min(mu_in[vi][tn] for vi, tn in ants)
            if a <= 0.0:
                continue
            self.rule_fire_counts[r_idx] += 1
            self.rule_fire_strength_sum[r_idx] += a
            for vo, tn in cons:
                if a > act[vo][tn]:
                    act[vo][tn] = a

        # 3. Defuzzification (centroid)
        outs = []
        for k in range(len(self._out_vars)):
            univ = self._out_univ[k]
            mfs = self._out_mfs[k]
            agg = np.zeros_like(univ, dtype=float)
            for tn, a in act[k].items():
                if a > 0.0:
                    agg = np.fmax(agg, np.fmin(a, mfs[tn]))
            total = agg.sum()
            if total == 0.0:
                outs.append(float(univ.mean()))
            else:
                outs.append(float(fuzz.defuzz(univ, agg, 'centroid')))

        return tuple(outs)


# Backward-compatibility alias
EnhancedFuzzyMOSWOController = FuzzyMOSWOController
