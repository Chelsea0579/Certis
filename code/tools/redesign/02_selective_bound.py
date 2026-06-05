"""Distribution-free Hoeffding selective-risk bound (NOT PAC-Bayes).

Given a verifier h fixed on train/cal_fit (before opening cert), evaluate on cert:
  For each threshold t = (a, 1-a) in fixed grid G (|G|=K=7):
    n_t = #{i : q(X_i) <= a OR q(X_i) >= 1-a}  -- accepted count
    err_t = #{i accepted : h(X_i) != Y_i}      -- accepted errors
    R_hat_t = err_t / n_t                          -- empirical accepted-error rate
    C_hat_t = n_t / m                              -- empirical coverage

  With probability >= 1 - delta_R (using union bound over K thresholds):
    R_t <= R_hat_t + sqrt(log(K/delta_R) / (2 n_t))    -- one-sided Hoeffding

  With probability >= 1 - delta_C (using union bound over K):
    C_t >= C_hat_t - sqrt(log(K/delta_C) / (2 m))

  Both jointly hold with probability >= 1 - (delta_R + delta_C) (second union bound).

Selection rule: t* = argmax_t C_hat_t subject to U_R(t) <= 0.08 AND L_C(t) >= 0.90.

This is the *only* selective bound permitted by the redesign. NO PAC-Bayes term.
"""
import argparse, math, json, os
import numpy as np
import pandas as pd

THRESHOLD_GRID = [(0.50, 0.50), (0.45, 0.55), (0.40, 0.60), (0.35, 0.65),
                  (0.30, 0.70), (0.25, 0.75), (0.20, 0.80)]
K = len(THRESHOLD_GRID)


def hoeffding_upper(rhat: float, n: int, delta: float, K_thresholds: int = K) -> float:
    """One-sided Hoeffding upper bound on R_t with union bound over K thresholds."""
    if n == 0:
        return 1.0
    rad = math.sqrt(math.log(K_thresholds / delta) / (2 * n))
    return min(1.0, rhat + rad)


def hoeffding_lower_coverage(chat: float, m: int, delta: float, K_thresholds: int = K) -> float:
    """One-sided Hoeffding lower bound on C_t with union bound over K thresholds."""
    if m == 0:
        return 0.0
    rad = math.sqrt(math.log(K_thresholds / delta) / (2 * m))
    return max(0.0, chat - rad)


def certify(cert_df: pd.DataFrame, prob_col: str, label_col: str,
            delta_R: float = 0.025, delta_C: float = 0.025,
            U_R_target: float = 0.08, L_C_target: float = 0.90):
    """Run certification over the fixed grid on the certification set.

    cert_df must have:
      - prob_col: calibrated P(y=1 | features)
      - label_col: ground-truth 0/1

    Returns a dict with per-threshold bounds and the selected t*.
    """
    m = len(cert_df)
    probs = cert_df[prob_col].values
    y = cert_df[label_col].values

    rows = []
    for (a, b) in THRESHOLD_GRID:
        accepted_mask = (probs <= a) | (probs >= b)
        n_t = int(accepted_mask.sum())
        if n_t == 0:
            continue
        accepted_probs = probs[accepted_mask]
        accepted_y = y[accepted_mask]
        accepted_pred = (accepted_probs >= 0.5).astype(int)
        err = int((accepted_pred != accepted_y).sum())
        rhat = err / n_t
        chat = n_t / m
        ur = hoeffding_upper(rhat, n_t, delta_R)
        lc = hoeffding_lower_coverage(chat, m, delta_C)
        rows.append({
            "t_lo": a, "t_hi": b, "n_t": n_t, "m": m,
            "err": err, "rhat": rhat, "chat": chat,
            "U_R": ur, "L_C": lc,
            "qualifies": (ur <= U_R_target) and (lc >= L_C_target),
        })
    qualifying = [r for r in rows if r["qualifies"]]
    if qualifying:
        # Pick highest empirical coverage among qualifying - equivalent to largest accepted set
        t_star = max(qualifying, key=lambda r: r["chat"])
    else:
        t_star = None
    return {"grid": rows, "t_star": t_star,
            "U_R_target": U_R_target, "L_C_target": L_C_target,
            "delta_R": delta_R, "delta_C": delta_C, "K": K}


# ----- Unit tests -----
def unit_tests():
    results = {}
    # Test 1: Hoeffding closed-form
    # rad = sqrt( log(K/delta) / (2n) )
    n, K_, d = 2000, 7, 0.025
    expected = math.sqrt(math.log(K_ / d) / (2 * n))
    got = hoeffding_upper(0.0, n, d, K_) - 0.0
    results["hoeffding_radius_close_to_expected"] = {
        "expected": expected, "got": got, "diff": abs(expected - got)
    }
    assert abs(expected - got) < 1e-10, f"radius wrong: {expected} vs {got}"

    # Test 2: At n=2000, K=7, delta=0.025: rad ~ 0.0376
    results["radius_at_n2000_K7_d025"] = round(got, 4)
    assert abs(got - 0.0376) < 0.001, f"radius mismatch: {got}"

    # Test 3: At n=500, K=7, delta=0.025: rad ~ 0.0751
    n2 = 500
    rad2 = math.sqrt(math.log(K_ / d) / (2 * n2))
    results["radius_at_n500_K7_d025"] = round(rad2, 4)
    assert abs(rad2 - 0.0751) < 0.001, f"500-cell radius mismatch: {rad2}"

    # Test 4: Synthetic Bernoulli check - random verifier with TRUE accepted-error 0.04
    rng = np.random.default_rng(0)
    n_trials = 10000
    true_R = 0.04
    losses = rng.binomial(1, true_R, n_trials)
    empirical = losses.mean()
    ur = hoeffding_upper(empirical, n_trials, 0.025, K_)
    results["synth_empirical"] = round(empirical, 4)
    results["synth_upper_bound"] = round(ur, 4)
    # Bound must hold (with high probability - sanity check)
    assert ur >= true_R, f"bound failed: UR={ur} < true_R={true_R}"

    return results


def main(args):
    ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    OUT = os.path.join(ROOT, "outputs", "redesign")

    # Run unit tests
    test_results = unit_tests()
    json.dump(test_results, open(os.path.join(OUT, "w1_bound_unit_tests.json"), "w"), indent=2)
    print("unit tests PASS:")
    for k, v in test_results.items():
        print(f"  {k}: {v}")

    # Write bound specification doc
    spec = f"""# Distribution-Free Hoeffding Selective-Risk Bound

## Statement

Let h, q be a verifier (binary prediction + calibrated probability) fixed before
opening the certification set `cert` of size m. For each threshold t = (a, 1-a)
in the fixed grid G of size K = {K}:

    A_t(X) = 1{{q(X) <= a OR q(X) >= 1 - a}}   (accepted indicator)
    L(X, Y) = 1{{h(X) != Y}}                  (error indicator)
    R_t = Pr(L = 1 | A_t = 1)                (selective error)
    C_t = Pr(A_t = 1)                        (coverage)

Empirical estimates on cert:
    n_t = sum_i A_t(X_i)
    R_hat_t = sum_i A_t(X_i) L(X_i, Y_i) / n_t   (if n_t > 0)
    C_hat_t = n_t / m

**One-sided Hoeffding + union bound over K thresholds.**
With probability >= 1 - delta_R (delta_R = {0.025}), simultaneously for every t  in  G:

    R_t <= U_R(t) = R_hat_t + sqrt(log(K/delta_R) / (2 n_t))

With probability >= 1 - delta_C (delta_C = {0.025}), simultaneously for every t  in  G:

    C_t >= L_C(t) = C_hat_t - sqrt(log(K/delta_C) / (2 m))

Both hold jointly with probability >= 1 - 0.05 by a second union bound.

## Selection rule (the *only* allowed rule)

    t* = argmax_{{t  in  G}} C_hat_t  s.t.  U_R(t) <= 0.08  AND  L_C(t) >= 0.90.

If no threshold qualifies, the certified result is *failure*; no fallback to
lower-coverage cells is permitted.

## Skeptical consequence

At n_t = 2000, K = 7, delta_R = 0.025: Hoeffding radius = {round(test_results['radius_at_n2000_K7_d025'], 4)}.
So U_R <= 0.08 needs empirical R_hat_t <= 0.0424.

At n_t = 500: Hoeffding radius = {round(test_results['radius_at_n500_K7_d025'], 4)}.
So U_R <= 0.08 needs empirical R_hat_t <= 0.0049. **500-cand cells cannot support this.**

## Forbidden terminology

The paper text MUST NOT use "PAC-Bayes" anywhere. The bound is "distribution-free
Hoeffding selective-risk upper bound with explicit K-grid union-bound adjustment".

## File locations

- Implementation: `_tools/redesign/02_selective_bound.py`
- Unit tests: `outputs/redesign/w1_bound_unit_tests.json`
"""
    open(os.path.join(OUT, "w1_bound_spec.md"), "w", encoding="utf-8").write(spec)
    print(f"\nwrote w1_bound_spec.md (theory) and w1_bound_unit_tests.json (unit tests)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    args = p.parse_args()
    main(args)
