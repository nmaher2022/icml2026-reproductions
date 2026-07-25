# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Numerical audit harness for ICML 2026 #9559 (OpenReview tBS3uBG6Pv):
"Improved Dynamic Algorithm for Non-monotone Submodular Maximization under
Cardinality Constraint".

The paper text is not publicly accessible, so the algorithms here are
*reconstructions* built from the stated guarantees and the standard toolbox
of the surrounding literature:

  - A dynamic threshold-bucket greedy structure (in the spirit of
    Lattanzi et al. NeurIPS'20 / Monemizadeh NeurIPS'20 / Banihashem et al.),
    with geometric thresholds tau_j = M/(1+eps)^j down to eps*M/(2k).
  - Variant A1 ("sampled single pass"): each qualifying element is accepted
    with probability 1/2 (the classic subsampling correction that converts
    monotone-style threshold guarantees into non-monotone guarantees),
    answer = best of {S, best singleton}.  Audited against Claim 1 (0.262).
  - Variant A2 ("two-pass + offline combine"): pass 1 builds S1 (p=1),
    pass 2 builds S2 on V \\ S1, then an offline randomized greedy
    (Buchbinder-Feldman-Naor-Schwartz style) runs on the buffer S1 u S2;
    answer = best candidate.  This mirrors the framework behind the
    streaming-optimal 0.2779 ratio (Alaluf et al.), matching Claim 3 (0.277).
  - An ablated control A0 (p=1, single pass, no combine) and a
    non-submodular control oracle, to show the audit can detect failures
    when the theorem's conditions are relaxed.

Every oracle call is counted; per-update query costs are recorded for the
Claim 2 scaling audit.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------
# Oracles (all with query counting).  Ground set elements are integer ids.
# ----------------------------------------------------------------------------

class CountingOracle:
    """Wraps a set function; counts distinct f(S) evaluations."""

    def __init__(self):
        self.calls = 0

    def value(self, s: frozenset) -> float:
        self.calls += 1
        return self._value(s)

    def _value(self, s: frozenset) -> float:  # pragma: no cover
        raise NotImplementedError


class MaxCutOracle(CountingOracle):
    """f(S) = total weight of edges crossing (S, V\\S).  Non-monotone, submodular,
    symmetric.  Restricted to the currently-present vertex set by the caller."""

    def __init__(self, n, p, rng, weighted=True):
        super().__init__()
        self.n = n
        self.adj = [dict() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    w = rng.uniform(0.2, 1.0) if weighted else 1.0
                    self.adj[i][j] = w
                    self.adj[j][i] = w

    def _value(self, s):
        if not s:
            return 0.0
        tot = 0.0
        for v in s:
            for u, w in self.adj[v].items():
                if u not in s:
                    tot += w
        return tot


class DiCutOracle(CountingOracle):
    """f(S) = weight of arcs from S to V\\S in a random digraph.  Non-monotone,
    submodular, asymmetric."""

    def __init__(self, n, p, rng):
        super().__init__()
        self.n = n
        self.out = [dict() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and rng.random() < p:
                    self.out[i][j] = rng.uniform(0.2, 1.0)

    def _value(self, s):
        tot = 0.0
        for v in s:
            for u, w in self.out[v].items():
                if u not in s:
                    tot += w
        return tot


class CoverageCostOracle(CountingOracle):
    """f(S) = weighted coverage(S) - lam * |S|.  Non-monotone (for lam > 0),
    submodular.  Clipped at 0 only implicitly (values may go negative for
    large S; the algorithms only ever benefit from nonneg marginals)."""

    def __init__(self, n, universe, rng, lam=None):
        super().__init__()
        self.n = n
        self.wts = [rng.uniform(0.5, 1.5) for _ in range(universe)]
        self.sets = [frozenset(rng.sample(range(universe), rng.randint(2, max(3, universe // 4))))
                     for _ in range(n)]
        avg_gain = sum(sum(self.wts[e] for e in st) for st in self.sets) / n
        self.lam = lam if lam is not None else 0.6 * avg_gain

    def _value(self, s):
        if not s:
            return 0.0
        cov = set()
        for v in s:
            cov |= self.sets[v]
        return sum(self.wts[e] for e in cov) - self.lam * len(s)


class SupermodularControl(CountingOracle):
    """CONTROL: violates submodularity.  f(S) = base linear value + a
    supermodular hidden-clique bonus that only fires once |S ∩ C| is large.
    Threshold-greedy-type algorithms cannot see the bonus coming, so their
    ratio against OPT should fall below the claimed bounds on many seeds."""

    def __init__(self, n, rng, clique_frac=0.4, bonus=25.0):
        super().__init__()
        self.n = n
        m = max(3, int(n * clique_frac))
        self.clique = frozenset(rng.sample(range(n), m))
        self.base = [rng.uniform(0.5, 1.0) for _ in range(n)]
        # decoys: attractive singletons outside the clique
        for v in range(n):
            if v not in self.clique:
                self.base[v] += rng.uniform(1.0, 2.0)
        self.bonus = bonus
        self.need = max(2, m - 1)

    def _value(self, s):
        v = sum(self.base[x] for x in s)
        inter = len(s & self.clique)
        if inter >= self.need:
            v += self.bonus * (inter - self.need + 1)
        return v


def make_oracle(kind, n, rng):
    if kind == "maxcut":
        return MaxCutOracle(n, p=0.35, rng=rng)
    if kind == "dicut":
        return DiCutOracle(n, p=0.25, rng=rng)
    if kind == "covcost":
        return CoverageCostOracle(n, universe=max(12, 2 * n), rng=rng)
    if kind == "supermod":
        return SupermodularControl(n, rng=rng)
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# Exact OPT by enumeration (small n only).
# ----------------------------------------------------------------------------

def brute_force_opt(oracle, ground, k):
    """Max f(S) over |S| <= k.  Does not count oracle calls (uses _value)."""
    best, best_set = 0.0, frozenset()
    ground = sorted(ground)
    for r in range(1, k + 1):
        for combo in itertools.combinations(ground, r):
            v = oracle._value(frozenset(combo))
            if v > best:
                best, best_set = v, frozenset(combo)
    return best, best_set


# ----------------------------------------------------------------------------
# Dynamic threshold-bucket algorithm (reconstruction).
# ----------------------------------------------------------------------------

@dataclass
class PassState:
    """One threshold-greedy pass: solution list + per-level checkpoints."""
    sol: list = field(default_factory=list)          # elements in pick order
    sol_vals: list = field(default_factory=list)     # f(prefix) after each pick
    pick_level: list = field(default_factory=list)   # level index of each pick
    stop_level: int = 0                              # first level not fully processed


class DynamicSubmod:
    """Reconstructed dynamic algorithm.

    variant='A1'  : sampled single pass (p=1/2) + best singleton     (Claim 1)
    variant='A2'  : two passes (p=1) + offline randomized greedy on
                    the buffer S1 u S2                               (Claim 3)
    variant='A0'  : ablated control - single pass, p=1, no combine
    """

    def __init__(self, oracle, k, eps, variant="A1", seed=0):
        self.f = oracle
        self.k = k
        self.eps = eps
        self.variant = variant
        self.rng = random.Random(seed)
        self.V = {}            # id -> random priority
        self.singleton = {}    # id -> f({v})
        self.passes = [PassState()] if variant in ("A0", "A1") else [PassState(), PassState()]
        self.p_sample = 0.5 if variant == "A1" else 1.0
        self._coins = {}       # (elem, pass) -> bool, fixed per element for stability
        self._cached_answer = None

    # -- levels ---------------------------------------------------------------
    def _levels(self, M):
        """Geometric thresholds from M down to eps*M/(2k)."""
        if M <= 0:
            return [0.0]
        taus = []
        tau = M
        floor = self.eps * M / (2 * self.k)
        while tau >= floor:
            taus.append(tau)
            tau /= (1 + self.eps)
        taus.append(floor)
        return taus

    def _coin(self, v, pi):
        key = (v, pi)
        if key not in self._coins:
            self._coins[key] = (self.rng.random() < self.p_sample)
        return self._coins[key]

    # -- full / suffix rebuild ------------------------------------------------
    def _build_pass(self, pi, exclude, from_level=0):
        """(Re)build pass pi from threshold level `from_level`.
        Elements are scanned in random-priority order within each level."""
        st = self.passes[pi]
        if not self.V:
            self.passes[pi] = PassState()
            return
        M = max(self.singleton[v] for v in self.V)
        taus = self._levels(M)
        if from_level == 0:
            st.sol, st.sol_vals, st.pick_level = [], [], []
        else:
            # truncate solution to picks made strictly below level `from_level`
            keep = [i for i, lv in enumerate(st.pick_level) if lv < from_level]
            n_keep = len(keep)
            st.sol = st.sol[:n_keep]
            st.sol_vals = st.sol_vals[:n_keep]
            st.pick_level = st.pick_level[:n_keep]
        cur = frozenset(st.sol)
        cur_val = st.sol_vals[-1] if st.sol else 0.0
        # candidates in priority order (this scan is what real per-update
        # machinery avoids; rebuilds are rare, cost is measured empirically)
        cands = sorted((x for x in self.V if x not in exclude and x not in cur),
                       key=lambda x: self.V[x])
        for j in range(from_level, len(taus)):
            st.stop_level = j
            if len(st.sol) >= self.k:
                break
            tau = taus[j]
            nxt = []
            for v in cands:
                if len(st.sol) >= self.k:
                    break
                if self.singleton[v] < tau:
                    nxt.append(v)
                    continue
                gain = self.f.value(cur | {v}) - cur_val
                if gain >= tau:
                    if self._coin(v, pi):
                        st.sol.append(v)
                        st.pick_level.append(j)
                        cur = cur | {v}
                        cur_val += gain
                        st.sol_vals.append(cur_val)
                    # coin=False: element is *rejected for this pass* (sampled out)
                else:
                    nxt.append(v)
            cands = [x for x in nxt if x not in cur]
        self._cached_answer = None

    def _rebuild_all(self, from_pass=0, from_level=0):
        for pi in range(from_pass, len(self.passes)):
            excl = frozenset(self.passes[pi - 1].sol) if pi > 0 else frozenset()
            self._build_pass(pi, excl, from_level if pi == from_pass else 0)

    # -- updates --------------------------------------------------------------
    def insert(self, v):
        self.V[v] = self.rng.random()
        sv = self.f.value(frozenset([v]))
        self.singleton[v] = sv
        # Does v qualify at a level at/above where some pass stopped, or beat
        # the current max singleton (levels shift)?  Then rebuild lazily.
        need = None
        for pi, st in enumerate(self.passes):
            if not self.V or not st.sol and st.stop_level == 0:
                need = (pi, 0)
                break
            M = max(self.singleton[x] for x in self.V)
            taus = self._levels(M)
            j = min(st.stop_level, len(taus) - 1)
            cur = frozenset(st.sol)
            cur_val = st.sol_vals[-1] if st.sol else 0.0
            if len(st.sol) < self.k:
                gain = self.f.value(cur | {v}) - cur_val
                if gain >= taus[j]:
                    need = (pi, j)
                    break
            elif sv >= taus[0]:
                # new global-max singleton while solution is full: thresholds
                # shifted, rebuild from the top
                need = (pi, 0)
                break
        if need is not None:
            self._rebuild_all(from_pass=need[0], from_level=need[1])
        else:
            self._cached_answer = None  # answer may still improve via singleton

    def delete(self, v):
        self.V.pop(v, None)
        self.singleton.pop(v, None)
        hit = None
        for pi, st in enumerate(self.passes):
            if v in st.sol:
                i = st.sol.index(v)
                hit = (pi, st.pick_level[i])
                break
        if hit is not None:
            self._rebuild_all(from_pass=hit[0], from_level=hit[1])
        else:
            self._cached_answer = None

    # -- answer extraction ----------------------------------------------------
    def _offline_random_greedy(self, buffer, repeats=3):
        """Buchbinder et al.-style randomized greedy on a small buffer,
        best of `repeats` runs.  ~1/e offline guarantee for non-monotone."""
        best_v, best_s = 0.0, frozenset()
        buffer = list(buffer)
        for _ in range(repeats):
            s, sval = frozenset(), 0.0
            for _ in range(min(self.k, len(buffer))):
                gains = []
                for v in buffer:
                    if v in s:
                        continue
                    g = self.f.value(s | {v}) - sval
                    gains.append((g, v))
                gains.sort(reverse=True)
                top = [gv for gv in gains[:self.k] if gv[0] > 0]
                if not top:
                    break
                g, v = self.rng.choice(gains[:max(1, min(self.k, len(gains)))])
                if g > 0:
                    s = s | {v}
                    sval += g
            if sval > best_v:
                best_v, best_s = sval, s
        return best_v, best_s

    def answer(self):
        """Current solution value (and set)."""
        if self._cached_answer is not None:
            return self._cached_answer
        cands = []
        for st in self.passes:
            sv = st.sol_vals[-1] if st.sol else 0.0
            cands.append((sv, frozenset(st.sol)))
        if self.V:
            bs = max(self.V, key=lambda x: self.singleton[x])
            cands.append((self.singleton[bs], frozenset([bs])))
        if self.variant == "A2":
            buffer = set().union(*(set(st.sol) for st in self.passes)) if self.passes else set()
            if buffer:
                cands.append(self._offline_random_greedy(buffer))
        self._cached_answer = max(cands, key=lambda t: t[0]) if cands else (0.0, frozenset())
        return self._cached_answer


# ----------------------------------------------------------------------------
# Streams
# ----------------------------------------------------------------------------

def run_stream(oracle, n, k, eps, variant, seed, n_ops, adversarial=False,
               check_every=1, brute=True, collect_updates=False, alg_seed=None):
    """Insert all n elements, then n_ops mixed delete/insert operations.
    Returns per-checkpoint ratios and per-update query counts.
    `alg_seed` decouples the algorithm's coins from the stream randomness so
    E[f(S_t)] can be estimated by replication over alg seeds."""
    rng = random.Random(seed + 7919)
    alg = DynamicSubmod(oracle, k, eps, variant=variant,
                        seed=seed if alg_seed is None else alg_seed)
    present = set()
    absent = set(range(n))
    ratios = []
    upd_queries = []
    t0 = time.time()

    def checkpoint():
        val, _ = alg.answer()
        if brute:
            opt, _ = brute_force_opt(oracle, present, k)
            if opt > 1e-12:
                ratios.append(val / opt)

    order = list(range(n))
    rng.shuffle(order)
    for i, v in enumerate(order):
        q0 = oracle.calls
        alg.insert(v)
        present.add(v)
        absent.discard(v)
        _ = alg.answer()
        if collect_updates:
            upd_queries.append(("ins", oracle.calls - q0))
        if brute and (i % check_every == 0):
            checkpoint()

    for t in range(n_ops):
        if t % 2 == 0 and len(present) > max(k + 2, n // 3):
            if adversarial and alg.passes[0].sol:
                v = rng.choice(alg.passes[0].sol)  # target the solution
            else:
                v = rng.choice(sorted(present))
            q0 = oracle.calls
            alg.delete(v)
            present.discard(v)
            absent.add(v)
            if collect_updates:
                upd_queries.append(("del", oracle.calls - q0))
        elif absent:
            v = rng.choice(sorted(absent))
            q0 = oracle.calls
            alg.insert(v)
            present.add(v)
            absent.discard(v)
            if collect_updates:
                upd_queries.append(("ins", oracle.calls - q0))
        _ = alg.answer()
        if brute and (t % check_every == 0):
            checkpoint()

    return {
        "ratios": ratios,
        "min_ratio": min(ratios) if ratios else None,
        "mean_ratio": float(np.mean(ratios)) if ratios else None,
        "upd_queries": upd_queries,
        "total_queries": oracle.calls,
        "wall_s": time.time() - t0,
    }


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------

def exp_ratio(args):
    """Claims 1/3: empirical approximation ratio vs exact OPT under dynamic
    streams, plus controls."""
    rows = []
    configs = []
    for kind in args.oracles.split(","):
        for k in [int(x) for x in args.ks.split(",")]:
            for seed in range(args.seed_base, args.seed_base + args.seeds):
                configs.append((kind, k, seed))
    for kind, k, seed in configs:
        for adversarial in ([False, True] if args.adversarial else [False]):
            reps = []
            for r in range(args.alg_seeds):
                rng = random.Random(1000 * seed + k)
                oracle = make_oracle(kind, args.n, rng)  # same instance
                res = run_stream(oracle, args.n, k, args.eps, args.variant, seed,
                                 n_ops=args.ops, adversarial=adversarial,
                                 check_every=args.check_every, brute=True,
                                 alg_seed=5000 + 131 * r)
                reps.append(res)
            # single-run stats (replica 0) + expectation stats across replicas
            res0 = reps[0]
            mean_min = float(np.mean([rp["min_ratio"] for rp in reps]))
            exp_min = None
            if not adversarial and len({len(rp["ratios"]) for rp in reps}) == 1:
                # streams identical across replicas -> align checkpoints,
                # audit min_t E_coins[f(S_t)]/OPT_t
                mat = np.array([rp["ratios"] for rp in reps])
                exp_min = float(np.min(mat.mean(axis=0)))
            rows.append({
                "variant": args.variant, "oracle": kind, "n": args.n, "k": k,
                "eps": args.eps, "seed": seed,
                "adversarial": int(adversarial), "alg_seeds": args.alg_seeds,
                "min_ratio": res0["min_ratio"], "mean_ratio": res0["mean_ratio"],
                "mean_min_ratio": mean_min, "exp_min_ratio": exp_min,
                "n_checkpoints": len(res0["ratios"]),
                "total_queries": res0["total_queries"], "wall_s": res0["wall_s"],
            })
            print(f"[ratio] var={args.variant} f={kind} n={args.n} k={k} "
                  f"seed={seed} adv={int(adversarial)} "
                  f"min={res0['min_ratio']:.4f} mean={res0['mean_ratio']:.4f} "
                  f"mean_min={mean_min:.4f} "
                  f"exp_min={exp_min if exp_min is None else round(exp_min, 4)}",
                  flush=True)
    write_csv(args.out, rows)
    mins = [r["min_ratio"] for r in rows]
    emins = [r["exp_min_ratio"] for r in rows if r["exp_min_ratio"] is not None]
    print(f"SUMMARY variant={args.variant} instances={len(rows)} "
          f"global_min_single={min(mins):.4f} "
          + (f"global_min_expected={min(emins):.4f} " if emins else "")
          + f"mean_of_means={np.mean([r['mean_ratio'] for r in rows]):.4f}",
          flush=True)


def exp_scaling(args):
    """Claim 2: per-update oracle-query scaling in n, k, eps."""
    rows = []
    sweeps = []
    for n in [int(x) for x in args.ns.split(",")]:
        sweeps.append(("n", n, args.k, args.eps))
    for k in [int(x) for x in args.ks.split(",")]:
        sweeps.append(("k", args.n, k, args.eps))
    for eps in [float(x) for x in args.epss.split(",")]:
        sweeps.append(("eps", args.n, args.k, eps))
    for sweep, n, k, eps in sweeps:
        for seed in range(args.seeds):
            rng = random.Random(31337 + seed)
            oracle = make_oracle(args.oracle, n, rng)
            res = run_stream(oracle, n, k, eps, args.variant, seed,
                             n_ops=args.ops, adversarial=args.adversarial,
                             brute=False, collect_updates=True)
            qs = [q for _, q in res["upd_queries"]]
            # steady-state = after initial build
            ss = qs[n:] if len(qs) > n else qs
            naive = n * k  # naive recompute cost per update (greedy from scratch)
            row = {
                "sweep": sweep, "variant": args.variant, "oracle": args.oracle,
                "adversarial": int(args.adversarial),
                "n": n, "k": k, "eps": eps, "seed": seed,
                "mean_q": float(np.mean(ss)), "p50_q": float(np.percentile(ss, 50)),
                "p95_q": float(np.percentile(ss, 95)), "max_q": int(np.max(ss)),
                "naive_q": naive, "n_updates": len(ss), "wall_s": res["wall_s"],
                "bound_shape": bound_shape(k, eps),
            }
            rows.append(row)
            print(f"[scal] sweep={sweep} n={n} k={k} eps={eps} seed={seed} "
                  f"mean={row['mean_q']:.1f} p95={row['p95_q']:.1f} "
                  f"max={row['max_q']} naive={naive}", flush=True)
    write_csv(args.out, rows)
    # crude exponent fits
    for sweep, xkey in [("n", "n"), ("k", "k")]:
        sub = [r for r in rows if r["sweep"] == sweep]
        if len({r[xkey] for r in sub}) >= 3:
            xs = np.log([r[xkey] for r in sub])
            ys = np.log([max(r["mean_q"], 1e-9) for r in sub])
            slope = np.polyfit(xs, ys, 1)[0]
            print(f"SCALING fit: mean queries ~ {xkey}^{slope:.2f}  (sweep over {xkey})",
                  flush=True)


def bound_shape(k, eps):
    """The claimed Claim-2 bound, up to constants."""
    return (eps ** -3) * math.log(k) * math.log(k / eps) + (eps ** -2) * k * k * math.log(k)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)
    # dump to stdout so job logs carry the raw data even if uploads fail
    if len(rows) <= 5000:
        print(f"CSV_BEGIN {path}", flush=True)
        with open(path) as fh:
            sys.stdout.write(fh.read())
        print(f"CSV_END {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("ratio")
    r.add_argument("--variant", default="A1", choices=["A0", "A1", "A2"])
    r.add_argument("--oracles", default="maxcut,dicut,covcost")
    r.add_argument("--n", type=int, default=14)
    r.add_argument("--ks", default="3,5")
    r.add_argument("--eps", type=float, default=0.1)
    r.add_argument("--seeds", type=int, default=10)
    r.add_argument("--seed-base", type=int, default=0)
    r.add_argument("--ops", type=int, default=60)
    r.add_argument("--check-every", type=int, default=2)
    r.add_argument("--alg-seeds", type=int, default=1)
    r.add_argument("--adversarial", action="store_true")
    r.add_argument("--out", default="ratio.csv")
    r.set_defaults(fn=exp_ratio)

    s = sub.add_parser("scaling")
    s.add_argument("--variant", default="A1", choices=["A0", "A1", "A2"])
    s.add_argument("--oracle", default="maxcut")
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--eps", type=float, default=0.2)
    s.add_argument("--ns", default="250,500,1000,2000")
    s.add_argument("--ks", default="5,10,20,40")
    s.add_argument("--epss", default="0.1,0.2,0.4")
    s.add_argument("--seeds", type=int, default=3)
    s.add_argument("--ops", type=int, default=400)
    s.add_argument("--adversarial", action="store_true")
    s.add_argument("--out", default="scaling.csv")
    s.set_defaults(fn=exp_scaling)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
