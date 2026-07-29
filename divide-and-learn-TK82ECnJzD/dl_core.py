"""
Independent from-scratch implementation of D&L (Divide and Learn), following
Algorithm 1-5 and Definitions/Theorems in Section 4-5 of:
  "Divide and Learn: Multi-Objective Combinatorial Optimization at Scale"
  (arXiv:2602.11346, OpenReview TK82ECnJzD).

No official code was released at reproduction time (arXiv comment: "Code URL
coming soon", verified 2026-07-28). This module is a faithful-effort
reconstruction of the algorithm's *mechanics* (decomposition, multi-expert
position-wise bandits, Lagrangian dual coordination, zeroth-order local
refinement) for the purpose of auditing the paper's regret/coupling-error
scaling claims (Claims 1-3) and running from-scratch MOCO experiments
(Claims 4-6). Design choices not fully pinned down by the paper text
(e.g. exact mirror-descent step for the dual update, exact FTRL bonus
constant) are documented inline; where the paper gives an explicit
pseudocode line (Algorithm 2), we follow it literally.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ExpertState:
    """Shared position-action statistics (V_hat, N, W, L_tilde) -- Sec 4.2.2.

    Shape (n, num_actions). When action sets are position-dependent (e.g.
    masked permutation construction), entries for invalid actions are simply
    never selected/updated and are ignored via an availability mask passed
    into select_action.
    """
    n: int
    num_actions: int
    V: np.ndarray = field(init=False)   # value estimate \hat V
    N: np.ndarray = field(init=False)   # visit counts
    W: np.ndarray = field(init=False)   # EXP3 weights
    L: np.ndarray = field(init=False)   # FTRL cumulative loss estimate \tilde L
    M2: np.ndarray = field(init=False)  # Welford M2 for Var(V_hat) per position (overall, not per-action)

    def __post_init__(self):
        self.V = np.zeros((self.n, self.num_actions))
        self.N = np.zeros((self.n, self.num_actions))
        self.W = np.ones((self.n, self.num_actions))
        self.L = np.zeros((self.n, self.num_actions))
        self.M2 = np.zeros(self.n)


def select_action(i, t, state: ExpertState, mask, rho0, c_ucb, gamma_ftrl,
                   rng, use_ts=False):
    """Algorithm 2: SelectAction (position i).

    mask: boolean array of length num_actions, True = available action.
    use_ts=True substitutes Expert 1 (UCB) with Thompson Sampling / posterior
    sampling over a Beta/Normal posterior on V_hat (D&L-TS variant, Sec 6.2).
    """
    avail = np.flatnonzero(mask)
    Ni = state.N[i]
    mean_visits = Ni[avail].mean() if len(avail) else 0.0
    u_i = 1.0 / (1.0 + np.log(1.0 + mean_visits))  # in (0,1], decays slowly with visits

    rho = np.array([(1 - rho0) * u_i / 2, (1 - rho0) * (1 - u_i / 2), rho0])
    e = rng.choice(3, p=rho / rho.sum())

    if e == 0:
        if use_ts:
            # Posterior sampling: Normal-Normal posterior on mean reward per
            # action, tighter variance with more visits (Thompson Sampling).
            var = 1.0 / (1.0 + Ni[avail])
            sample = state.V[i, avail] + rng.normal(0, 1, size=len(avail)) * np.sqrt(var)
            a = avail[np.argmax(sample)]
        else:
            bonus = c_ucb * np.sqrt(np.log(max(t, 2)) / np.maximum(Ni[avail], 1e-6))
            a = avail[np.argmax(state.V[i, avail] + bonus)]
    elif e == 1:
        w = state.W[i, avail]
        p = w / w.sum()
        a = rng.choice(avail, p=p)
    else:
        bonus = gamma_ftrl * np.sqrt(np.log(max(t, 2)) / np.maximum(Ni[avail], 1e-6))
        a = avail[np.argmax(-state.L[i, avail] + bonus)]
    return int(a)


def update_experts(x, r, state: ExpertState, t, eta_exp3=0.1, clip_eps=0.05,
                    B=1.0, pos_penalty=None):
    """Algorithm 5: UpdateExperts.

    Full-bandit feedback: a single scalar reward r is credited to every
    position-action pair (i, x_i) actually used in the executed solution x,
    following "all position-action pairs in the selected solution update
    simultaneously" (Sec 4.1). EXP3 uses a clipped importance-weighted
    estimator with marginal selection probability pi_{t,i}(x_i) approximated
    by the empirical visit frequency at that position (Sec 4.2.2).

    pos_penalty: optional length-n array, the Lagrangian coordination
    penalty lambda_i * xi_i (Sec 4.2, L(x,lambda) = r_t(x) - sum_i
    lambda_i*xi_i) computed from the *previous* round's dual variables and
    *this* round's pre-update violation measure. Subtracted from the credited
    reward at overlapping positions only (zero elsewhere), so the multi-armed
    bandit experts actually see the coordination signal instead of just
    having it logged as a side statistic.
    """
    n = state.n
    for i in range(n):
        a = int(x[i])
        r_i = r - (pos_penalty[i] if pos_penalty is not None else 0.0)
        r_i = float(np.clip(r_i, 0.0, B))
        state.N[i, a] += 1
        cnt = state.N[i, a]
        old_mean = state.V[i, a]
        state.V[i, a] += (r_i - old_mean) / cnt
        # streaming variance of V_hat across actions at this position (for xi)
        pos_mean = state.V[i].mean()
        state.M2[i] = np.var(state.V[i])

        pi = max(state.N[i, a] / max(state.N[i].sum(), 1.0), clip_eps)
        reward_hat = np.clip(r_i / pi, -B / clip_eps, B / clip_eps)
        state.W[i, a] *= np.exp(eta_exp3 * reward_hat / B)
        # renormalize to avoid overflow
        state.W[i] /= state.W[i].max()

        loss_hat = (B - r_i) / pi
        state.L[i, a] += loss_hat / max(t, 1)


def compute_xi(i, state: ExpertState, x_i, k_i):
    """Soft violation measure xi_i^(t) (Sec 4.2, 'Coordinating Overlapping
    Subproblems via Lagrangian Relaxation'):
      xi_i = (k_i - 1) * Var(V_hat_{i,.}) * (1 - N_{i,x_i} / sum_a N_{i,a})
    """
    var = state.M2[i]
    tot = max(state.N[i].sum(), 1.0)
    conf = state.N[i, x_i] / tot
    return max(k_i - 1, 0) * var * (1 - conf)


def dual_update(lam, xi, t, base_lr=1.0):
    """Projected dual ascent, step size alpha_t = O(1/sqrt(t)) (Sec 4.2,
    Theorem 4.5 preamble: "accelerated mirror descent ... step size
    alpha_t=O(1/sqrt(t))"). We use additive projected-gradient ascent with a
    mild decay term (so lambda relaxes back toward 0 as xi -> 0, matching
    "as learning progresses ... penalties naturally diminish") since the
    paper does not pin down the exact per-step update formula. Note: an
    exponentiated-gradient form (lam *= exp(...)) is degenerate here because
    lambda is initialized at 0 and multiplicative updates can never leave 0.
    """
    alpha_t = base_lr / np.sqrt(max(t, 1))
    lam_new = lam + alpha_t * (xi - 0.5 * lam)
    return float(np.clip(lam_new, 0.0, 10.0))


def local_refine(x, Sk, reward_fn, rng, n_flip=1):
    """Algorithm 3: zeroth-order local search within subproblem Sk. Proposes
    a perturbation of n_flip random positions in Sk (resampled uniformly from
    the position's full action set) and keeps it iff the (noisy) scalarized
    reward improves. Costs one extra reward-oracle evaluation.
    """
    x_new = x.copy()
    idx = rng.choice(Sk, size=min(n_flip, len(Sk)), replace=False)
    for i in idx:
        x_new[i] = reward_fn.random_action(i, rng)
    if reward_fn(x_new) > reward_fn(x):
        return x_new
    return x


def sliding_window_subproblems(n, K, t, o0, alpha=0.5):
    """Definition 4.2 (1): sliding-window decomposition with diminishing
    overlap o_t = floor(o0 * t^-alpha). Returns list of K index arrays.
    """
    o_t = int(np.floor(o0 * (max(t, 1) ** (-alpha))))
    o_t = max(o_t, 0)
    s = int(np.ceil(n / K)) + o_t
    stride = max(s - o_t, 1)
    subs = []
    for k in range(K):
        start = k * stride
        end = min(n, start + s)
        if start >= n:
            start, end = max(0, n - s), n
        subs.append(np.arange(start, end))
    return subs


def overlap_multiplicity(subs, n):
    counts = np.zeros(n, dtype=int)
    for S in subs:
        counts[S] += 1
    return counts


def run_dl(n, num_actions, reward_fn, K=4, T=500, o0=None, alpha=0.5,
           rho0=0.1, c_ucb=1.0, gamma_ftrl=1.0, eta_exp3=0.3, clip_eps=0.05,
           local_every=5, seed=0, use_ts=False, use_lagrangian=True,
           experts="all", mask_fn=None, B=1.0):
    """Algorithm 1 (single scalarization run): returns dict with per-step
    realized reward, expected (noiseless) reward regret trace, and coupling
    error trace (sum of xi over overlapping positions, weighted by dual
    variable magnitude as a proxy for realized coordination cost).

    experts: "all" (UCB/TS + EXP3 + FTRL mixture, default) or one of
    {"ucb","ts","exp3","ftrl"} to force a single-expert baseline (rho0=1 with
    that expert's branch only) for the Claim-3 ablation.
    """
    rng = np.random.default_rng(seed)
    state = ExpertState(n, num_actions)
    d = int(np.ceil(n / K))
    o0 = o0 if o0 is not None else max(d // 2, 1)
    lam = np.zeros(n)

    x = np.array([reward_fn.random_action(i, rng) for i in range(n)])
    rewards, exp_rewards, coupling_err, subprob_sizes = [], [], [], []
    best_r, best_x = -np.inf, x.copy()

    for t in range(1, T + 1):
        subs = sliding_window_subproblems(n, K, t, o0, alpha)
        mult = overlap_multiplicity(subs, n)
        subprob_sizes.append(np.mean([len(s) for s in subs]))

        for k, Sk in enumerate(subs):
            for i in Sk:
                mask = mask_fn(i, x) if mask_fn else np.ones(num_actions, dtype=bool)
                if experts == "all":
                    a = select_action(i, t, state, mask, rho0, c_ucb, gamma_ftrl,
                                       rng, use_ts=use_ts)
                elif experts == "ftrl":
                    avail = np.flatnonzero(mask)
                    bonus = gamma_ftrl * np.sqrt(np.log(max(t, 2)) / np.maximum(state.N[i, avail], 1e-6))
                    a = int(avail[np.argmax(-state.L[i, avail] + bonus)])
                elif experts == "ucb":
                    avail = np.flatnonzero(mask)
                    bonus = c_ucb * np.sqrt(np.log(max(t, 2)) / np.maximum(state.N[i, avail], 1e-6))
                    a = int(avail[np.argmax(state.V[i, avail] + bonus)])
                elif experts == "exp3":
                    avail = np.flatnonzero(mask)
                    w = state.W[i, avail]
                    p = w / w.sum()
                    a = int(rng.choice(avail, p=p))
                elif experts == "ts":
                    avail = np.flatnonzero(mask)
                    var = 1.0 / (1.0 + state.N[i, avail])
                    sample = state.V[i, avail] + rng.normal(0, 1, size=len(avail)) * np.sqrt(var)
                    a = int(avail[np.argmax(sample)])
                x[i] = a

        if t % local_every == 0:
            # One zeroth-order refinement step per outer iteration (Algorithm
            # 3), within a single randomly-chosen subproblem's index set --
            # NOT once per subproblem. Calling it inside the "for k, Sk"
            # loop above (K times per triggered iteration, 2 extra oracle
            # calls each) was a real bug: it makes LocalRefine's cost scale
            # with K, which silently inflates D&L's total oracle-call budget
            # well beyond T and confounds any compute-efficiency comparison
            # (Claim 4) -- caught via the MOCO benchmark's eval-count
            # bookkeeping (dl_T=100 was showing up as ~4300 real oracle
            # calls per weight-sweep run).
            Sk = subs[int(rng.integers(len(subs)))]
            x = local_refine(x, Sk, reward_fn, rng)

        # Compute this round's violation measure xi_i and the Lagrangian
        # penalty lambda_i(t-1)*xi_i *before* crediting the reward, so the
        # penalty is causal (uses only information available prior to this
        # round's update) and actually reaches the experts via update_experts
        # rather than being logged as a side statistic only (see docstring on
        # update_experts / Claim 3b).
        overlap_positions = np.flatnonzero(mult > 1)
        xi_vals = {}
        pos_penalty = np.zeros(n)
        if use_lagrangian:
            for i in overlap_positions:
                xi_vals[i] = compute_xi(i, state, x[i], mult[i])
                pos_penalty[i] = lam[i] * xi_vals[i]

        r = reward_fn(x)
        rewards.append(r)
        exp_rewards.append(reward_fn.expected(x))
        if r > best_r:
            best_r, best_x = r, x.copy()
        update_experts(x, r, state, t, eta_exp3=eta_exp3, clip_eps=clip_eps, B=B,
                        pos_penalty=pos_penalty if use_lagrangian else None)

        c_t = 0.0
        for i in overlap_positions:
            if use_lagrangian:
                xi = xi_vals[i]
                lam[i] = dual_update(lam[i], xi, t)
                c_t += lam[i] * xi
            else:
                xi = compute_xi(i, state, x[i], mult[i])
                c_t += xi  # no coordination: raw uncontrolled violation measure
        coupling_err.append(c_t)

    return {
        "rewards": np.array(rewards),
        "exp_rewards": np.array(exp_rewards),
        "coupling_err": np.array(coupling_err),
        "subprob_size": np.array(subprob_sizes),
        "final_x": best_x,
    }
