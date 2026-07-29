"""Synthetic decomposable-with-bounded-coupling reward oracle used for the
Claim 1-3 numerical audits (regret scaling, coupling-error scaling,
multi-expert ablation). Satisfies Assumption A.3 (bounded coupling, NOT
additive decomposability) via an adjacent-position interaction term, and
A.4 (bounded range) via clipping to [0, 1].
"""
import numpy as np


class SyntheticEnv:
    def __init__(self, n, num_actions, coupling=0.3, noise_sigma=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.n = n
        self.num_actions = num_actions
        self.mu = rng.uniform(0, 1, size=(n, num_actions))       # per-position action quality
        self.pair_bonus = rng.uniform(0, coupling, size=(n, num_actions))  # coupling strength cap
        self.coupling = coupling
        self.noise_sigma = noise_sigma
        self.rng = rng
        self.best_x = np.argmax(self.mu, axis=1)  # good proxy optimum (coupling is bounded/small)

    def expected(self, x):
        base = self.mu[np.arange(self.n), x].mean()
        coup = 0.0
        if self.coupling > 0:
            same = (x[:-1] == x[1:]).astype(float)
            coup = self.coupling * same.mean()
        return float(np.clip(base * (1 - self.coupling) + coup, 0, 1))

    def __call__(self, x):
        r = self.expected(x) + self.rng.normal(0, self.noise_sigma)
        return float(np.clip(r, 0, 1))

    def random_action(self, i, rng):
        return int(rng.integers(0, self.num_actions))

    def optimal_expected(self):
        return self.expected(self.best_x)
