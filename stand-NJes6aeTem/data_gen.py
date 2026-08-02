# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Synthetic precondition-induction data generator, per STAND paper (arXiv 2409.07653v2)
Appendix B. Produces a feature matrix of integer-categorical features and binary labels
following a disjunction-of-conjunctions target concept, with structured spurious
co-occurrence and negative-example resampling, matching the paper's described procedure.
"""
import numpy as np


def _sample_conjunct(rng, n_features, n_values_per_feature, n_literals, used_literals=None,
                      overlap_prob=0.0):
    """Sample a conjunct: a dict {feature_idx: value} of n_literals non-overlapping literals.
    If used_literals is given, with probability overlap_prob reuse one of its (feature,value)
    pairs instead of sampling fresh (Appendix B: distractor conjuncts overlap with previously
    sampled literals 20% of the time)."""
    conjunct = {}
    attempts = 0
    while len(conjunct) < n_literals and attempts < n_literals * 20:
        attempts += 1
        if used_literals and rng.random() < overlap_prob:
            f, v = used_literals[rng.integers(len(used_literals))]
            if f in conjunct:
                continue
            conjunct[f] = v
        else:
            f = rng.integers(n_features)
            if f in conjunct:
                continue
            v = rng.integers(n_values_per_feature[f])
        conjunct[f] = v
    return conjunct


def generate_synthetic_dataset(seed, n_samples=2100, n_features=400, n_train=100,
                                n_train_neg=20, holdout_size=2000,
                                n_distractor_conjuncts=100, distractor_apply_rate=0.8,
                                distractor_overlap_prob=0.2, target_apply_rate=0.28,
                                resample_prob=0.10, pool_size=0):
    """Generate one repetition of the synthetic precondition-induction benchmark.

    Returns dict with X_train, y_train (ordered: negatives skewed earlier), X_holdout,
    y_holdout, and the ground-truth target concept (for optional sanity checks).
    """
    rng = np.random.default_rng(seed)

    n_values_per_feature = 2 + rng.poisson(1, size=n_features)
    n_values_per_feature = np.maximum(n_values_per_feature, 2)

    # Feature matrix: uniform random categorical values.
    X = np.zeros((n_samples, n_features), dtype=np.int32)
    for f in range(n_features):
        X[:, f] = rng.integers(0, n_values_per_feature[f], size=n_samples)

    # Target preconditions: 1+Poisson(1) disjuncts, each an OR of TWO conjuncts of
    # 1+Poisson(1) non-overlapping literals (paper says "two conjuncts" explicitly for the
    # disjunction structure it describes, i.e. each disjunct is itself OR(conj_a, conj_b)).
    n_disjuncts = max(1, 1 + rng.poisson(1))
    target_conjuncts = []
    all_target_literals = []
    for _ in range(n_disjuncts):
        n_lits = max(1, 1 + rng.poisson(1))
        conj = _sample_conjunct(rng, n_features, n_values_per_feature, n_lits)
        target_conjuncts.append(conj)
        all_target_literals.extend(conj.items())

    def satisfies_conjunct(x_row, conj):
        return all(x_row[f] == v for f, v in conj.items())

    def satisfies_target(x_row):
        return any(satisfies_conjunct(x_row, c) for c in target_conjuncts)

    # Structured co-occurrence: 100 distractor conjuncts, each applied to 80% of samples
    # independently (injects spurious feature-feature correlation, NOT correlation with the
    # target label — literals overlap with PREVIOUSLY SAMPLED DISTRACTOR literals 20% of the
    # time, building realistic co-occurring feature clusters, per Appendix B / Fig. 5).
    sampled_distractor_literals = []
    for _ in range(n_distractor_conjuncts):
        n_lits = 2 + rng.poisson(3)
        conj = _sample_conjunct(rng, n_features, n_values_per_feature, n_lits,
                                 used_literals=sampled_distractor_literals,
                                 overlap_prob=distractor_overlap_prob)
        sampled_distractor_literals.extend(conj.items())
        apply_mask = rng.random(n_samples) < distractor_apply_rate
        for f, v in conj.items():
            X[apply_mask, f] = v

    # Apply each target conjunct to target_apply_rate fraction of samples (some samples
    # already satisfy a conjunct from the co-occurrence step above and count toward this).
    y = np.zeros(n_samples, dtype=np.int32)
    for conj in target_conjuncts:
        already = np.array([satisfies_conjunct(X[i], conj) for i in range(n_samples)])
        need = int(round(target_apply_rate * n_samples)) - int(already.sum())
        candidates = np.where(~already)[0]
        rng.shuffle(candidates)
        to_apply = candidates[:max(0, need)]
        for f, v in conj.items():
            X[to_apply, f] = v

    y = np.array([1 if satisfies_target(X[i]) else 0 for i in range(n_samples)], dtype=np.int32)

    # Negative examples: force-violate the target by resampling literals with 10% prob each,
    # guaranteeing at least one resampled feature per negative sample.
    neg_idx = np.where(y == 0)[0]
    for i in neg_idx:
        resampled_any = False
        for conj in target_conjuncts:
            for f, v in conj.items():
                if rng.random() < resample_prob:
                    choices = [vv for vv in range(n_values_per_feature[f]) if vv != v]
                    X[i, f] = rng.choice(choices) if choices else v
                    resampled_any = True
        if not resampled_any:
            conj = target_conjuncts[rng.integers(len(target_conjuncts))]
            f, v = next(iter(conj.items()))
            choices = [vv for vv in range(n_values_per_feature[f]) if vv != v]
            if choices:
                X[i, f] = rng.choice(choices)
        assert not satisfies_target(X[i]), "negative sample must not satisfy target concept"

    y = np.array([1 if satisfies_target(X[i]) else 0 for i in range(n_samples)], dtype=np.int32)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_train_pos = n_train - n_train_neg
    train_pos = pos_idx[:n_train_pos]
    train_neg = neg_idx[:n_train_neg]
    train_idx = np.concatenate([train_pos, train_neg])

    # Order training sequence so negatives skew earlier (simulate agent improving over time):
    # assign each training index a random "arrival rank" biased so negatives get earlier ranks.
    pos_ranks = rng.uniform(0.3, 1.0, size=len(train_pos))
    neg_ranks = rng.uniform(0.0, 0.7, size=len(train_neg))
    ranks = np.concatenate([pos_ranks, neg_ranks])
    order = np.argsort(ranks)
    train_idx = train_idx[order]

    remaining = np.setdiff1d(np.arange(n_samples), train_idx)
    rng.shuffle(remaining)
    holdout_idx = remaining[:holdout_size]
    pool_idx = remaining[holdout_size:holdout_size + pool_size]

    result = {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_holdout": X[holdout_idx],
        "y_holdout": y[holdout_idx],
        "n_values_per_feature": n_values_per_feature,
        "target_conjuncts": target_conjuncts,
    }
    if pool_size:
        result["X_pool"] = X[pool_idx]
        result["y_pool"] = y[pool_idx]
    return result


if __name__ == "__main__":
    d = generate_synthetic_dataset(seed=0, n_samples=210, n_features=40, n_train=20,
                                    n_train_neg=4, holdout_size=100, n_distractor_conjuncts=10)
    print("train:", d["X_train"].shape, "pos frac:", d["y_train"].mean())
    print("holdout:", d["X_holdout"].shape, "pos frac:", d["y_holdout"].mean())
    print("n target disjuncts:", len(d["target_conjuncts"]))
