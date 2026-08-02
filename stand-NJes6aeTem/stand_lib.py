# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""STAND: Self-Aware Precondition Induction, reimplemented from arXiv 2409.07653v2 Sections 3-4.

Scope/documented assumptions (see BUGFIX_LOG.md for the full list):
- The lattice node-caching/dedup trick (sample-subset hashing so multiple edges route to a
  shared node) is a stated PERFORMANCE optimization, not required for correctness of
  predictions -- this implementation builds an explicit (undeduped) option-tree instead, which
  is mathematically equivalent for prediction purposes at the toy scale used here.
- Eq. 9 (hierarchical shrinkage of joint probabilities P(c,j) feeding into split-gain/impurity
  calculations) is implemented directly: for every candidate literal considered at a node, walk
  the root-to-node path accumulating raw joint-probability estimates and apply the same
  recursive shrinkage formula as Eq. 8.
- Eq. 7's "in-degree opportunity set" O_x is implemented as: for every node on any root-to-leaf
  path reaching a leaf x filters into that has MORE THAN ONE accepted split literal (a genuine
  alpha-tie / lattice branch point), every one of that node's accepted-split literals (whether or
  not x's raw feature value satisfies it) is one (literal, node_weight) opportunity pair.
  Singleton-split nodes (the common case, no tie) contribute no opportunity: with only one
  accepted literal there is no alternative model in G to disagree with, so testing x's raw value
  against that lone literal would incorrectly register total disagreement for every example that
  legitimately takes the reject branch even though the tree is unambiguous at that node. This
  matches the paper's own framing of Agr_G as measuring disagreement introduced specifically by
  G's lattice of near-tied alternatives (Sec 3.5), not ordinary decision-tree branching. See
  BUGFIX_LOG.md for the debugging trail that found the original (buggy) reading collapsed
  Cert(y=c|x) to exactly (0.0, 0.0) for any example reaching a leaf via a reject edge off a
  singleton-split ancestor.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Node:
    sample_idx: np.ndarray
    depth: int
    parent: "Node | None" = None
    incoming_literal: tuple | None = None  # (feature, value, accept_bool) edge from parent
    splits: list = field(default_factory=list)  # list of (feature, value) accepted split literals
    children: dict = field(default_factory=dict)  # (feature, value, accept_bool) -> Node
    is_leaf: bool = True
    label: int | None = None
    specific_extension: dict = field(default_factory=dict)  # {feature: value} constant-in-leaf
    n_pos: int = 0
    n_neg: int = 0

    @property
    def n(self):
        return self.n_pos + self.n_neg


def _gini(p_pos):
    return 1.0 - p_pos**2 - (1 - p_pos)**2


def _entropy(p_pos):
    h = 0.0
    for p in (p_pos, 1 - p_pos):
        if p > 0:
            h -= p * np.log(p)
    return h


class STAND:
    """STAND classifier. hierarchical_shrinkage=False reproduces the plain "STAND" rows in the
    paper's tables; True reproduces "STAND (h.s.)" / "STAND (heir)"."""

    def __init__(self, alpha=0.1, alpha0=0.1, alpha_M=50, max_depth=12, max_splits_per_node=4,
                 min_samples_leaf=1, hierarchical_shrinkage=False, use_dynamic_alpha=True,
                 lambda_p=25.0, lambda_s=25.0, lambda_n=50.0, criterion="gini",
                 tau_min=0.5, random_state=0, max_total_nodes=20000):
        self.alpha = alpha
        self.use_dynamic_alpha = use_dynamic_alpha
        self.max_total_nodes = max_total_nodes
        self.alpha0 = alpha0
        self.alpha_M = alpha_M
        self.max_depth = max_depth
        self.max_splits_per_node = max_splits_per_node
        self.min_samples_leaf = min_samples_leaf
        self.hierarchical_shrinkage = hierarchical_shrinkage
        self.lambda_p = lambda_p
        self.lambda_s = lambda_s
        self.lambda_n = lambda_n
        self.criterion = criterion
        self.tau_min = tau_min
        self.rng = np.random.default_rng(random_state)
        self.root = None
        self.X = None
        self.y = None
        self.n_features = None
        self._all_nodes = []

    # ---- fitting ----

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        self.X, self.y = X, y
        self.n_features = X.shape[1]
        self._all_nodes = []
        idx = np.arange(len(y))
        self.root = Node(sample_idx=idx, depth=0)
        self._all_nodes.append(self.root)
        self._grow(self.root)
        self._compute_leaf_params()
        return self

    def _impurity(self, sample_idx):
        if len(sample_idx) == 0:
            return 0.0
        p_pos = self.y[sample_idx].mean()
        return _gini(p_pos) if self.criterion == "gini" else _entropy(p_pos)

    def _node_alpha(self, n_samples):
        """Eq. 13's dynamic acceptance-rate schedule (Appendix A.1: "our STAND implementation
        varies the acceptance rate for each node..." -- described unconditionally, not gated on
        hierarchical shrinkage, which is a separate, orthogonal feature governed by
        lambda_p/lambda_s/lambda_n). Used by default for both STAND variants; pass
        use_dynamic_alpha=False for Sec 3.1's simpler fixed-alpha mode instead
        ("alpha = 0.1 works well")."""
        if not self.use_dynamic_alpha:
            return self.alpha
        a = 1 - min((1 - self.alpha0) + self.alpha0 * (n_samples / self.alpha_M), 1.0)
        return max(a, 1e-6)

    def _shrunk_joint_prob(self, node, feature, value, cls, accept):
        """Eq. 9: hierarchical shrinkage of P(y=cls, X_feature==value) (accept=True) or
        P(y=cls, X_feature!=value) (accept=False) along the root-to-node path, using raw
        per-ancestor empirical joint-probability estimates."""
        path = []
        n = node
        while n is not None:
            path.append(n)
            n = n.parent
        path.reverse()  # root .. node

        def raw_joint(nd):
            idx = nd.sample_idx
            if len(idx) == 0:
                return 0.0
            feat_match = self.X[idx, feature] == value
            if not accept:
                feat_match = ~feat_match
            match = feat_match & (self.y[idx] == cls)
            return match.sum() / len(idx)

        est = raw_joint(path[0])
        for level in range(1, len(path)):
            parent_n = len(path[level - 1].sample_idx)
            step = raw_joint(path[level]) - raw_joint(path[level - 1])
            est += step / (1 + self.lambda_p / max(parent_n, 1))
        return est

    def _shrunk_child_impurity(self, node, feature, value, accept):
        """Shrunk-impurity of a would-be child (accept=True: X_feature==value branch;
        accept=False: X_feature!=value branch), using Eq. 9's shrunk joint probabilities
        renormalized within the branch."""
        q = {c: max(self._shrunk_joint_prob(node, feature, value, c, accept), 0.0)
             for c in (0, 1)}
        denom = q[0] + q[1]
        if denom <= 1e-12:
            idx = node.sample_idx
            feat_match = self.X[idx, feature] == value
            if not accept:
                feat_match = ~feat_match
            sub_idx = idx[feat_match]
            p_pos = self.y[sub_idx].mean() if len(sub_idx) else 0.5
        else:
            p_pos = q[1] / denom
        return _gini(p_pos) if self.criterion == "gini" else _entropy(p_pos)

    def _gain(self, node, feature, value):
        idx = node.sample_idx
        if len(idx) == 0:
            return -1.0
        base_imp = self._impurity(idx)
        mask = self.X[idx, feature] == value
        left, right = idx[mask], idx[~mask]
        if len(left) == 0 or len(right) == 0:
            return -1.0

        if self.hierarchical_shrinkage:
            left_imp = self._shrunk_child_impurity(node, feature, value, True)
            right_imp = self._shrunk_child_impurity(node, feature, value, False)
        else:
            left_imp = self._impurity(left)
            right_imp = self._impurity(right)

        w_left, w_right = len(left) / len(idx), len(right) / len(idx)
        return base_imp - (w_left * left_imp + w_right * right_imp)

    def _grow(self, node):
        idx = node.sample_idx
        labels = self.y[idx]
        if len(idx) < 2 * self.min_samples_leaf or node.depth >= self.max_depth or \
                len(set(labels.tolist())) <= 1 or len(self._all_nodes) >= self.max_total_nodes:
            self._finalize_leaf(node)
            return

        candidates = []
        for f in range(self.n_features):
            for v in np.unique(self.X[idx, f]):
                g = self._gain(node, f, int(v))
                if g > 1e-9:
                    candidates.append((g, f, int(v)))
        if not candidates:
            self._finalize_leaf(node)
            return

        max_gain = max(c[0] for c in candidates)
        thresh_alpha = self._node_alpha(len(idx))
        accepted = [c for c in candidates if c[0] >= max_gain * (1 - thresh_alpha)]
        accepted.sort(key=lambda c: -c[0])
        accepted = accepted[: self.max_splits_per_node]

        if not accepted:
            self._finalize_leaf(node)
            return

        node.is_leaf = False
        node.n_pos = int((labels == 1).sum())
        node.n_neg = int((labels == 0).sum())
        for g, f, v in accepted:
            node.splits.append((f, v))
            mask = self.X[idx, f] == v
            for accept_bool, sub_idx in ((True, idx[mask]), (False, idx[~mask])):
                if len(sub_idx) == 0:
                    continue
                child = Node(sample_idx=sub_idx, depth=node.depth + 1, parent=node,
                             incoming_literal=(f, v, accept_bool))
                node.children[(f, v, accept_bool)] = child
                self._all_nodes.append(child)
                self._grow(child)

    def _finalize_leaf(self, node):
        node.is_leaf = True
        idx = node.sample_idx
        labels = self.y[idx]
        node.n_pos = int((labels == 1).sum())
        node.n_neg = int((labels == 0).sum())
        node.label = 1 if node.n_pos >= node.n_neg else 0
        # Specific extension: features constant across all samples in this leaf.
        if len(idx) > 0:
            Xi = self.X[idx]
            const_mask = np.all(Xi == Xi[0], axis=0)
            node.specific_extension = {f: int(Xi[0, f]) for f in np.where(const_mask)[0]}

    # ---- hierarchical-shrinkage leaf/node weights (Eq. 10-12) ----

    def _compute_leaf_params(self):
        for node in self._all_nodes:
            if node.is_leaf:
                node.tau = self._min_acceptance_rate_for_path(node)
            else:
                node.tau = self._min_acceptance_rate_for_path(node)
            if self.hierarchical_shrinkage:
                node.w_n = (1 - node.tau) / (1 + self.lambda_n / max(node.n, 1))
            else:
                node.w_n = 1.0

        self._leaf_wsk = {}
        for node in self._all_nodes:
            if not node.is_leaf:
                continue
            wsk = {}
            for f, v in node.specific_extension.items():
                if self.hierarchical_shrinkage:
                    p_star = self._shrunk_feature_invariance(node, f, v)
                    Nk = node.n
                    wsk[f] = (Nk + 0.5 + p_star) / (Nk + 2)
                else:
                    wsk[f] = 1.0
            self._leaf_wsk[id(node)] = wsk

    def _min_acceptance_rate_for_path(self, node):
        """tau_nk: minimum alpha (acceptance rate) that would still let this node's incoming
        edge be selected -- approximated here as 1 minus the node's own accept-threshold ratio,
        i.e. how close the least-competitive accepted split at this node's PARENT was to the
        best split's gain (0 = only the single best split would have kept this edge, i.e. a
        picky/unambiguous edge; higher = many alternatives were nearly as good)."""
        if node.parent is None or node.incoming_literal is None:
            return 0.0
        f, v, accept_bool = node.incoming_literal
        parent = node.parent
        if not accept_bool:
            return 0.0
        # Recomputing gains here (rather than caching them at growth time) is O(splits) extra
        # work per node; acceptable at the toy scale (<=100 training samples) used in this repro.
        gains = [self._gain(parent, sf, sv) for sf, sv in parent.splits]
        if not gains:
            return 0.0
        max_gain = max(gains)
        this_gain = self._gain(parent, f, v)
        if max_gain <= 1e-9:
            return 0.0
        return max(0.0, 1 - this_gain / max_gain)

    def _shrunk_feature_invariance(self, leaf, feature, value):
        """Eq. 11: shrunk estimate of P(X_feature == value | class) invariance, using the
        leaf's majority class and the leaf-to-root path."""
        cls = leaf.label
        path = []
        n = leaf
        while n is not None:
            path.append(n)
            n = n.parent
        path.reverse()

        def raw_cond(nd):
            idx = nd.sample_idx
            cls_idx = idx[self.y[idx] == cls]
            if len(cls_idx) == 0:
                return 0.0
            return (self.X[cls_idx, feature] == value).mean()

        est = raw_cond(path[0])
        for level in range(1, len(path)):
            parent_n = len(path[level - 1].sample_idx)
            step = raw_cond(path[level]) - raw_cond(path[level - 1])
            est += step / (1 + self.lambda_s / max(parent_n, 1))
        return float(np.clip(est, 0.0, 1.0))

    # ---- prediction / certainty ----

    def _reachable_leaves(self, x):
        """Return list of (leaf_node, path_nodes) for every leaf x filters into, following
        every accepted split at each node (option-tree branching)."""
        leaves = []
        visited_nodes = set()
        stack = [(self.root, [self.root])]
        while stack:
            node, path = stack.pop()
            if id(node) in visited_nodes and node.is_leaf:
                continue
            if node.is_leaf:
                leaves.append((node, path))
                continue
            for f, v in node.splits:
                accept = bool(x[f] == v)
                child = node.children.get((f, v, accept))
                if child is not None:
                    stack.append((child, path + [child]))
        return leaves

    def _ancestor_nodes_and_literals(self, leaves_paths):
        """Union of (node, literal) opportunity pairs from all ancestor nodes on any path to
        any reached leaf (Eq. 7's O_x, see class docstring for the documented interpretation)."""
        seen_nodes = {}
        for _, path in leaves_paths:
            for node in path[:-1]:  # exclude the leaf itself, it has no outgoing splits
                seen_nodes[id(node)] = node
        pairs = []
        for node in seen_nodes.values():
            if len(node.splits) < 2:
                continue  # no lattice tie at this node -- nothing to disagree about
            for f, v in node.splits:
                pairs.append((node, f, v))
        return pairs

    def certainty(self, x):
        """Returns (cert_pos, cert_neg) = (Cert(y=1|x), Cert(y=0|x)) per Eq. 4-7."""
        leaves_paths = self._reachable_leaves(x)
        if not leaves_paths:
            return 0.0, 0.0

        num = {0: 0.0, 1: 0.0}
        den = 0.0
        for leaf, _ in leaves_paths:
            wsk = self._leaf_wsk.get(id(leaf), {})
            overlap = sum(wsk.get(f, 0.0) for f, v in leaf.specific_extension.items()
                          if x[f] == v)
            norm = sum(wsk.values()) if wsk else 0.0
            num[leaf.label] += leaf.w_n * overlap
            den += leaf.w_n * norm
        cert_s = {c: (num[c] / den if den > 1e-12 else 0.0) for c in (0, 1)}

        pairs = self._ancestor_nodes_and_literals(leaves_paths)
        if pairs:
            num_agr = sum(node.w_n * (1.0 if x[f] == v else 0.0) for node, f, v in pairs)
            den_agr = sum(node.w_n for node, f, v in pairs)
            agr = num_agr / den_agr if den_agr > 1e-12 else 1.0
        else:
            agr = 1.0

        return cert_s[1] * agr, cert_s[0] * agr

    def predict_proba_pos(self, X):
        X = np.asarray(X)
        out = np.zeros(len(X))
        for i, x in enumerate(X):
            cp, cn = self.certainty(x)
            total = cp + cn
            out[i] = cp / total if total > 1e-12 else 0.5
        return out

    def predict(self, X):
        return (self.predict_proba_pos(X) >= 0.5).astype(int)

    def predict_cert(self, X):
        """Returns raw Cert(y=1|x) - Cert(y=0|x) style signed certainty in [-1, 1], matching
        the paper's IC(x) convention (positive => predicted correct/positive, magnitude =
        certainty)."""
        X = np.asarray(X)
        out = np.zeros(len(X))
        for i, x in enumerate(X):
            cp, cn = self.certainty(x)
            out[i] = cp if cp >= cn else -cn
        return out
