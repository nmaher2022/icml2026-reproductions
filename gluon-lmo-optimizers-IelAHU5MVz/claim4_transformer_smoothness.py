# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pandas"]
# ///
"""Claim 4 (NanoGPT/FineWeb layer-wise smoothness), TOY reproduction.

The real paper trains a 124M-param NanoGPT on real FineWeb tokens, 4xA100,
5000 iters -- infeasible on CPU/numpy. This script instead hand-implements a
tiny causal transformer (forward+backward by hand, no autodiff, no torch --
this repo has none installed on purpose) trained on a synthetic Markov-chain
token stream, applies the exact unScion layer-wise LMO update from the
briefing (spectral/SVD update scaled by sqrt(m/n) for transformer-block
weight matrices; sign update scaled by 1/n_p for the weight-tied
embedding/output matrix), and fits the Eq. 10/Eq. 30 layer-wise
(L^0_i, L^1_i) smoothness constants from the resulting stochastic gradient
trajectory. Only the QUALITATIVE direction is being checked: L^0~0 for both
groups, and L^1 for transformer-block layers >> L^1 for the tied
embedding/output layer (paper: ~70 vs ~1.3, ratio ~54x).

Norm conventions (derived from the briefing's unScion definitions via
standard operator/nuclear-norm and max/L1 trace duality):
  block layer i, weight X_i in R^{m_i x n_i} (out_dim x in_dim, nn.Linear
  convention): primal ||X||_(i) = sqrt(n_i/m_i) * sigma_max(X)  (spectral)
              dual   ||G||_(i)* = sqrt(m_i/n_i) * sum(singular values of G) (nuclear)
  embed/output layer p, weight-tied matrix E in R^{V x d} (V=vocab, d=n_p):
              primal ||X||_(p) = n_p * max(|X|)     (entrywise max, = ||.||_{1->inf})
              dual   ||G||_(p)* = (1/n_p) * sum(|G|) (entrywise L1)
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

RNG_SEED = 0


# ----------------------------- data: synthetic Markov chain -----------------
def make_markov_chain(V, k_top, rng):
    P = np.zeros((V, V))
    for i in range(V):
        idxs = rng.choice(V, size=min(k_top, V), replace=False)
        probs = rng.dirichlet(np.ones(len(idxs)) * 2.0)
        P[i, idxs] = probs
    return P


def sample_batch(P, V, B, T_seq, rng):
    seqs = np.zeros((B, T_seq + 1), dtype=np.int64)
    seqs[:, 0] = rng.integers(0, V, size=B)
    for t in range(T_seq):
        cur = seqs[:, t]
        # vectorized categorical sampling per row
        cdf = np.cumsum(P[cur], axis=1)
        u = rng.random(B)[:, None]
        nxt = (u > cdf).sum(axis=1)
        nxt = np.clip(nxt, 0, V - 1)
        seqs[:, t + 1] = nxt
    return seqs[:, :T_seq], seqs[:, 1:T_seq + 1]  # x_ids, y_ids


def sinusoidal_pos_enc(max_T, d):
    pos = np.arange(max_T)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d)
    pe = np.zeros((max_T, d))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


# ----------------------------- generic linear layer --------------------------
def linear_forward(x, W):  # x: (..., in), W: (out, in) -> (..., out)
    return x @ W.T


def linear_backward(dy, x, W):
    x2 = x.reshape(-1, x.shape[-1])
    dy2 = dy.reshape(-1, dy.shape[-1])
    dW = dy2.T @ x2
    dx = dy @ W
    return dx, dW


# ----------------------------- layernorm (no affine) --------------------------
def ln_forward(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    std = np.sqrt(var + eps)
    xn = (x - mu) / std
    return xn, (xn, std)


def ln_backward(dxn, cache):
    xn, std = cache
    dx = (1.0 / std) * (dxn - dxn.mean(-1, keepdims=True)
                         - xn * (dxn * xn).mean(-1, keepdims=True))
    return dx


# ----------------------------- softmax / attention -----------------------------
def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def attention_forward(xn, Wq, Wk, Wv, Wo, mask, d):
    Q = linear_forward(xn, Wq)
    K = linear_forward(xn, Wk)
    Vv = linear_forward(xn, Wv)
    scores = np.einsum('btd,bsd->bts', Q, K) / np.sqrt(d) + mask[None]
    A = softmax(scores, axis=-1)
    AV = np.einsum('bts,bsd->btd', A, Vv)
    out = linear_forward(AV, Wo)
    return out, (xn, Q, K, Vv, A, AV)


def attention_backward(dout, cache, Wq, Wk, Wv, Wo, d):
    xn, Q, K, Vv, A, AV = cache
    dAV, dWo = linear_backward(dout, AV, Wo)
    dA = np.einsum('btd,bsd->bts', dAV, Vv)
    dVv = np.einsum('bts,btd->bsd', A, dAV)
    dscores = A * (dA - (A * dA).sum(-1, keepdims=True))
    dQKT = dscores / np.sqrt(d)
    dQ = np.einsum('bts,bsd->btd', dQKT, K)
    dK = np.einsum('bts,btd->bsd', dQKT, Q)
    dxn_q, dWq = linear_backward(dQ, xn, Wq)
    dxn_k, dWk = linear_backward(dK, xn, Wk)
    dxn_v, dWv = linear_backward(dVv, xn, Wv)
    dxn = dxn_q + dxn_k + dxn_v
    return dxn, dWq, dWk, dWv, dWo


# ----------------------------- MLP -----------------------------
def mlp_forward(xn, W1, W2):
    h1 = linear_forward(xn, W1)
    a1 = np.maximum(h1, 0.0)
    h2 = linear_forward(a1, W2)
    return h2, (xn, h1, a1, W1, W2)


def mlp_backward(dh2, cache):
    xn, h1, a1, W1, W2 = cache
    da1, dW2 = linear_backward(dh2, a1, W2)
    dh1 = da1 * (h1 > 0)
    dxn, dW1 = linear_backward(dh1, xn, W1)
    return dxn, dW1, dW2


# ----------------------------- transformer block -----------------------------
def block_forward(x, p, mask, d):
    xn1, ln1c = ln_forward(x)
    attn_out, attnc = attention_forward(xn1, p['Wq'], p['Wk'], p['Wv'], p['Wo'], mask, d)
    x2 = x + attn_out
    xn2, ln2c = ln_forward(x2)
    mlp_out, mlpc = mlp_forward(xn2, p['W1'], p['W2'])
    x3 = x2 + mlp_out
    return x3, (ln1c, attnc, ln2c, mlpc)


def block_backward(dx3, cache, p, d):
    ln1c, attnc, ln2c, mlpc = cache
    dmlp_out = dx3
    dxn2, dW1, dW2 = mlp_backward(dmlp_out, mlpc)
    dx2 = dx3 + ln_backward(dxn2, ln2c)
    dattn_out = dx2
    dxn1, dWq, dWk, dWv, dWo = attention_backward(dattn_out, attnc, p['Wq'], p['Wk'], p['Wv'], p['Wo'], d)
    dx = dx2 + ln_backward(dxn1, ln1c)
    grads = {'Wq': dWq, 'Wk': dWk, 'Wv': dWv, 'Wo': dWo, 'W1': dW1, 'W2': dW2}
    return dx, grads


# ----------------------------- full model -----------------------------
def init_params(V, d, mlp_mult, n_blocks, rng):
    params = {'E': rng.normal(0, 0.02, size=(V, d))}
    blocks = []
    for b in range(n_blocks):
        scale_attn = 1.0 / np.sqrt(d)
        scale_mlp_in = 1.0 / np.sqrt(d)
        scale_mlp_out = 1.0 / np.sqrt(d * mlp_mult)
        blk = {
            'Wq': rng.normal(0, scale_attn, size=(d, d)),
            'Wk': rng.normal(0, scale_attn, size=(d, d)),
            'Wv': rng.normal(0, scale_attn, size=(d, d)),
            'Wo': rng.normal(0, scale_attn, size=(d, d)),
            'W1': rng.normal(0, scale_mlp_in, size=(d * mlp_mult, d)),
            'W2': rng.normal(0, scale_mlp_out, size=(d, d * mlp_mult)),
        }
        blocks.append(blk)
    params['blocks'] = blocks
    return params


def forward_backward(params, x_ids, y_ids, mask, d):
    B, T = x_ids.shape
    E = params['E']
    x = E[x_ids] + pos_enc[None, :T, :]
    caches = []
    for blk in params['blocks']:
        x, c = block_forward(x, blk, mask, d)
        caches.append(c)
    xf, lnfc = ln_forward(x)
    logits = linear_forward(xf, E)  # weight tying
    # stable cross-entropy
    m = logits.max(-1, keepdims=True)
    logsumexp = m[..., 0] + np.log(np.exp(logits - m).sum(-1))
    logit_y = np.take_along_axis(logits, y_ids[..., None], axis=-1)[..., 0]
    loss = float(np.mean(logsumexp - logit_y))

    probs = softmax(logits, axis=-1)
    onehot = np.zeros_like(probs)
    np.put_along_axis(onehot, y_ids[..., None], 1.0, axis=-1)
    dlogits = (probs - onehot) / (B * T)

    dxf, dE_out = linear_backward(dlogits, xf, E)
    dx = ln_backward(dxf, lnfc)

    block_grads = []
    for blk, c in zip(reversed(params['blocks']), reversed(caches)):
        dx, g = block_backward(dx, c, blk, d)
        block_grads.append(g)
    block_grads = list(reversed(block_grads))

    dE_emb = np.zeros_like(E)
    np.add.at(dE_emb, x_ids, dx)
    dE_total = dE_out + dE_emb

    return loss, {'E': dE_total, 'blocks': block_grads}


# ----------------------------- unScion update -----------------------------
def svd_update(X, G, t):
    m, n = X.shape
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    direction = U @ Vt
    return X - t * np.sqrt(m / n) * direction


def sign_update(X, G, t, n_p):
    return X - (t / n_p) * np.sign(G)


def apply_update(params, grads, t_block, t_embed):
    new_blocks = []
    for blk, g in zip(params['blocks'], grads['blocks']):
        nb = {k: svd_update(blk[k], g[k], t_block) for k in blk}
        new_blocks.append(nb)
    new_E = sign_update(params['E'], grads['E'], t_embed, params['E'].shape[1])
    return {'E': new_E, 'blocks': new_blocks}


# ----------------------------- dual/primal norms -----------------------------
def block_dual_norm(G):
    m, n = G.shape
    s = np.linalg.svd(G, compute_uv=False)
    return np.sqrt(m / n) * s.sum()


def block_primal_norm(X):
    m, n = X.shape
    s = np.linalg.svd(X, compute_uv=False)
    return np.sqrt(n / m) * s.max()


def embed_dual_norm(G, n_p):
    return np.abs(G).sum() / n_p


def embed_primal_norm(X, n_p):
    return n_p * np.abs(X).max()


# ----------------------------- Eq 30 hinge-penalized fit -----------------------------
def fit_L0_L1(Lhat, gnorm, lam=5.0):
    Lhat = np.asarray(Lhat)
    gnorm = np.asarray(gnorm)

    def loss(theta):
        L0, L1 = theta
        approx = L0 + L1 * gnorm
        resid = Lhat - approx
        under = np.maximum(0.0, resid)
        return float(np.sum(resid ** 2) + lam * np.sum(under ** 2))

    x0 = np.array([0.0, max(np.median(Lhat) / max(np.median(gnorm), 1e-8), 1e-3)])
    res = minimize(loss, x0, method='L-BFGS-B', bounds=[(0, None), (0, None)])
    return float(res.x[0]), float(res.x[1])


# ----------------------------- main training + logging loop -----------------------------
def run(V, d, mlp_mult, n_blocks, T_seq, B, K, k_top, t_block, t_embed, seed, verbose=True):
    global pos_enc
    rng = np.random.default_rng(seed)
    P = make_markov_chain(V, k_top, rng)
    pos_enc = sinusoidal_pos_enc(T_seq + 4, d)
    mask = np.triu(np.full((T_seq, T_seq), -1e9), k=1)

    params = init_params(V, d, mlp_mult, n_blocks, rng)

    block_names = []
    for b in range(n_blocks):
        for name in ['Wq', 'Wk', 'Wv', 'Wo', 'W1', 'W2']:
            block_names.append((b, name))

    grad_snapshots = []  # list of dicts: {'E': arr, ('b','name'): arr}
    param_snapshots = []  # same, length K+1
    losses = []

    def flatten_params(p):
        d_ = {'E': p['E'].copy()}
        for b, blk in enumerate(p['blocks']):
            for name, arr in blk.items():
                d_[(b, name)] = arr.copy()
        return d_

    param_snapshots.append(flatten_params(params))

    for k in range(K):
        x_ids, y_ids = sample_batch(P, V, B, T_seq, rng)
        loss, grads = forward_backward(params, x_ids, y_ids, mask, d)
        losses.append(loss)
        gflat = {'E': grads['E'].copy()}
        for b, g in enumerate(grads['blocks']):
            for name, arr in g.items():
                gflat[(b, name)] = arr.copy()
        grad_snapshots.append(gflat)

        params = apply_update(params, grads, t_block, t_embed)
        param_snapshots.append(flatten_params(params))

        if verbose and (k % max(1, K // 10) == 0 or k == K - 1):
            print(f"step {k:5d}  loss={loss:.4f}")

    n_p = params['E'].shape[1]

    # per-layer trajectories of (Lhat, ||g||*) for k=0..K-2, pairing grads[k],grads[k+1]
    # and param deltas snapshot[k+1]-snapshot[k]  ==(same indexing)== step k's delta.
    layer_keys = ['E'] + block_names
    traj = {key: {'Lhat': [], 'gnorm': []} for key in layer_keys}

    for k in range(K - 1):
        g_k = grad_snapshots[k]
        g_k1 = grad_snapshots[k + 1]
        X_k = param_snapshots[k]
        X_k1 = param_snapshots[k + 1]
        for key in layer_keys:
            dg = g_k1[key] - g_k[key]
            dX = X_k1[key] - X_k[key]
            if key == 'E':
                num = embed_dual_norm(dg, n_p)
                den = embed_primal_norm(dX, n_p)
                gn = embed_dual_norm(g_k1[key], n_p)
            else:
                num = block_dual_norm(dg)
                den = block_primal_norm(dX)
                gn = block_dual_norm(g_k1[key])
            if den < 1e-12:
                continue
            traj[key]['Lhat'].append(num / den)
            traj[key]['gnorm'].append(gn)

    return traj, losses, block_names, layer_keys


def main(smoketest):
    if smoketest:
        cfg = dict(V=20, d=8, mlp_mult=4, n_blocks=1, T_seq=6, B=4, K=20,
                   k_top=3, t_block=0.02, t_embed=0.3, seed=RNG_SEED)
    else:
        cfg = dict(V=80, d=24, mlp_mult=4, n_blocks=2, T_seq=16, B=16, K=700,
                   k_top=3, t_block=0.02, t_embed=0.5, seed=RNG_SEED)

    print(f"config: {cfg}")
    traj, losses, block_names, layer_keys = run(**cfg)

    if smoketest:
        assert not any(np.isnan(l) or np.isinf(l) for l in losses), "NaN/Inf loss in smoketest"
        print(f"smoketest OK: loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f}")
        for key in layer_keys:
            n = len(traj[key]['Lhat'])
            print(f"  layer {key}: {n} (Lhat,gnorm) samples, "
                  f"Lhat range [{min(traj[key]['Lhat']):.4g},{max(traj[key]['Lhat']):.4g}]"
                  if n else f"  layer {key}: 0 samples")
        return

    print(f"loss[0]={losses[0]:.4f}  loss[-1]={losses[-1]:.4f}  "
          f"loss[mean last 20%]={np.mean(losses[-len(losses)//5:]):.4f}")

    # per-matrix fits
    per_matrix_rows = []
    for key in layer_keys:
        Lhat = traj[key]['Lhat']
        gnorm = traj[key]['gnorm']
        if len(Lhat) < 5:
            continue
        L0, L1 = fit_L0_L1(Lhat, gnorm)
        name = 'E (embed/output)' if key == 'E' else f"block{key[0]}.{key[1]}"
        group = 'embed_output' if key == 'E' else 'transformer_block'
        per_matrix_rows.append({'matrix': name, 'layer_group': group,
                                 'L0_fit': L0, 'L1_fit': L1,
                                 'n_samples': len(Lhat)})

    per_matrix_df = pd.DataFrame(per_matrix_rows)
    per_matrix_df.to_csv("claim4_per_matrix_smoothness.csv", index=False)
    print("\nPer-matrix fits:")
    print(per_matrix_df.to_string(index=False))

    # pooled group-level fits (all block-layer (Lhat,gnorm) pairs pooled together,
    # and E's own trajectory for embed_output) -- this is the primary claim-4 check.
    rows = []
    block_Lhat, block_gnorm = [], []
    for bn in block_names:
        block_Lhat.extend(traj[bn]['Lhat'])
        block_gnorm.extend(traj[bn]['gnorm'])
    L0_blk, L1_blk = fit_L0_L1(block_Lhat, block_gnorm)
    rows.append({'layer_group': 'transformer_block', 'L0_fit': L0_blk, 'L1_fit': L1_blk,
                 'n_samples': len(block_Lhat),
                 'predicted_stepsize_1_over_L1': 1.0 / L1_blk if L1_blk > 0 else float('inf'),
                 'tuned_stepsize_used': cfg['t_block']})

    L0_emb, L1_emb = fit_L0_L1(traj['E']['Lhat'], traj['E']['gnorm'])
    rows.append({'layer_group': 'embed_output', 'L0_fit': L0_emb, 'L1_fit': L1_emb,
                 'n_samples': len(traj['E']['Lhat']),
                 'predicted_stepsize_1_over_L1': 1.0 / L1_emb if L1_emb > 0 else float('inf'),
                 'tuned_stepsize_used': cfg['t_embed']})

    df = pd.DataFrame(rows)
    df.to_csv("claim4_transformer_smoothness.csv", index=False)
    print("\nPooled group-level fits (primary result):")
    print(df.to_string(index=False))

    ratio = L1_blk / L1_emb if L1_emb > 0 else float('inf')
    print(f"\nL1_block / L1_embed ratio = {ratio:.2f}  (paper: ~70/1.3 ~= 54)")

    # trajectory summary csv (median per layer group, for inspection/plotting)
    traj_rows = []
    for key in layer_keys:
        group = 'embed_output' if key == 'E' else 'transformer_block'
        name = 'E' if key == 'E' else f"block{key[0]}.{key[1]}"
        for i, (lh, gn) in enumerate(zip(traj[key]['Lhat'], traj[key]['gnorm'])):
            traj_rows.append({'step': i, 'matrix': name, 'layer_group': group,
                               'Lhat': lh, 'gnorm': gn})
    pd.DataFrame(traj_rows).to_csv("claim4_trajectory.csv", index=False)
    print("\nWrote claim4_transformer_smoothness.csv, claim4_per_matrix_smoothness.csv, "
          "claim4_trajectory.csv")


if __name__ == "__main__":
    smoketest = "smoke" in sys.argv
    main(smoketest)
