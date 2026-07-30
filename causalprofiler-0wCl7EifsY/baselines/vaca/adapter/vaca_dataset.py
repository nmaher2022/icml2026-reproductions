"""
Runs INSIDE the `vaca` conda env (python 3.9, torch 1.13.1+cpu, PyG 2.2.0).

Wraps a CausalProfiler-generated (data, graph, index_to_variable) triple as a
VACA-compatible `HeterogeneousSCM` dataset + `LightningDataModule`, without
needing access to the true structural equations (CausalProfiler only gives us
samples, not the generating functions) -- so `has_ground_truth` stays False
and VACA's own internal counterfactual/intervention-vs-ground-truth self
tests are simply skipped; we compute our own ground truth via the targets
CausalProfiler already gave us.

Must be run with VACA's repo root on sys.path (see run_vaca.py).
"""
import numpy as np
import torch
from sklearn import preprocessing
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torchvision import transforms as transform_lib

from datasets._heterogeneous import HeterogeneousSCM
from datasets.transforms import ToTensor
import utils.likelihoods as ul
from data_modules._scalers import MaskedTensorLikelihoodScaler


class CausalProfilerSCMDataset(HeterogeneousSCM):
    """
    A HeterogeneousSCM whose X data comes directly from a numpy array (from
    CausalProfiler) rather than from known structural equations. All
    variables are treated as unidimensional continuous with a Delta
    likelihood ('d'), matching what VACA's own toy datasets use for
    continuous variables (see e.g. _params/dataset_chain.yaml:
    likelihood_names 'd_d_d').
    """

    def __init__(self, root_dir, nodes_list, adj_edges, X, lambda_=0.05, transform=None):
        super().__init__(
            root_dir=root_dir,
            transform=transform,
            nodes_to_intervene=list(nodes_list),
            nodes_list=list(nodes_list),
            adj_edges=adj_edges,
            structural_eq=None,  # unknown -> has_ground_truth = False
            noises_distr=None,
            lambda_=lambda_,
        )
        self._X_injected = X.astype(np.float32)

    @property
    def likelihoods(self):
        return [[ul.DeltaLikelihood(1, lambda_=self.lambda_, normalize="dim")]
                for _ in self.nodes_list]

    @property
    def std_list(self):
        return [-1, 1]

    def _create_data(self):
        self.X = self._X_injected
        # No ground truth noise values available; U is unused for training
        # (only referenced for VACA's internal counterfactual GT machinery,
        # which we don't invoke since has_ground_truth=False).
        self.U = np.zeros((self.X.shape[0], self.num_nodes), dtype=np.float32)

    def node_is_image(self):
        return [False for _ in range(self.num_nodes)]


class CausalProfilerDataModule:
    """
    Minimal re-implementation of VACA's HeterogeneousSCMDataModule, but
    built directly from three CausalProfilerSCMDataset splits instead of
    picking a canned dataset by name. Mirrors the subset of the API that
    models/vaca/vaca.py + main.py actually use.
    """

    def __init__(self, train_dataset, valid_dataset, test_dataset,
                 batch_size=32, num_workers=0, normalize="lik"):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.scaler = None
        self._shuffle_train = True

    @property
    def likelihood_list(self):
        return self.train_dataset.likelihood_list

    @property
    def node_dim(self):
        return self.train_dataset.node_dim

    @property
    def num_nodes(self):
        return self.train_dataset.num_nodes

    @property
    def edge_dimension(self):
        return self.train_dataset.num_edges

    @property
    def is_heterogeneous(self):
        return self.train_dataset.is_heterogeneous

    def get_random_train_sampler(self):
        self.train_dataset.set_transform(self._default_transforms())

        def _sample(num_samples):
            loader = DataLoader(self.train_dataset, batch_size=num_samples, shuffle=True)
            return next(iter(loader))

        return _sample

    def query_dataloader(self, dataset, batch_size=None):
        """
        A dataloader over the *entire* given split (drop_last=False, no
        shuffle), used at query-answering time via
        VACA.get_interventional_distr(). The regular train/val/test
        dataloaders above use drop_last=True (needed for stable-shaped
        training batches) which would silently drop data / yield zero
        batches on our tiny toy splits if reused here.
        """
        dataset.set_transform(self._default_transforms())
        bs = batch_size or min(self.batch_size, len(dataset))
        bs = max(1, bs)
        return DataLoader(dataset, batch_size=bs, shuffle=False,
                          num_workers=0, drop_last=False, pin_memory=False)

    def get_deg(self, indegree=True):
        d_list = []
        idx = 1 if indegree else 0
        for data in self.train_dataset:
            d = degree(data.edge_index[idx], num_nodes=data.num_nodes, dtype=torch.long)
            d_list.append(d)
        return torch.cat(d_list).float()

    def prepare_data(self):
        self.train_dataset.prepare_data(normalize_A=None, add_self_loop=True)
        self.valid_dataset.prepare_data(normalize_A=None, add_self_loop=True)
        self.test_dataset.prepare_data(normalize_A=None, add_self_loop=True)
        if self.normalize == "lik":
            self.scaler = MaskedTensorLikelihoodScaler(
                likelihoods=self.train_dataset.likelihoods,
                mask_x0=self.train_dataset.mask_X0[0, :],
            )
            self.scaler.fit(self.train_dataset.X0)
        else:
            self.scaler = preprocessing.FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)

    def _default_transforms(self):
        return transform_lib.Compose(
            [lambda x: self.scaler.transform(x.reshape(1, self.train_dataset.total_num_dim_x0)), ToTensor()]
        )

    def train_dataloader(self):
        self.train_dataset.set_transform(self._default_transforms())
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=self._shuffle_train, num_workers=self.num_workers,
                          drop_last=True, pin_memory=False)

    def val_dataloader(self):
        self.valid_dataset.set_transform(self._default_transforms())
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          drop_last=True, pin_memory=False)

    def test_dataloader(self):
        self.test_dataset.set_transform(self._default_transforms())
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          drop_last=True, pin_memory=False)


def build_datamodule(root_dir, nodes_list, adj_edges, X_all, batch_size, val_frac=0.2, test_frac=0.2, seed=0):
    """
    Split CausalProfiler's observational data into train/valid/test and wrap
    each split as a CausalProfilerSCMDataset sharing the same graph.
    """
    n = X_all.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val - n_test
    assert n_train > 0, "Not enough samples to split into train/valid/test"

    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    train_ds = CausalProfilerSCMDataset(root_dir, nodes_list, adj_edges, X_all[idx_train])
    valid_ds = CausalProfilerSCMDataset(root_dir, nodes_list, adj_edges, X_all[idx_val])
    test_ds = CausalProfilerSCMDataset(root_dir, nodes_list, adj_edges, X_all[idx_test])

    # batch_size must not exceed the smallest split (DataLoaders use
    # drop_last=True, so an oversized batch silently yields 0 batches).
    safe_bs = max(2, min(batch_size, n_train, n_val, n_test))

    dm = CausalProfilerDataModule(train_ds, valid_ds, test_ds, batch_size=safe_bs)
    dm.prepare_data()
    return dm
