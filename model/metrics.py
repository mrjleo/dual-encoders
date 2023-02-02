import numpy as np
import torch
from scipy.spatial import cKDTree
from torchmetrics import Metric


class KLDivergence(Metric):
    """Multivariate KL divergence Metric."""

    def __init__(self, emb_size: int, dist_sync_on_step: bool = False) -> None:
        """Constructor.

        Args:
            emb_size (int): Embedding size.
            dist_sync_on_step (bool, optional): Synchronize the attributes after every step. Defaults to False.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "p", default=torch.Tensor(size=[0, emb_size]), dist_reduce_fx=None
        )
        self.add_state(
            "q", default=torch.Tensor(size=[0, emb_size]), dist_reduce_fx=None
        )

    def update(self, p: torch.Tensor, q: torch.Tensor) -> None:
        """Update the metric.

        Args:
            p (torch.Tensor): The embeddings of the document encoder.
            q (torch.Tensor): The embeddings of the query encoder.
        """
        assert self.p.shape[1] == p.shape[1]
        assert self.q.shape[1] == q.shape[1]
        self.p = torch.vstack((self.p, p))
        self.q = torch.vstack((self.q, q))

    def compute(self) -> float:
        """Compute the multivariate KL divergence metric.
        Adapted from https://mail.python.org/pipermail/scipy-user/2011-May/029521.html.

        Returns:
            float: The multivariate KL divergence.
        """
        x = (
            self.p.view(self.p.shape[0] * self.p.shape[1], self.p.shape[2])
            .cpu()
            .numpy()
        )
        y = (
            self.q.view(self.q.shape[0] * self.q.shape[1], self.q.shape[2])
            .cpu()
            .numpy()
        )
        n, d = x.shape
        m, dy = y.shape
        assert d == dy

        r = cKDTree(x).query(x, k=2, eps=0.01, p=2)[0][:, 1]
        s = cKDTree(y).query(x, k=1, eps=0.01, p=2)[0]
        return -np.log(r / s).sum() * d / n + np.log(m / (n - 1.0))


class TensorStack(Metric):
    """Metric used for stacking and synchronizing embedding tensors."""

    def __init__(
        self, emb_size: int, max_size: int = None, dist_sync_on_step: bool = False
    ) -> None:
        """Constructor.

        Args:
            emb_size (int): Embedding size.
            max_size (int, optional): Maximum number of embeddings. Defaults to None.
            dist_sync_on_step (bool, optional): Synchronize the attributes after every step. Defaults to False.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "t", default=torch.Tensor(size=[0, emb_size]), dist_reduce_fx=None
        )
        self.max_size = max_size

    def update(self, t: torch.Tensor) -> None:
        """Update the metric.

        Args:
            t (torch.Tensor): Tensor of embeddings to add.
        """
        assert self.t.shape[1] == t.shape[1]
        if self.t.shape[0] < self.max_size:
            self.t = torch.vstack((self.t, t))

    def compute(self) -> torch.Tensor:
        """Return the stacked tensors.

        Returns:
            torch.Tensor: All embeddings stacked.
        """
        return self.t.view(self.t.shape[0] * self.t.shape[1], self.t.shape[2])[
            : self.max_size
        ]
