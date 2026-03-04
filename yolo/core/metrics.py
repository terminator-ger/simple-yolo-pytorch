from typing import Optional, Sequence, Union

import torchmetrics
import torch as th
import matplotlib.pyplot as plt
import matplotlib

class EucildeanDistance(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._m = torchmetrics.functional.pairwise_euclidean_distance
        self.add_state('_cnt', default=th.tensor(0), dist_reduce_fx='sum')
        self.add_state('_sum', default=th.tensor(0), dist_reduce_fx='sum')
        self.add_state('_mean', default=th.tensor(0), dist_reduce_fx='mean')

    def update(self, pred: th.Tensor, target: th.Tensor) -> None:
        self._cnt += th.tensor(1)
        self._mean = self._mean + (1/self._cnt) * (self._m(pred, target) - self._mean)

    def compute(self) -> th.Tensor:
        return self._mean

    def plot(
        self, 
        val: Optional[Union[th.Tensor, Sequence[th.Tensor]]] = None, 
        ax: Optional[matplotlib.axes.Axes] = None
    ) -> tuple[object, object]:
        if ax is None:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(val.detach().cpu().numpy())
        return fig, ax