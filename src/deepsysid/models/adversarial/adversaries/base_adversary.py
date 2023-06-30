import abc
from typing import Tuple

import torch


class Adversary(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def attack(
            self,
            control_tensor: torch.Tensor,
            state_tensor: torch.Tensor,
            init_hx: Tuple[torch.Tensor]
    ):
        pass
