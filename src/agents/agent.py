from abc import ABC, abstractmethod
import torch

from ..models.dqn import DQN

class BaseAgent(ABC):
    @property
    @abstractmethod
    def model(self) -> DQN:
        pass

    @abstractmethod
    def select_action(self, obs: torch.Tensor) -> int:
        pass

    @abstractmethod
    def update(self) -> float:
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        pass
