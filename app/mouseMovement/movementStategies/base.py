from abc import ABC, abstractmethod
from ..mouseContext import MouseContext

class MovementStrategy(ABC):
    @abstractmethod
    def move(self, context: MouseContext) -> None:
        pass