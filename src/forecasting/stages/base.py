from abc import ABC, abstractmethod
import pandas as pd

class PipelineStage(ABC):
    @abstractmethod
    def run(self, pd:pd.DataFrame) -> pd.DataFrame:
        """Must be implemented by child class"""
        pass