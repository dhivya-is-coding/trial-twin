"""Abstract base class for trial data sources."""

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Interface for loading clinical trial data.

    All data sources (synthetic, GBSG2, PDS, etc.) implement this interface,
    returning data in a unified dict format that the Harmonizer can process.
    """

    @abstractmethod
    def load(self) -> dict[str, pd.DataFrame]:
        """Load and return trial data.

        Returns:
            Dict with keys:
                "subjects": One row per patient (demographics, baseline values)
                "longitudinal": Multiple rows per patient (visits, labs, tumors)
                "endpoints": One row per patient (survival times and events)
            Some sources may not have longitudinal data (e.g., GBSG2),
            in which case "longitudinal" will be an empty DataFrame.
        """
        ...
