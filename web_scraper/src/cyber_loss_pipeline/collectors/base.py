from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..models import SourceRecord


class Collector(ABC):
    @abstractmethod
    def collect(self) -> Iterator[SourceRecord]:
        """Yield source records without writing to storage."""
        raise NotImplementedError
