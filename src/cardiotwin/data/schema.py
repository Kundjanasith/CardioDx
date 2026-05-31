from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np

@dataclass
class ECGRecord:
    record_id: str
    signal: np.ndarray
    fs: float
    leads: List[str]
    labels: Optional[Dict[str, int]] = None
    metadata: Optional[Dict] = None

    def to_dict(self):
        d = asdict(self)
        d["signal_shape"] = list(self.signal.shape)
        d.pop("signal")
        return d
