"""Process-wide numerical safety defaults."""

from __future__ import annotations

import numpy as np

np.seterr(all="raise")
