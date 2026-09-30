"""Alpha-diversity estimates for one sample.

Units are the taxa ATLAS observed: classified genera plus Explorer clusters
(each cluster counts as one operational unit). Reads the Explorer left as
noise are not counted, since they can't be assigned to any unit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


def alpha_diversity(counts: Mapping[str, int]) -> dict[str, float]:
    abundances = [c for c in counts.values() if c > 0]
    n = sum(abundances)
    s = len(abundances)
    if n == 0:
        return {"observed": 0, "shannon": 0.0, "simpson": 0.0, "pielou": 0.0, "chao1": 0.0, "reads": 0}
    p = [c / n for c in abundances]
    shannon = -sum(x * math.log(x) for x in p)
    simpson = 1.0 - sum(x * x for x in p)  # Gini-Simpson: chance two reads differ
    pielou = shannon / math.log(s) if s > 1 else 0.0
    f1 = sum(1 for c in abundances if c == 1)
    f2 = sum(1 for c in abundances if c == 2)
    # Bias-corrected Chao1 (Chao 1987; Chiu et al. 2014), defined when f2 == 0.
    chao1 = s + (f1 * (f1 - 1)) / (2 * (f2 + 1)) if f2 == 0 else s + (f1 * f1) / (2 * f2)
    return {
        "observed": s,
        "shannon": round(shannon, 4),
        "simpson": round(simpson, 4),
        "pielou": round(pielou, 4),
        "chao1": round(chao1, 2),
        "reads": n,
    }
