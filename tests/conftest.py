import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emfeatures import MARKETS, make_synthetic_frames, merge_frames  # noqa: E402

SAMPLE_SIZE = 6  # benchmark + volatility index + 4 others: every code path, a fraction of the cost


def _subset(market: str, frames: dict) -> dict:
    spec = MARKETS[market]
    required = [spec.benchmark, spec.vol_index]
    others = [n for n in sorted(frames) if n not in required][: SAMPLE_SIZE - 2]
    return {n: frames[n] for n in required + others}


@pytest.fixture(scope="session")
def brazil_frames():
    return _subset("brazil", make_synthetic_frames("brazil", n_days=1000, seed=3))


@pytest.fixture(scope="session")
def india_frames():
    return _subset("india", make_synthetic_frames("india", n_days=900, seed=4))


@pytest.fixture(scope="session")
def brazil_master(brazil_frames):
    return merge_frames(brazil_frames, MARKETS["brazil"])


@pytest.fixture(scope="session")
def india_master(india_frames):
    return merge_frames(india_frames, MARKETS["india"])
