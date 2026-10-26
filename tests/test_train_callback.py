"""FiniteLossCallback must abort on NaN/inf and stay silent on finite loss."""

import math

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from cyberft.errors import NonFiniteLossError
from cyberft.train import FiniteLossCallback


class _State:
    global_step = 7


def test_silent_on_finite_loss():
    cb = FiniteLossCallback()
    cb.on_log(None, _State(), None, logs={"loss": 1.25})  # must not raise


def test_silent_on_empty_logs():
    cb = FiniteLossCallback()
    cb.on_log(None, _State(), None, logs=None)
    cb.on_log(None, _State(), None, logs={})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("key", ["loss", "eval_loss"])
def test_raises_on_non_finite(bad, key):
    cb = FiniteLossCallback()
    assert not math.isfinite(bad)
    with pytest.raises(NonFiniteLossError, match=key):
        cb.on_log(None, _State(), None, logs={key: bad})
