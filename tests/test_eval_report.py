"""EvalReport is the API contract for evaluation; it must never hide a NaN."""

import pytest

pytest.importorskip("torch")

from cyberft.errors import NonFiniteLossError
from cyberft.evaluate import EvalReport, RowLoss


def make_report() -> EvalReport:
    return EvalReport("eval", [
        RowLoss(0, 10, 1.0, "first"),
        RowLoss(1, 20, 3.0, "worst"),
        RowLoss(2, 30, float("nan"), "broken"),
    ])


def test_mean_and_median_ignore_non_finite():
    report = make_report()
    assert report.mean == pytest.approx(2.0)
    assert report.median == pytest.approx(2.0)
    assert len(report.broken) == 1


def test_worst_is_sorted_descending():
    assert [row.index for row in make_report().worst(2)] == [1, 0]


def test_summary_names_the_offender():
    summary = make_report().summary()
    assert "non-finite: row=2" in summary
    assert "worst: row=1" in summary


def test_assert_finite_raises_on_nan():
    with pytest.raises(NonFiniteLossError, match=r"row\(s\)"):
        make_report().assert_finite()


def test_assert_finite_returns_self_when_clean():
    report = EvalReport("eval", [RowLoss(0, 10, 1.0, "ok")])
    assert report.assert_finite() is report
