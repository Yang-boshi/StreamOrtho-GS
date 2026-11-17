"""P3.3: the average-confidence convention (upstream's C / N)."""
from __future__ import annotations

import numpy as np

from stream_ortho.filter.pointmap import conf_stat_value


def test_avg_divides_by_the_update_count():
    assert conf_stat_value(np.array([3.0]), 3.0, "avg")[0] == 1.0
    assert conf_stat_value(np.array([3.0]), 1.0, "avg")[0] == 3.0


def test_max_keeps_the_raw_accumulator():
    assert conf_stat_value(np.array([3.0]), 3.0, "max")[0] == 3.0


def test_zero_or_bad_counts_do_not_divide_by_zero():
    out = conf_stat_value(np.array([3.0, 3.0, 3.0]), np.array([0.0, np.nan, -2.0]), "avg")
    assert np.allclose(out, [3.0, 3.0, 3.0])


def test_gate_is_a_threshold_on_the_average():
    c = np.array([3.0, 30.0, 300.0])
    assert (conf_stat_value(c, 10.0, "avg") >= 1.5).tolist() == [False, True, True]
