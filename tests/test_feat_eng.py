import datetime
import pandas as pd
import numpy as np

from levtools import feat_eng


def test_one_hot_enc():

    dt_series = pd.DataFrame([[datetime.datetime.now()]], columns=['datetime'])['datetime']
    columns = feat_eng.get_onehot_time_columns(dt_series, use_hour_of_week=True).columns

    assert 'datetime.is_weekend' in columns
    assert 'datetime.is_holiday' in columns
    assert 'datetime.hour_of_week' in columns


def test_harmonic_time_encoding():
    n = 100
    time_series = pd.DataFrame(
        pd.date_range(start=pd.to_datetime("2021-01-01"), end=pd.to_datetime('2022-01-01'), periods=100,
                      name='time_col'))['time_col']

    features = feat_eng.HarmonicTimeColumns(n_octaves=3, n_phases=2, base_functions=[np.sin, np.cos], periods=['day', 'week', 'year'])(time_series)

    assert features.shape == (n, 3 * 2 * 2 * 3 + 2)
