import pandas as pd
import numpy as np


# time related features


def get_hour_of_week(dt):
    return dt.dt.dayofweek * 24 + dt.dt.hour


def get_minutes_of_day(dt):
    return dt.dt.hour * 60 + dt.dt.minute


def get_hour_of_week_2_weekend_mapper(hour_of_week):
    if hour_of_week < 24:
        return (24 - hour_of_week) / 24  # decrease to Tuesday
    if hour_of_week < 4 * 24:
        return 0
    if hour_of_week < 5 * 24:  # increase from Thursday
        return (hour_of_week - 4 * 24) / 24
    return 1


def get_weekend(hour_of_week):
    return hour_of_week.apply(get_hour_of_week_2_weekend_mapper)


def get_hour_of_year(dt):
    return dt.dt.dayofyear * 24 + dt.dt.hour


def get_trig_bases(n_phases, n_max_length, n_octaves=1, base_func=np.cos):
    # retuns an array with indices, n_harmonic x n_bases x n_phases

    column_ids = np.arange(n_phases)
    x = column_ids / n_phases

    bases = [[base_func(((base_row / n_max_length)-x) * 2 * np.pi * octave)
              for base_row in np.arange(n_max_length)]
             for octave in np.arange(1, n_octaves + 1)]

    bases = np.array(bases)
    return bases


def to_datetime(date_str, to_naive=True):
    dt = pd.to_datetime(date_str)

    if to_naive:
        if dt.tz is not None:
            return dt.tz_convert(None)
    return dt


class PeakSeason:
    def __init__(self):
        periods = \
            [
                {"from": "2019-01-01", "to": "2019-01-13"},
                {"from": "2019-04-13", "to": "2019-04-28"},
                {"from": "2019-06-16", "to": "2019-09-22"},
                {"from": "2019-12-18", "to": "2020-01-13"},
                {"from": "2020-04-04", "to": "2020-04-20"},
                {"from": "2020-07-01", "to": "2020-09-17"},
                {"from": "2020-12-17", "to": "2021-01-11"},
                {"from": "2021-03-27", "to": "2021-04-11"},
                {"from": "2021-06-14", "to": "2021-09-13"},
                {"from": "2021-12-16", "to": "2022-01-10"},
                {"from": "2022-04-08", "to": "2022-04-25"},
                {"from": "2022-06-15", "to": "2022-09-15"},
                {"from": "2022-12-05", "to": "2023-01-09"}]

        self.periods = [{'from': to_datetime(period['from']),
                         'to': to_datetime(period['to'])} for period in periods]

    def get_season(self, dt):
        dates_by_one_period = np.zeros_like(dt).astype(bool)
        for period in self.periods:
            dates_by_one_period = np.logical_or(dates_by_one_period,
                                                np.logical_and(dt >= period['from'], dt <= period['to']))
        return dates_by_one_period


def get_onehot_time_columns(dt, use_day_of_week=True, use_month_of_year=True, use_hour_of_day=True,
                            use_hour_of_week=False):
    prefix = dt.name
    hour_of_week = get_hour_of_week(dt).rename(f'{prefix}.hour_of_week')

    if use_day_of_week:
        df = pd.get_dummies(dt.dt.dayofweek, prefix=f'{prefix}.in_week')
    else:
        df = pd.DataFrame()

    if use_month_of_year:
        month_onehot = pd.get_dummies(dt.dt.month, prefix=f'{prefix}.in_year')
        df = pd.concat([df, month_onehot], axis=1)

    if use_hour_of_day:
        hour_onehot = pd.get_dummies(dt.dt.hour, prefix=f'{prefix}.in_day')
        df = pd.concat([df, hour_onehot], axis=1)

    if use_hour_of_week:
        df = pd.concat([df, hour_of_week], axis=1)

    df[prefix + '.is_weekend'] = get_weekend(hour_of_week)

    ps = PeakSeason()
    df[prefix + '.is_holiday'] = ps.get_season(dt).astype(float)

    return df


class HarmonicTimeColumns:
    def __init__(self, n_octaves=1, n_phases=1, base_functions=[np.cos, np.sin], periods=['day', 'week', 'year']):
        self.n_octaves = n_octaves
        self.n_phases = n_phases
        self.base_functions = base_functions
        self.periods = periods

    def __call__(self, dt):
        prefix = dt.name
        min_of_day = get_minutes_of_day(dt)
        hour_of_week = get_hour_of_week(dt)
        day_of_year = dt.dt.dayofyear

        period_descs = {
            'day': {'unit': min_of_day, 'length': 24 * 60},
            'week': {'unit': hour_of_week, 'length': 168},
            'year': {'unit': day_of_year, 'length': 366},
        }

        time_cols = []
        column_names = []

        for period in self.periods:
            period_desc = period_descs[period]

            for base_func in self.base_functions:
                bases = get_trig_bases(n_phases=self.n_phases, n_max_length=period_desc['length'],
                                       n_octaves=self.n_octaves, base_func=base_func)

                for octave in range(self.n_octaves):
                    time_cols += [bases[octave][period_desc['unit']]]

                    freq_tag = f"_freq_{octave}" if self.n_octaves > 1 else ""

                    for phase in range(self.n_phases):
                        phase_tag = f"_phase_{phase}" if self.n_phases > 1 else ""

                        column_names += \
                            [f'{prefix}.in_{period}_{base_func.__name__}{freq_tag}{phase_tag}']

        time_cols_np = np.concatenate(time_cols, axis=1)
        df = pd.DataFrame(time_cols_np, columns=column_names)

        df[f'{prefix}.is_weekend'] = get_weekend(hour_of_week)

        ps = PeakSeason()
        df[f'{prefix}.is_holiday'] = ps.get_season(dt).astype(float)

        return df
