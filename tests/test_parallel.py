import time
import numpy as np

from levtools import parallel as par


def test_same_data_multiple_executions_args_only():

    # given
    def func(a):
        return a+a

    data = 3
    n_workers = 4
    expected = [func(data) for _ in range(n_workers)]

    # that
    actual = par.parallel_executions_on_same_data(func, args=data, workers=n_workers)

    # then
    assert expected == actual


def test_same_data_multiple_executions_args_and_kwargs():

    # given
    def func(a, b):
        return a*a + b

    data_args = 3
    data_kwargs = {'b': 4}
    n_workers = 4
    expected = [func(data_args, **data_kwargs) for _ in range(n_workers)]

    # that
    actual = par.parallel_executions_on_same_data(func, args=data_args, workers=n_workers, **data_kwargs)

    # then
    assert expected == actual


def test_map():

    # given
    def func(a):
        return a + a

    data = [1, 2, 3, 4, 5, 6, 7, 8]
    n_workers = 4
    expected = list(map(func, data))

    # that
    actual = par.map(func, data, workers=n_workers)

    # then
    assert expected == actual


def test_map_with_timeout():

        # given
        def sleeper_proc(a):
            time.sleep(a)
            return a * 10

        input = np.array([1, 1, 1, 5, 5, 5]) / 10
        timeout = 0.3  # sec
        expected = [1.0, 1.0, 1.0, None, None, None]  # kills process longer then 0.35 sec
        n_workers = 4

        # that
        actual = par.map(sleeper_proc, input, workers=n_workers, timeout=timeout)

        # then
        assert expected == actual


def test_map_without_timeout():
    # given
    def sleeper_proc(a):
        time.sleep(a)
        return a * 10

    input = np.array([1, 2, 3, 4]) / 10
    expected = [1.0, 2.0, 3.0, 4.0]
    n_workers = 4

    # that
    actual = par.map(sleeper_proc, input, workers=n_workers)

    # then
    assert expected == actual
