import numpy as np
import pandas as pd

from levtools import tools as t

def test_find_distribution():

    p = np.random.randn(1000)
    fitter = t.FindDistribution('neg_lik')
    res = fitter.fit(p).query('n_params == 2')

    assert 'Normal' in set(res[:3]['name'].values)

def test_isiterable():
    assert t.isiterable([1, 2])
    assert t.isiterable(str)
    assert not t.isiterable(1)


def test_zip_map():
    t.zip_map(lambda x, y: x + y, [1, 2], [3, 4]) == [4, 6]


def test_Lp_norm():
    assert t.Lp_norm(2)(np.array([1, 2, 3])) == np.sqrt(14)


def test_Lp_norm_with_weights():
    assert t.Lp_norm(p=2, w=[3, 1])([1, 1]) == 2


def test_get_evenly_balanced_bins():

    expected_bins=[(0, 3), (3, 6), (6, 9)]
    actual_bins = t.get_evenly_balanced_bins(data=np.arange(10), n_slices=3)
    np.testing.assert_array_almost_equal(actual_bins, expected_bins)


def test_to_nd_arr_conversions():

    mtx_data = [[1, 2], [4, 5]]
    mtx_data_arr = np.array(mtx_data)

    np.testing.assert_array_equal(t.get_np_matrix(mtx_data), mtx_data_arr)
    np.testing.assert_array_equal(t.get_np_matrix(mtx_data_arr), mtx_data_arr)

    mtx_data_df = pd.DataFrame(mtx_data, columns=['A', 'B'])
    np.testing.assert_array_equal(t.get_np_matrix(mtx_data_df), mtx_data_arr)

    np.testing.assert_array_equal(t.get_np_matrix(mtx_data_arr), mtx_data_arr)

    mtx_data_ser = mtx_data_df['A']
    np.testing.assert_array_equal(t.get_np_matrix(mtx_data_ser), np.expand_dims(mtx_data_arr[:, 0], axis=1))


def test_to_nd_vect_conversions():

    vec_data = [1, 2]
    vec_data_arr = np.array(vec_data)

    np.testing.assert_array_equal(t.get_np_vector(vec_data), vec_data_arr)
    np.testing.assert_array_equal(t.get_np_vector(vec_data_arr), vec_data_arr)

    vec_data_ser = pd.Series(vec_data)
    np.testing.assert_array_equal(t.get_np_vector(vec_data_ser), vec_data_arr)

    vec_data_df = pd.DataFrame({"x":vec_data})
    np.testing.assert_array_equal(t.get_np_vector(vec_data_df), vec_data_arr)


def test_uniform_date_range_mtx():
    mtx = t.sample_uniform_date_range("2018-01-01", end="2020-01-01", size=(3, 3))
    assert mtx.shape == (3, 3)
    assert type(mtx[0][0]) == pd.Timestamp


def test_uniform_date_range_scalar():
    sample = t.sample_uniform_date_range("2018-01-01", end="2020-01-01", size=1)
    assert type(sample) == pd.Timestamp


def test_vectorpacker():
    vp = t.VectorPacker(sizes_dict={'DoW': (7, 1, 3), 'MoY': (12, 1, 3), 'base': (3)},
                        bounds_dict={'DoW': t.POS, 'MoY': (0, np.inf), 'base': (-np.inf, np.inf)})

    test_x = np.random.random(vp.len)
    assert all(vp.to_vector(vp.to_dict(test_x)) == test_x)
    print(vp.bounds_vect)


def test_duplicate():
    df = pd.DataFrame([[1, 2]], columns=['a', 'b'])
    assert t.duplicate_pd(df, 2).shape == (2, 2)


def test_one_encoder():

    # given
    input = np.array([4, 5, 6, 7, 6, 5, 5])
    expected = np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.],
                         [0., 0., 1., 0.],
                         [0., 1., 0., 0.],
                         [0., 1., 0., 0.]])

    # when
    actual = t.one_encoder(input)

    # then
    np.testing.assert_array_equal(actual, expected)


def test_if_find_distrib_class_works():

    x = np.random.randn(100)
    
    for loss in ['ks', 'kldiv', 'sym_kldiv', 'L2', 'neg_lik', 'neg_log_lik']:
        fd = t.FindDistribution(loss_type=loss)
        fd.fit(x)

