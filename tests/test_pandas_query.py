import pandas as pd
import numpy as np
import pytest

from levtools.pandas_query import get_query_mask


@pytest.mark.parametrize("query, expected", [
    ['colA == colB', [False, True, True, False, False]],
    ['colA == "A"', [False, False, True, False, False]],
    ['colA != None', [True, True, True, False, True]],
    ['colA != NA',  [True, True, True, False, True]],
    ['colB != inf', [True, True, True, False, True]]
])
def test_pandas_query(query, expected):
    data = [[1, 2], \
            [23, 23], \
            ['A', 'A'], \
            [None, np.inf], \
            ["B", "A"]]
    df = pd.DataFrame(data, columns=['colA', 'colB'])
    result = get_query_mask(df, query)
    np.testing.assert_array_equal(result.values, expected)
