import numpy as np
from per_point_pate.data.split_data import split_data


def test_split_data():
    np.random.seed(0)
    n_data = 10000
    x = np.random.rand(n_data, 8, 8, 3)
    y = np.random.randint(low=0, high=10, size=n_data)

    n_test = 1000
    n_private = 8000
    n_public = 1000

    splits = split_data(x=x,
                        y=y,
                        splits_length=(n_test, n_private, n_public),
                        balanced=True)

    test, private, public = splits

    x_test, y_test = test
    x_private, y_private = private
    x_public, y_public = public

    assert np.shape(x_test) == (1000, 8, 8, 3)

    assert len(x_test) == len(y_test)
    assert len(x_private) == len(y_private)
    assert len(x_public) == len(y_public)