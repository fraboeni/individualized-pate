from typing import Tuple
import numpy as np


def split_data(x: np.array, y: np.array, splits_length: Tuple[int, int, int],
               balanced):
    """Split the data into test, private, and public sets.
        Note: this used to be random - now this is deterministic.
        7/14/22
        # split lengths : [n_test, n_private, n_public]
    Returns:
        _type_: data_test, data_private, data_public
    """
    n_data = len(y)
    assert n_data == sum(splits_length)
    
    n_test, n_private, n_public = splits_length

    
    idx_private = np.arange(0, n_private)
    public_end = n_private + n_public
    idx_public = np.arange(n_private, public_end)
    idx_test = np.arange(public_end , public_end+ n_test)

    print(splits_length)
    print(f"Private range: {idx_private[0]} - {idx_private[-1]}")
    print(f"Public : {idx_public[0]} - {idx_public[-1]}")
    print(f"Test : {idx_test[0]} - {idx_test[-1]}")
    print("\n"* 10)
    data_test = x[idx_test], y[idx_test]
    data_private = x[idx_private], y[idx_private]
    data_public = x[idx_public], y[idx_public]

    return data_test, data_private, data_public
