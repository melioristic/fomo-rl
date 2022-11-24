#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Created by : Mohit Anand
# Created on : Tue Nov 22 2022 at 6:40:00 PM
# ==========================================================
# Created on Tue Nov 22 2022
# __copyright__ = Copyright (c) 2022, fomo-rl Project
# __credits__ = [Mohit Anand, Julian]
# __license__ = Private
# __version__ = 0.0.0
# __maintainer__ = Mohit Anand
# __email__ = itsmohitanand@gmail.com
# __status__ = Development
# ==========================================================

import pytest
from fomo.utils import *

TOL = 1e-10

binarize_in_out = [
    (np.array([0.2, 0.7]), 50,  np.array([0,1])),
    (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 90,  np.array([0,0 , 0, 0, 0, 0, 0, 0, 0, 1])),
]

@pytest.mark.parametrize("non_binary, t,  binary", binarize_in_out)
def test_binarize(non_binary, t, binary):
    assert np.array_equal(binarize(non_binary, t), binary)

arr_to_frac_in_out = [
    (np.array([0.2, 0.7, 0.1]), np.array([0.2, 0.7, 0.1])),
    (np.array([0, 1, 2, 3]), np.array([0, 1/6, 2/6, 3/6])),
]
    

@pytest.mark.parametrize("arr, out_arr", arr_to_frac_in_out)
def test_arr_to_frac(arr, out_arr):
    
    assert np.sum(arr_to_frac(arr) - out_arr) < TOL
    
cumsum_with_0_in_out = [
    (np.array([0.2,0.5,0.3]), np.array([0, 0.2, 0.7, 1]))
]

@pytest.mark.parametrize("arr, cumsum", cumsum_with_0_in_out)
def test_cumsum_with_0(arr, cumsum):
    assert np.all(cumsum_with_0(arr) == cumsum)