#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Created by : Mohit Anand
# Created on : Tue Oct 04 2022 at 10:44:10 PM
# ==========================================================
# Created on Tue Oct 04 2022
# __copyright__ = Copyright (c) 2022, Project FOMO-RL
# __credits__ = [Mohit Anand,]
# __license__ = Private
# __version__ = 0.0.0
# __maintainer__ = Mohit Anand
# __email__ = itsmohitanand@gmail.com
# __status__ = Development
# ==========================================================
import h5py

def read_benchmark_data(path = "benchmark-dataset/", pft="beech"):

    with h5py.File(path + f"train_formind_{pft}_monthly.h5", "r") as f:
        Xd_train = f["Xd"][:]
        Xs_train = f["Xs"][:]
        Y_train = f["Y"][:]

    with h5py.File(path + f"val_formind_{pft}_monthly.h5", "r") as f:
        Xd_val = f["Xd"][:]
        Xs_val = f["Xs"][:]
        Y_val = f["Y"][:]

    with h5py.File(path + f"test_formind_{pft}_monthly.h5", "r") as f:
        Xd_test = f["Xd"][:]
        Xs_test = f["Xs"][:]
        Y_test = f["Y"][:]

    return (
        (Xd_train, Xs_train, Y_train),
        (Xd_val, Xs_val, Y_val),
        (Xd_test, Xs_test, Y_test),
    )
