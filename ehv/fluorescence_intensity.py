# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/2a_fluorescence_intensity_preprocessing.ipynb (unless otherwise specified).

__all__ = ['apply_logicle']

# Cell
# export

import pandas
import os
import numpy
import seaborn
import logging
import matplotlib.pyplot as plt
from importlib import reload

# Cell
from ehv import load
import flowutils
import ppscore
reload(load)

# Cell
def apply_logicle(df, columns=["Intensity_MC_TMR", "Intensity_MC_Cy5", "Intensity_MC_DAPI"]):
    df = df.copy()

    for (t_idx, r_idx), group_df in df.groupby(["timepoint", "replicate"]):

        for feat in columns:

            y = group_df[[feat]].values

            df.loc[(df["timepoint"] == t_idx) & (df["replicate"] == r_idx), feat+"_logicle"] = \
                flowutils.transforms.logicle(y, numpy.arange(y.shape[1]), r_quant=True)

    return df