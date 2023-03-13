import pickle
import tempfile
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import numpy
import pyarrow.parquet as pq

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, HalvingRandomSearchCV, RandomizedSearchCV, PredefinedSplit)
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
import pandas
from sklearn.dummy import DummyClassifier

# LOAD DATA

df = pq.read_table(snakemake.input.features).to_pandas()
labels = pq.read_table(snakemake.input.labels).to_pandas()

# df = df.fillna(0)

df = df[numpy.load(snakemake.input.columns, allow_pickle=True)]
df = df.loc[numpy.load(snakemake.input.index, allow_pickle=True)]

# drop samples not used in CytoA
if snakemake.wildcards["full"] == "cyto":
    df = df.drop('late', level="meta_fix")
    df = df.drop(0, level="meta_group")
    test_fold = df.index.get_level_values("meta_group").map(lambda x: 0 if x == 2 else -1).values
else:
    test_fold = df.index.to_frame().apply(
        lambda x: 0 if (x.meta_group == 2) and (x.meta_fix == "early") else -1).values

df = df.merge(labels, left_index=True, right_index=True)
df = df[df["meta_label"] != "unknown"]

print(df.shape)

# PREP CLASSIFICATION INPUT

enc = LabelEncoder().fit(df["meta_label"])
y = enc.transform(df["meta_label"])

# selection of the generic channel features for SCIP
if snakemake.wildcards["type"] == "ideas":
    Xs = df.filter(regex="(bf420nm480nm|bf570nm595nm|m01|m06|m09|ssc)$")
else:
    if snakemake.wildcards["mask"] == "otsu":
        Xs = df.filter(regex=".*_otsu_.*(BF1|BF2|SSC)$")
    elif snakemake.wildcards["mask"] == "li":
        Xs = df.filter(regex=".*_li_.*(BF1|BF2|SSC)$")
    elif snakemake.wildcards["mask"] == "otsuli":
        Xs = pandas.concat([
            df.filter(regex=".*_otsu_.*(SSC)$"),
            df.filter(regex=".*_li_.*(BF1|BF2)$"),
        ], axis=1)
    else:
        raise ValueError(snakemake.wildcards["mask"])

if snakemake.wildcards["model"] == 'true':
    steps = [DummyClassifier(strategy="uniform", random_state=0)]
    param_distributions = {}
    resource = 'n_samples'
else:
    steps = [
        RandomUnderSampler(sampling_strategy="majority", random_state=0),
        RandomOverSampler(sampling_strategy="not majority", random_state=0),
        XGBClassifier(
            booster="gbtree",
            objective="multi:softmax",
            eval_metric="merror",
            tree_method="gpu_hist",
            use_label_encoder=False,
            random_state=0,
            n_jobs=1,
            gpu_id=snakemake.config["gpu_id"]
        )
    ]
    param_distributions = {
        "xgbclassifier__max_depth": [7, 6, 5, 4, 3, 2],
        "xgbclassifier__learning_rate": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001],
        "xgbclassifier__subsample": numpy.arange(start=0.1, stop=1.1, step=.1),
        "xgbclassifier__colsample_bytree": numpy.arange(start=0.1, stop=1.1, step=.1),
        "xgbclassifier__gamma": numpy.arange(start=0, stop=31, step=2),
        "xgbclassifier__min_child_weight": numpy.arange(start=1, stop=32, step=2)
        # "xgbclassifier__n_estimators": numpy.arange(start=10, stop=301, step=10)
    }
    resource = 'xgbclassifier__n_estimators'

with tempfile.TemporaryDirectory(dir="/srv/scratch/maximl/sklearn_cache") as tmp_path:
    model = make_pipeline(
        *steps,
        memory=tmp_path
    )

    if snakemake.wildcards["grid"] == "random":
        grid = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=int(snakemake.config["n"]),
            refit=True,
            n_jobs=snakemake.threads,
            cv=5,
            scoring='balanced_accuracy',
            verbose=2,
            return_train_score=True,
            random_state=0

        )
    else:
        grid = HalvingRandomSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            factor=2,
            resource=resource,
            n_candidates=int(snakemake.config["n"]),
            min_resources=10,
            max_resources=640,
            aggressive_elimination=False,
            refit=True,
            n_jobs=snakemake.threads,
            cv=5,
            scoring='balanced_accuracy',
            verbose=2,
            return_train_score=True,
            random_state=0
        )

    test_fold = df.index.get_level_values("meta_group").map(lambda x: 0 if x == 2 else -1).values
    scores = cross_validate(
        grid, Xs, y,
        scoring=('balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro'),
        cv=PredefinedSplit(test_fold),
        return_train_score=True,
        return_estimator=True
    )

# STORE RESULTS

with open(snakemake.output[0], "wb") as fh:
    pickle.dump(scores, fh)
