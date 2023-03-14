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

import logging
logging.basicConfig(
    filename=snakemake.log[0],
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s;%(levelname)s;%(message)s"
)

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()

logging.info("Snakemake config %s", snakemake.config)
logging.info("Snakemake wildcards %s", snakemake.wildcards)

# LOAD DATA

df = pq.read_table(snakemake.input.features).to_pandas()
labels = pq.read_table(snakemake.input.labels).to_pandas()

# df = df.fillna(0)

df = df[numpy.load(snakemake.input.columns, allow_pickle=True)]
df = df.loc[numpy.load(snakemake.input.index, allow_pickle=True)]

df = df.merge(labels, left_index=True, right_index=True)
df = df[df["meta_label"] != "unknown"]

logging.info("Dataframe shape %s", df.shape)

# drop samples not used in CytoA
if snakemake.wildcards["full"] == "cyto":
    logging.info("Using cyto paper subset")

    df = df.drop('late', level="meta_fix")
    df = df.drop(0, level="meta_group")
    test_fold_outer = df.index.get_level_values("meta_group").map(lambda x: 0 if x == 2 else -1).values
    outer_cv = PredefinedSplit(test_fold_outer)

    inner_cv = 5
else:
    logging.info("Using all available samples")

    test_fold_outer = df.index.to_frame().apply(
        lambda x: 0 if (x["meta_group"] == 2) and (x["meta_fix"] == "early") else -1,
        axis="columns"
    ).values 
    outer_cv = PredefinedSplit(test_fold_outer)

    if "innersample" in snakemake.wildcards["grid"]:
        logging.info("Predefining inner CV split per sample")

        sel = pandas.Series([True]*df.shape[0], index=df.index)
        sel.loc[2, :, 'early'] = False
        inner_index = df[sel].index.to_frame(index=False)
        unique_combos = inner_index.set_index(['meta_group','meta_fix']).index.unique()
        test_fold_inner = inner_index.apply(lambda x: unique_combos.get_loc((x.meta_group, x.meta_fix)), axis="columns")
        inner_cv = PredefinedSplit(test_fold_inner)

        # assert that for the predefined splits each test split consists of instances from only one sample
        for outer_train, outer_test in outer_cv.split(df):
            assert len(df.iloc[outer_test].reset_index().set_index(["meta_group", "meta_fix"]).index.unique()) == 1
            for _, inner_test in inner_cv.split(df.iloc[outer_train]):
                unique_combos = df.iloc[outer_train].iloc[inner_test].reset_index().set_index(
                    ["meta_group", "meta_fix"]
                ).index.unique()
                assert len(unique_combos) == 1

    else:
        logging.info("Using default shuffled stratified KFold")
        inner_cv = 5

# PREP CLASSIFICATION INPUT

enc = LabelEncoder().fit(df["meta_label"])
y = enc.transform(df["meta_label"])

# selection of the generic channel features for SCIP
if snakemake.wildcards["type"] == "ideas":
    X = df.filter(regex="(bf420nm480nm|bf570nm595nm|m01|m06|m09|ssc)$")
else:
    if snakemake.wildcards["mask"] == "otsu":
        X = df.filter(regex=".*_otsu_.*(BF1|BF2|SSC)$")
    elif snakemake.wildcards["mask"] == "li":
        X = df.filter(regex=".*_li_.*(BF1|BF2|SSC)$")
    elif snakemake.wildcards["mask"] == "otsuli":
        X = pandas.concat([
            df.filter(regex=".*_otsu_.*(SSC)$"),
            df.filter(regex=".*_li_.*(BF1|BF2)$"),
        ], axis=1)
    else:
        raise ValueError(snakemake.wildcards["mask"])

logging.info("X, y shape (%s, %s)", X.shape, y.shape)  

if snakemake.wildcards["model"] == 'true':
    logging.info("Using dummy model")
    steps = [DummyClassifier(strategy="uniform", random_state=0)]
    param_distributions = {}
    resource = 'n_samples'
else:
    logging.info("Using XGB model")

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

    if snakemake.wildcards["grid"].startswith("random"):
        logging.info("Random hpo")

        grid = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=int(snakemake.config["n"]),
            refit=True,
            n_jobs=snakemake.threads,
            cv=inner_cv,
            scoring='balanced_accuracy',
            verbose=2,
            return_train_score=True,
            random_state=0

        )
    else:
        logging.info("Halving random search hpo")
        
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
            cv=inner_cv,
            scoring='balanced_accuracy',
            verbose=2,
            return_train_score=True,
            random_state=0
        )

    scores = cross_validate(
        grid, X, y,
        scoring=('balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro'),
        cv=outer_cv,
        return_train_score=True,
        return_estimator=True
    )

# STORE RESULTS

with open(snakemake.output[0], "wb") as fh:
    pickle.dump(scores, fh)

logging.info("Output at %s", snakemake.output[0])
