use rule preprocessing as WBC_preprocessing with:
    output:
        "WBC_features.parquet"

rule WBC:
    input:
        "figures/WBC_scip_full_cv_confmat.png",
        "figures/WBC_scip_cyto_cv_confmat.png",
        "figures/WBC_scip_full_cv_metrics.png",
        "figures/WBC_scip_cyto_cv_metrics.png",
        "figures/WBC_ideas_cyto_cv_confmat.png"

rule WBC_labels:
    input:
        features="WBC_features.parquet",
        population_dir="../../meta"
    output:
        "WBC_labels.parquet"
    conda:
        "../envs/environment.yml"
    threads: 1
    log:
        notebook="notebooks/labels.ipynb"
    notebook:
        "../notebooks/WBC/labels.ipynb"

rule WBC_qc_paper_figure:
    input:
        features="WBC_features.parquet",
        columns="indices/columns.npy",
        index="indices/index.npy"
    output:
        "figures/WBC_qc_masks.png"
    conda:
        "../envs/environment.yml"
    threads: 1
    log:
        notebook="notebooks/qc_paper_figure.ipynb"
    notebook:
        "../notebooks/WBC/qc_paper_figure.ipynb"

rule WBC_feature_comparison:
    input:
        features="WBC_features.parquet",
        columns="indices/columns.npy",
        index="indices/index.npy",
        ideas="../../ideas/WBC_ideas_features.parquet"
    conda:
        "../envs/environment.yml"
    threads: 1
    log:
        notebook="notebooks/feature_comparison.ipynb"
    notebook:
        "../notebooks/WBC/feature_comparison.ipynb"

rule WBC_all_hyperparameter_optimization_scip:
    input:
        expand(
            "hpo/wbc_{grid}_{type}_{full}_{mask}_{model}.pickle",
            full=["full", "cyto"],
            grid=["rsh", "random"],
            mask=["otsu", "li", "otsuli"],
            type="scip",
            model="xgboost"
        )

rule WBC_hyperparameter_optimization:
    input:
        features="WBC_features.parquet",
        columns="indices/columns.npy",
        index="indices/index.npy",
        labels="labels.parquet"
    output:
        "hpo/WBC_{grid}_{type}_{full}_{mask}_{model}.pickle"
    conda:
        "../envs/environment.yml"
    threads:
        32
    log:
        "hyperparameter_optimization_{grid}_{type}_{full}_{mask}_{model}.log"
    script:
        "../scripts/WBC/xgb_parameter_search.py"

rule WBC_scip_classification:
    input:
        features="WBC_features.parquet",
        columns="indices/columns.npy",
        index="indices/index.npy",
        hpo_full="hpo/WBC_rsh_scip_full_li_xgboost.pickle",
        hpo_cyto="hpo/WBC_rsh_scip_cyto_li_xgboost.pickle"
    output:
        confmat_full="figures/WBC_scip_full_cv_confmat.png",
        confmat_cyto="figures/WBC_scip_cyto_cv_confmat.png",
        metrics_full="figures/WBC_scip_full_cv_metrics.png",
        metrics_cyto="figures/WBC_scip_cyto_cv_metrics.png"
    threads: 1
    conda:
        "../envs/environment.yml"
    log:
        notebook="notebooks/scip_classification.ipynb"
    notebook:
        "../notebooks/WBC/scip_classification.ipynb"

rule WBC_ideas_classification:
    input:
        hpo_cyto="hpo/WBC_rsh_ideas_cyto_li_xgboost.pickle",
        features="WBC_ideas_features.parquet"
    output:
        confmat="figures/WBC_ideas_cv_confmat.png",
        metrics="figures/WBC_ideas_cv_metrics.png"
    threads: 1
    conda:
        "../envs/environment.yml"
    log:
        notebook="notebooks/ideas_classification.ipynb"
    notebook:
        "../notebooks/WBC/ideas_classification.ipynb"

rule WBC_classification_comparison:
    input:
        hpo_ideas="hpo/WBC_rsh_ideas_cyto_li_xgboost.pickle",
        hpo_scip="hpo/WBC_rsh_scip_cyto_li_xgboost.pickle"
    output:
        "tables/WBC_classification_comparison.tex"
    threads: 1
    conda:
        "../envs/environment.yml"
    log:
        notebook="notebooks/classification_comparison.ipynb"
    notebook:
        "../notebooks/WBC/classification_comparison.ipynb"
