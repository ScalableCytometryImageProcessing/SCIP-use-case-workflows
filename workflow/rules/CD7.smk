use rule preprocessing as CD7_preprocessing with:
    output:
        "CD7_features.parquet"

rule CD7_all_clustering:
    input:
        "CD7_adata_0.h5ad", "CD7_adata_1.h5ad"

rule CD7_clustering:
    input:
        features="CD7_features.parquet",
        columns="indices/columns.npy",
        index="indices/index.npy"
    output:
        "CD7_adata_{fillna}.h5ad"
    conda:
        "../envs/environment.yml"
    threads: 1
    log:
        notebook="notebooks/clustering_{fillna}.ipynb"
    notebook:
        "../notebooks/CD7/clustering.ipynb"
        
rule CD7_cluster_annotation:
    input:
        "CD7_adata_0.h5ad"
    output:
        "figures/CD7_cluster_annotation.png"
    threads: 1
    conda:
        "../envs/environment.yml"
    log:
        notebook="notebooks/cluster_annotation.ipynb"
    notebook:
        "../notebooks/CD7/cluster_annotation.ipynb"
        
rule CD7_image_inspection:
    input:
        "Experiment-800.czi"
    output:
        "CD7_scenes.txt"
    threads: 1
    conda:
        "../envs/environment.yml"
    log:
        notebook="notebooks/image_inspection.ipynb"
    notebook:
        "../notebooks/image_inspection.ipynb"
