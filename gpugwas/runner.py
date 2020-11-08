"""Module for running parallel GWAS analysis per independent feature."""

from collections import defaultdict
import cupy as cp
import pandas as pd
import cudf

def run_gwas(phenotypes_df, phenotype_col, feature_cols, algorithm):
    p_value_dict = defaultdict(list)
    matrix = cp.array(phenotypes_df[feature_cols].as_gpu_matrix()).astype(cp.float32)
    y = phenotypes_df[phenotype_col].values.astype(cp.float64)
    del phenotypes_df

    for i, f in enumerate(feature_cols):
        model  = algorithm()
        X = cp.expand_dims(matrix[:,i],1)
        model.fit(X,y)
        for p_val,coef in zip(model.p_values[1:],model.coefficients[1:]):
            #print(f'Feature:{f} p_value:{p_val}  coef:{coef}')
            p_value_dict["feature"].append(i)
            p_value_dict["p_value"].append(p_val)
            #p_value_dict["coef"].append(coef)
            p_value_dict["chrom"].append(1)

    print("Visualizing p values")
    df = pd.DataFrame(p_value_dict)
    df = cudf.DataFrame(df)
    return df
