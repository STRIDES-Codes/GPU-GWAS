"""Module for running parallel GWAS analysis per independent feature."""

from collections import defaultdict
import cupy as cp
import pandas as pd
import cudf

def run_gwas(phenotypes_df, phenotype_col, n_features, algorithm, add_cols=[]):
    p_value_dict = defaultdict(list)
    for i in range(n_features):
        model  = algorithm()
        feature_columns = [f'variant_{i}']
        feature_columns.extend(add_cols)
        X = cp.array(phenotypes_df[feature_columns].as_gpu_matrix()).astype(cp.float64)
        model.fit(X,phenotypes_df[phenotype_col].values.astype(cp.float64))
        
        for p_val,coef,f in zip(model.p_values[1:],model.coefficients[1:],feature_columns):
            #print(f'Feature:{f} p_value:{p_val}  coef:{coef}')
            p_value_dict["feature"].append(i)
            p_value_dict["p_value"].append(p_val)
            #p_value_dict["coef"].append(coef)
            p_value_dict["chrom"].append(1)

    print("Visualizing p values")
    df = pd.DataFrame(p_value_dict)
    df = cudf.DataFrame(df)
    return df