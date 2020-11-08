"""Module for running parallel GWAS analysis per independent feature."""

from collections import defaultdict
import cupy as cp
import pandas as pd
import cudf

def run_gwas(phenotypes_df, phenotype_col, feature_cols, algorithm, add_cols=[]):
    p_value_dict = defaultdict(list)
    for i, f in enumerate(feature_cols):
        if phenotypes_df[f].sum() == 0:
            continue

        model  = algorithm()
        feature_columns = [f] + add_cols
        X = cp.array(phenotypes_df[feature_columns].as_gpu_matrix()).astype(cp.float64)
        #print("fit model for feature {}".format(f))
        model.fit(X,phenotypes_df[phenotype_col].values.astype(cp.float64))

        # We just want p value of feature column, not additional columns. so we grab the first element of the list.
        for p_val,coef,f in zip(model.p_values[1:2],model.coefficients[1:2],feature_columns[0]):
            #print(f'Feature:{f} p_value:{p_val}  coef:{coef}')
            p_value_dict["feature"].append(i)
            p_value_dict["p_value"].append(p_val)
            #p_value_dict["coef"].append(coef)
            p_value_dict["chrom"].append(1)

    print("Visualizing p values")
    df = pd.DataFrame(p_value_dict)
    df = cudf.DataFrame(df)
    return df
