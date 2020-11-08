import argparse
import time
from collections import defaultdict

import cupy as cp
import cudf
import pandas as pd

import gpugwas.io as gwasio
import gpugwas.filter as gwasfilter
import gpugwas.algorithms as algos
import gpugwas.viz as viz

#import gpugwas.processing as gwasproc

parser = argparse.ArgumentParser(description='Run GPU GWAS Pipeline')
parser.add_argument('--vcf_path', default = './data/test.vcf')
parser.add_argument('--annotation_path', default = './data/1kg_annotations.txt')
parser.add_argument('--workdir', default = './temp/')
args = parser.parse_args()

# Load data
print("Loading data")
vcf_df = gwasio.load_vcf(args.vcf_path, info_keys=["AF"], format_keys=["GT", "DP"])
print(vcf_df.head())
print("Loading annotations")
ann_df = gwasio.load_annotations(args.annotation_path)
#print(ann_df)

# Start benchmarking after I/O
t0 = time.time()

# Filter data
print("Filtering samples")
vcf_df = gwasfilter.filter_samples(vcf_df)
print(vcf_df.head())
print("Filtering variants")
vcf_df = gwasfilter.filter_variants(vcf_df)
print(vcf_df.head())

# Merge annotations with variant DF
print("Merging annotations")
f_df = ann_df.merge(vcf_df,how='inner',left_on = ['Sample'],right_on = ['sample'])
print(f_df)

# Create feature matrix
print("Creating feature matrix")
n_features = len(f_df["feature_id"].unique())
print(n_features)
matrix  = algos.create_matrix_from_features(f_df, n_features = n_features)
matrix = matrix.todense()

# Add variant features to phenotype df
print("Adding variant features to phenotype df")
phenotypes_df = f_df[['Sample','CaffeineConsumption','isFemale','PurpleHair']].drop_duplicates()
phenotypes_df = phenotypes_df.sort_values(by=['Sample']).reset_index(drop=True)
for i in range(n_features):
    phenotypes_df[f'variant_{i}']= matrix[:,i]
print(phenotypes_df)

# Fit linear regression model for each variant feature
print("Fitting linear regression model")
p_value_dict = defaultdict(list)
for i in range(n_features):
    model  = algos.cuml_LinearReg()
    feature_columns = [f'variant_{i}']
    X = cp.array(phenotypes_df[feature_columns].as_gpu_matrix()).astype(cp.float64)
    model.fit(X,phenotypes_df['CaffeineConsumption'].values.astype(cp.float64))
    
    for p_val,coef,f in zip(model.p_values[1:],model.coefficients[1:],feature_columns):
        print(f'Feature:{f} p_value:{p_val}  coef:{coef}')
        p_value_dict["feature"].append(i)
        p_value_dict["p_value"].append(p_val)
        p_value_dict["chrom"].append(1)

# Visualize p values
print("Visualizing p values")
df = pd.DataFrame(p_value_dict)
df = cudf.DataFrame(df)

manhattan_spec = {}
manhattan_spec['df'] = df
manhattan_spec['group_by'] = 'chrom'
manhattan_spec['x_axis'] = 'p_value'
manhattan_spec['y_axis'] = 'feature'

viz.ManhattanPlot({}, manhattan_spec)

print('Time Elapsed: {}'.format(time.time()- t0))
