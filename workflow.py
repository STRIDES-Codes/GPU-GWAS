import argparse
import time
from collections import defaultdict

import cupy as cp
import cudf
import pandas as pd
import rmm

import gpugwas.io as gwasio
import gpugwas.filter as gwasfilter
import gpugwas.algorithms as algos
import gpugwas.dataprep as dp
import gpugwas.runner as runner

from gpugwas.vizb import show_qq_plot, show_manhattan_plot
#import gpugwas.processing as gwasproc

import warnings
warnings.filterwarnings('ignore', 'Expected ')
warnings.simplefilter('ignore')


parser = argparse.ArgumentParser(description='Run GPU GWAS Pipeline')
parser.add_argument('--vcf_path', default = './data/test.vcf')
parser.add_argument('--annotation_path', default = './data/1kg_annotations.txt')
parser.add_argument('--workdir', default = './temp/')
args = parser.parse_args()

# Initialize Memory Pool to 10GB
cudf.set_allocator(pool=True, initial_pool_size=1e10)
cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

# Load data
print("Loading data")
vcf_df, feature_mapping = gwasio.load_vcf(args.vcf_path, info_keys=["AF"], format_keys=["GT", "DP"])
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

# Generate phenotypes dataframe
phenotypes_df, features = dp.create_phenotype_df(vcf_df, ann_df, ['CaffeineConsumption','isFemale','PurpleHair'], "call_GT",
                                       vcf_sample_col="sample", ann_sample_col="Sample")

# Run PCA on phenotype dataframe
phenotypes_df = algos.PCA_concat(phenotypes_df, 3)
print(phenotypes_df)

# Fit linear regression model for each variant feature
print("Fitting linear regression model")

p_value_df = runner.run_gwas(phenotypes_df, 'CaffeineConsumption', features, algos.cuml_LinearReg, add_cols=['PC0'])
print(p_value_df)

# Please save_to='manhattan.svg' argument to save the plot. This require firefox installed.
# conda install -c conda-forge firefox geckodriver
manhattan_plot = show_manhattan_plot(
    p_value_df, 
    'chrom',  
    'p_value', 'feature', 
    title='GWAS Manhattan Plot')

print('Time Elapsed: {}'.format(time.time()- t0))
