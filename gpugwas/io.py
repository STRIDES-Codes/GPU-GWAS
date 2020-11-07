"""Module for loading data into dataframe."""
import cudf

def _vcf_to_df(vcf_file):
    """Function to load VCF into gwas dataframe."""
    
def _load_annotations(annotation_path, delimiter = '\t'):
    """Function to load annotations into a Cudf (GPU accelerated) dataframe"""
    return cudf.read_csv(annotation_path,delimiter=delimiter)

