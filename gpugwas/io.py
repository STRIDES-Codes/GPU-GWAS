"""Module for loading data into dataframe."""

import cudf
import pysam
from collections import defaultdict
import pandas as pd

nucleotide_dict = {'A' : 1, 'C' : 2, 'G' : 3, 'T' : 4}

def _add_basic_component(record, sample, df_dict):
    df_dict["chrom"].append(record.chrom)
    df_dict["pos"].append(record.pos)
    df_dict["ref"].append(nucleotide_dict[record.ref])
    df_dict["alt"].append(nucleotide_dict[record.alts[0]])
    df_dict["sample"].append(sample)

def _add_key_value(key, value, df_dict):
    if isinstance(value, tuple) or isinstance(value, list):
        if len(value) == 1:
            df_dict["key"].append(key)
            df_dict["value"].append(value[0])
        else:
            for i, val in enumerate(value):
                df_dict["key"].append(f"{key}_{i}")
                df_dict["value"].append(val)
    else:
        df_dict["key"].append(key)
        df_dict["value"].append(value)

def _load_vcf(vcf_file, info_keys=[], format_keys=[]):
    """Function to load VCF into gwas dataframe."""
    # Load VCF file using pysam
    reader = pysam.VariantFile(vcf_file)

    info_keys = set(info_keys)
    format_keys = set(format_keys)

    df_dict = defaultdict(list)
    for record in reader:
        if len(record.alts) != 1:
            continue
        if record.ref not in nucleotide_dict or record.alts[0] not in nucleotide_dict:
            continue

        # Run through all variants and all their keys in format
        for sample in record.samples:
            #print(sample, dict(record.samples[sample]))
            format_dict = dict(record.samples[sample])
            for key, value in format_dict.items():
                if key not in format_keys:
                    continue
                _add_basic_component(record, sample, df_dict)
                if key == "GT":
                    if None in list(value):
                        value = -1
                    else:
                        value = sum(list(value))
                _add_key_value(f"call_{key}", value, df_dict)


            # Run through all variants and all their info keys
            info_dict = dict(record.info)
            for key, value in info_dict.items():
                if key not in info_keys:
                    continue
                _add_basic_component(record, sample, df_dict)
                _add_key_value(key, value, df_dict)

        #print(dict(record.info))

    return pd.DataFrame.from_dict(df_dict)

def _vcf_to_df(vcf_file):
    """Function to load VCF into gwas dataframe."""
    
def _load_annotations(annotation_path, delimiter = '\t'):
    """Function to load annotations into a Cudf (GPU accelerated) dataframe"""
    return cudf.read_csv(annotation_path,delimiter=delimiter)

