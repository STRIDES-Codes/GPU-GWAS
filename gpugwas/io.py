"""Module for loading data into dataframe."""

import cudf
import pysam
from collections import defaultdict
import pandas as pd
import math

nucleotide_dict = {'A' : 1, 'C' : 2, 'G' : 3, 'T' : 4}

def _add_basic_component(record, sample, df_dict):
    df_dict["chrom"].append(record.chrom)
    df_dict["pos"].append(record.pos)
    df_dict["ref"].append(nucleotide_dict[record.ref])
    df_dict["alt"].append(nucleotide_dict[record.alts[0]])
    df_dict["quality"].append(record.qual)
    df_dict["sample"].append(sample)

key_lengths = {"call_AD" : 2, "call_PL" : 3, "AC" : 1, "AF" : 1, "MLEAC" : 1, "MLEAF": 1}

def _add_key_value(record, sample, key, value, df_dict):
    if isinstance(value, tuple) or isinstance(value, list):
        if None in list(value):
            value = [math.nan] * key_lengths[key]
        if len(value) == 1:
            _add_basic_component(record, sample, df_dict)
            df_dict["key"].append(key)
            df_dict["value"].append(value[0])
        else:
            for i, val in enumerate(value):
                _add_basic_component(record, sample, df_dict)
                df_dict["key"].append(f"{key}_{i}")
                df_dict["value"].append(val)
    else:
        _add_basic_component(record, sample, df_dict)
        df_dict["key"].append(key)
        df_dict["value"].append(value)

def assert_one_value_per_feature(df):
    '''
        We should have more than onve value for each sample and feature
    '''
    df = df.groupby(by=['sample','feature_id']).count().max()
    assert df.max()<=1

def create_numerical_features(df, groupby_columns=['chrom','pos','ref','alt']):
    feature_mapping = df[groupby_columns].drop_duplicates().sort_values(by=groupby_columns).reset_index(drop=True)
    feature_mapping.reset_index(drop=False,inplace=True)
    feature_mapping.rename(columns={'index':'feature_id'},inplace=True)
    n_features = len(feature_mapping)

    df = df.merge(feature_mapping)
    #df = df[['sample','feature_id','value', 'key', 'quality'] + groupby_columns]

    #assert_one_value_per_feature(df)

    #df = df.sort_values(by=['sample','feature_id']).reset_index(drop=True)
    return df,feature_mapping

def _load_vcf(vcf_file, info_keys=[], format_keys=[]):
    """Function to load VCF into gwas dataframe."""
    # Load VCF file using pysam
    reader = pysam.VariantFile(vcf_file)

    if "*" in info_keys:
        header_dict = dict(reader.header.info)
        new_keys = []
        for k in header_dict.keys():
            new_keys.append(k)
        info_keys = new_keys
    if "*" in format_keys:
        header_dict = dict(reader.header.formats)
        new_keys = []
        for k in header_dict.keys():
            new_keys.append(k)
        format_keys = new_keys

    print(info_keys)
    info_keys = set(info_keys)
    print(format_keys)
    format_keys = set(format_keys)

    df_dict = defaultdict(list)
    for record in reader:
        if len(record.alts) != 1:
            continue
        if record.ref not in nucleotide_dict or record.alts[0] not in nucleotide_dict:
            continue

        # Run through all variants and all their keys in format
        for sample in record.samples:
            format_dict = dict(record.samples[sample])
            for key, value in format_dict.items():
                if key not in format_keys:
                    continue
                #_add_basic_component(record, sample, df_dict)
                if key == "GT":
                    if None in list(value):
                        value = -1
                    else:
                        value = sum(list(value))
                _add_key_value(record, sample, f"call_{key}", value, df_dict)


            # Run through all variants and all their info keys
            info_dict = dict(record.info)
            for key, value in info_dict.items():
                if key not in info_keys:
                    continue
                #_add_basic_component(record, sample, df_dict)
                _add_key_value(record, sample, key, value, df_dict)


    df = pd.DataFrame.from_dict(df_dict)
    df, feature_mapping = create_numerical_features(df)
    df = df.pivot_table(index=['chrom', 'pos', 'ref', 'alt', 'sample', 'quality', 'feature_id'], columns='key', values='value').reset_index()
    cuda_df = cudf.DataFrame(df)
    return cuda_df 

def _load_annotations(annotation_path, delimiter = '\t'):
    """Function to load annotations into a Cudf (GPU accelerated) dataframe"""
    return cudf.read_csv(annotation_path,delimiter=delimiter)

