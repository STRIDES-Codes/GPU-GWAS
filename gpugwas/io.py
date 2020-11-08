"""Module for loading data into dataframe."""

import cudf
import pysam
from collections import defaultdict
import pandas as pd
import math
import os


nucleotide_dict = {"A": 1, "C": 2, "G": 3, "T": 4}


def _add_basic_component(record, sample, df_dict):
    df_dict["chrom"].append(record.chrom)
    df_dict["pos"].append(record.pos)
    df_dict["ref"].append(nucleotide_dict[record.ref])
    df_dict["alt"].append(nucleotide_dict[record.alts[0]])
    df_dict["quality"].append(record.qual)
    df_dict["sample"].append(sample)


key_lengths = {"call_AD": 2, "call_PL": 3, "AC": 1, "AF": 1, "MLEAC": 1, "MLEAF": 1}


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
    """
    We should have more than onve value for each sample and feature
    """
    df = df.groupby(by=["sample", "feature_id"]).count().max()
    assert df.max() <= 1


def create_numerical_features(df, groupby_columns=["chrom", "pos", "ref", "alt"]):
    feature_mapping = (
        df[groupby_columns]
        .drop_duplicates()
        .sort_values(by=groupby_columns)
        .reset_index(drop=True)
    )
    feature_mapping.reset_index(drop=False, inplace=True)
    feature_mapping.rename(columns={"index": "feature_id"}, inplace=True)
    n_features = len(feature_mapping)

    df = df.merge(feature_mapping)
    # df = df[['sample','feature_id','value', 'key', 'quality'] + groupby_columns]

    # assert_one_value_per_feature(df)

    # df = df.sort_values(by=['sample','feature_id']).reset_index(drop=True)
    return df, feature_mapping


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
                # _add_basic_component(record, sample, df_dict)
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
                # _add_basic_component(record, sample, df_dict)
                _add_key_value(record, sample, key, value, df_dict)

    df = pd.DataFrame.from_dict(df_dict)
    df, feature_mapping = create_numerical_features(df)
    df = df.pivot_table(
        index=["chrom", "pos", "ref", "alt", "sample", "quality", "feature_id"],
        columns="key",
        values="value",
    ).reset_index()
    cuda_df = cudf.DataFrame(df)
    return cuda_df


def _load_annotations(annotation_path, delimiter="\t"):
    """Function to load annotations into a Cudf (GPU accelerated) dataframe"""
    return cudf.read_csv(annotation_path, delimiter=delimiter)


def _transform_df(df, sample_key_cols, common_key_cols, common_cols, drop_cols):
    """
    Inputs
    ------

    df: pd.DataFrame
        A pandas datafarme read from a vcf file using variantworks.io.vcfio.VCFReader
    sample_key_cols: list
                     List of `sample_variant` columns in the df
    common_key_cols: list
        List of common_variants columns across all samples at a location
    drop_cols : list
        Columns to drop
    Returns
    -------

    A cuDF dataframe modified to
    """
    sample_key_cols = list(set(sample_key_cols) - set(drop_cols))
    common_key_cols = list(set(common_key_cols) - set(drop_cols))
    common_cols = list(set(common_cols) - set(drop_cols))

    df2 = df.drop(columns=drop_cols)
    df2 = df2[sample_key_cols].transpose()
    df2.reset_index(inplace=True)
    pid_attr_split = df2["index"].str.split("_", expand=True)
    pid_attr_split.columns = ["sample", "key"]
    pid_attr_split["key"] = "call_" + pid_attr_split["key"]
    df2 = pd.concat([df2, pid_attr_split], axis=1)
    df2.drop(columns="index", axis=1, inplace=True)

    temp = pd.DataFrame(pid_attr_split["sample"].unique())
    unique_samples = len(temp)
    temp.columns = ["sample"]
    temp = temp.loc[temp.index.repeat(len(common_key_cols))]
    temp = temp.reset_index(drop=True)

    temp2 = df[common_key_cols].transpose().astype("float64")
    temp2["key"] = temp2.index

    temp2 = pd.concat([temp2] * unique_samples, axis=0)
    temp2 = temp2.reset_index(drop=True)
    temp = pd.concat([temp2, temp], axis=1)
    del temp2

    df2 = pd.concat([df2, temp], axis=0)
    del temp

    res_df = pd.melt(
        df2,
        id_vars=["sample", "key"],
        value_vars=df2.columns[:-2],
        var_name="location",
    )
    del df2
    gdf1 = cudf.DataFrame(res_df)
    gdf2 = cudf.DataFrame(df[common_cols])
    gdf1 = gdf1.merge(gdf2, how="left", left_on="location", right_index=True)

    del gdf2

    gdf1 = gdf1.astype({"ref": "int8", "alt": "int8"})
    gdf1 = gdf1[["chrom", "start_pos", "ref", "alt", "sample", "key", "value"]]
    gdf1 = gdf1.pivot(
        index=["chrom", "start_pos", "ref", "alt", "sample"],
        columns=["key"],
        values=["value"],
    ).reset_index()

    col_list = [i[1] if i[0] == "value" else i[0] for i in list(gdf1.columns)]
    gdf1.columns = col_list
    gdf1.rename(columns={"start_pos": "pos"}, inplace=True)
    return gdf1


def _load_vcf_variantworks(
    vcf_file=None,
    num_threads=os.cpu_count(),
    require_genotype=True,
    info_keys=None,
    format_keys=None,
):
    try:
        from variantworks.io.vcfio import VCFReader
    except ImportError:
        print(
            "Install VariantWorks from https://github.com/clara-parabricks/VariantWorks"
        )

    vcf_df = VCFReader(
        vcf_file,
        num_threads=num_threads,
        require_genotype=require_genotype,
        info_keys=info_keys,
        format_keys=format_keys,
    )
    return vcf_df.dataframe
