import cudf
import cupy as cp

# Example use with regression at:
# Link: https://gist.github.com/VibhuJawa/d932250a35d15197d35cf37c9d00ba42


def assert_one_value_per_feature(df):
    """
        We should not have more than one value for feature per sample
    """
    df = df.groupby(by=["sample", "feature_id"]).count().max()
    assert df.max() <= 1


def create_numerical_features(df, groupby_columns=["chrom", "pos", "ref", "alt"]):
    """
        Creates numerical encoding of features by 
        sorting their occurances by name
    """
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
    df = df[["sample", "feature_id", "value"]]

    assert_one_value_per_feature(df)

    df = df.sort_values(by=["sample", "feature_id"]).reset_index(drop=True)
    return df, feature_mapping


def create_matrix_from_features(f_df, n_features):
    """
        Create a sparse csr matrix of the shape n_samples,n_features
        Each row corresponds to the feature values
    """
    feature_counts = f_df["sample"].value_counts().reset_index()
    feature_counts.rename(
        columns={"index": "sample", "sample": "feature_counts"}, inplace=True
    )
    feature_counts = feature_counts.sort_values(by=["sample"]).reset_index(drop=True)

    n_samples = len(feature_counts)

    data = f_df["value"].values
    indices = f_df["feature_id"].values
    indptr = feature_counts["feature_counts"].cumsum().values
    indptr = cp.pad(indptr, (1, 0), "constant")

    return cp.sparse.csr_matrix(
        arg1=(data, indices, indptr), dtype=cp.float32, shape=(n_samples, n_features)
    )
