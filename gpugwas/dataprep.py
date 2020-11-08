import cudf
import cupy as cp

# Example use with regression at:
# Link: https://gist.github.com/VibhuJawa/d932250a35d15197d35cf37c9d00ba42

#
#def create_matrix_from_features(f_df, n_features):
#    """
#        Create a sparse csr matrix of the shape n_samples,n_features
#        Each row corresponds to the feature values
#    """
#    feature_counts = f_df["sample"].value_counts().reset_index()
#    feature_counts.rename(
#        columns={"index": "sample", "sample": "feature_counts"}, inplace=True
#    )
#    feature_counts = feature_counts.sort_values(by=["sample"]).reset_index(drop=True)
#
#    n_samples = len(feature_counts)
#
#    data = f_df["value"].values
#    indices = f_df["feature_id"].values
#    indptr = feature_counts["feature_counts"].cumsum().values
#    indptr = cp.pad(indptr, (1, 0), "constant")
#
#    return cp.sparse.csr_matrix(
#        arg1=(data, indices, indptr), dtype=cp.float32, shape=(n_samples, n_features)
#    )

def create_matrix_from_features(f_df, n_features, data_col):
    feature_counts = f_df["sample"].value_counts().reset_index()
    feature_counts.rename(columns={'index':'sample','sample':'feature_counts'},inplace=True)
    feature_counts = feature_counts.sort_values(by=['sample']).reset_index(drop=True)

    n_samples = len(feature_counts)

    data = f_df[data_col].values
    indices = f_df["feature_id"].values
    indptr = feature_counts['feature_counts'].cumsum().values
    indptr = cp.pad(indptr, (1, 0), "constant")

    return cp.sparse.csr_matrix(arg1=(data, indices, indptr), dtype=cp.float32,shape=(n_samples,n_features))


def create_phenotype_df(vcf_df, ann_df, phenotype_cols, vcf_col, vcf_sample_col="sample", ann_sample_col="Sample"):
    # Wrap all the matrix processing code here

    # Merge annotations with variant DF
    print("Merging annotations")
    f_df = ann_df.merge(vcf_df,how='inner',left_on = [ann_sample_col],right_on = [vcf_sample_col])
    print(f_df)
    
    # Create feature matrix
    print("Creating feature matrix")
    n_features = len(f_df["feature_id"].unique())
    print("Number of independent features is", n_features)
    matrix  = create_matrix_from_features(f_df, n_features = n_features, data_col=vcf_col)
    matrix = matrix.todense()
    
    # Add variant features to phenotype df
    print("Adding variant features to phenotype df")
    phenotypes_df = f_df[[ann_sample_col] + phenotype_cols].drop_duplicates()
    phenotypes_df = phenotypes_df.sort_values(by=[ann_sample_col]).reset_index(drop=True)
    features = []
    for i in range(n_features):
        f_name = f'variant_{i}'
        features.append(f_name)
        phenotypes_df[f_name]= matrix[:,i]
    print(phenotypes_df)
    return phenotypes_df, features
