import gpugwas.io as gwasio
import cudf

# Test loading VCF
print("Test loading VCF to DF")
vcf_df = gwasio._load_vcf("data/test.vcf", info_keys=["AC", "AF"], format_keys=["GT"])
print(vcf_df)

# Test loading annotation file
print("Test loading annotation file")
ann_df = gwasio._load_annotations("data/1kg_annotations.txt")
print(ann_df)

# Test loading VCF (Method 2)
print("Test loading VCF to DF (Method 2)")
vcf_df_2 = gwasio._load_vcf_variantworks(
    "data/test.vcf", info_keys=["AC", "AF"], format_keys=["GT"]
)
vcf_df_2 = gwasio._transform_df(
    vcf_df_2,
    sample_key_cols=list(vcf_df_2.columns[14:]),
    common_key_cols=list(vcf_df_2.columns[7:14]),
    common_cols=list(vcf_df_2.columns[0:7]),
    drop_cols=[
        "id",
        "variant_type",
        "AC-1",
        "AC-2",
        "AF-1",
        "AF-2",
        "end_pos",
    ],
)
print(vcf_df_2)

print("===== TEST PASSED ====")
