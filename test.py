import gpugwas.io as gwasio
import cudf

# Test loading VCF
print("Test loading VCF to DF")
vcf_df = gwasio._load_vcf("data/test.vcf", info_keys=["*"], format_keys=["*"])
#vcf_df = gwasio.load_vcf("/home/jdaw/1kg-data/1kg.vcf", info_keys=["*"], format_keys=["*"])
#vcf_df.to_parquet(path="1kg_full_jdaw_v2.pqt", compression="auto", index=False)
print(vcf_df)

# Test loading annotation file
print("Test loading annotation file")
ann_df = gwasio.load_annotations("data/1kg_annotations.txt")
print(ann_df)

# Test loading VCF (Method 2)
print("Test loading VCF to DF (Method 2)")
vcf_df_2 = gwasio.load_vcf_variantworks(
    "data/test.vcf", info_keys=["AC", "AF"], format_keys=["GT"]
)
print(vcf_df_2)

print("===== TEST PASSED ====")
