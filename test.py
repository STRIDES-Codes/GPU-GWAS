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
