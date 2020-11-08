import argparse
import time

import gpugwas.io as gwasio
import gpugwas.processing as gwasproc

parser = argparse.ArgumentParser(description='Run GPU GWAS Pipeline')
parser.add_argument('--vcf_path', default = './data/test.vcf')
parser.add_argument('--annotation_path', default = './data/1kg_annotations.txt')
parser.add_argument('--workdir', default = './temp/')
args = parser.parse_args()

vcf_df = gwasio.load_vcf(args.vcf_path, info_keys=["AC", "AF"], format_keys=["GT"])
ann_df = gwasio.load_annotations(args.annotation_path)

# Start benchmarking after I/O
t0 = time.time()

vcf_df = gwasproc.filter_vcf(vcf_df)

vcf_df = gwasproc.sample_vcf(vcf_df)

sampled_vcf_df = gwasproc.annotate_samples(vcf_df, ann_df)
print('Time Elapsed: {}'.format(time.time()- t0))
