import argparse




parser = argparse.ArgumentParser(description='Run GWAS Pipeline')
parser.add_argument('--vcf_path', default = './data/test.vcf')
parser.add_argument('--annotation_path', default = './data/1kg_annotations.txt')
parser.add_argument('--workdir', default = './temp/')
parser.add_argument('--gpu', action='store_true')


if(args.gpu):
    import cudf
    import gpugwas.io as gwasio
    import gpugwas.processing as gwasproc
    vcf_df = gwasio._load_vcf(args.vcf_path, info_keys=["AC", "AF"], format_keys=["GT"])


    ann_df = gwasio._load_annotations(args.annotation_path)
    
    vcf_df = gwasproc.filter_vcf(vcf_df)
    
    vcf_df = gwasproc.sample_vcf(vcf_df)
    
    sampled_vcf_df = gwasproc.annotate_samples(vcf_df, ann_df)
    
    
    
    
    
    
    
    
else:
    import hail as hl
    hl.import_vcf(args.vcf_path).write(args.workdir + 'hail.mt', overwrite=True)
    mt = hl.read_matrix_table(args.workdir + 'hail.mt')
    table = (hl.import_table(args.annotation_path, impute=True)
         .key_by('Sample'))
    mt = mt.annotate_cols(pheno = table[mt.s])
    mt = hl.sample_qc(mt)
    mt = mt.filter_cols((mt.sample_qc.dp_stats.mean >= 4) & (mt.sample_qc.call_rate >= 0.97))
    ab = mt.AD[1] / hl.sum(mt.AD)

    filter_condition_ab = ((mt.GT.is_hom_ref() & (ab <= 0.1)) |
                        (mt.GT.is_het() & (ab >= 0.25) & (ab <= 0.75)) |
                        (mt.GT.is_hom_var() & (ab >= 0.9)))

    fraction_filtered = mt.aggregate_entries(hl.agg.fraction(~filter_condition_ab))
    mt = mt.filter_entries(filter_condition_ab)
    mt = hl.variant_qc(mt)
    mt = mt.filter_rows(mt.variant_qc.AF[1] > 0.01)
    mt = mt.filter_rows(mt.variant_qc.p_value_hwe > 1e-6)
    gwas = hl.linear_regression_rows(y=mt.pheno.CaffeineConsumption,
                                 x=mt.GT.n_alt_alleles(),
                                 covariates=[1.0])
    p = hl.plot.manhattan(gwas.p_value)





