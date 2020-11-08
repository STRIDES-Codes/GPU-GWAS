import cudf

def filter_samples(df, min_dp_mean = 4, min_call_rate=0.95):
    
    print("Number of samples: " + str(len(df['sample'].unique())))
    
    # Filter DP
    dp_filter = df.groupby('sample').call_DP.mean() >= min_dp_mean
    dp_filter = dp_filter[dp_filter.values].index
    df_filtered = df[df['sample'].isin(dp_filter)]
    print("Number of samples after filtering DP: " + str(len(df_filtered['sample'].unique())))
    
    # Filter call rate
    df_filtered['called'] = df_filtered.call_GT!=-1
    call_rate_filter = df_filtered.groupby('sample').called.agg(['sum', 'count'])
    call_rate_filter['rate'] = call_rate_filter.iloc[:,0]/call_rate_filter.iloc[:,1]
    call_rate_filter = call_rate_filter.rate > min_call_rate
    call_rate_filter = call_rate_filter[call_rate_filter].index
    df_filtered = df_filtered[df_filtered['sample'].isin(call_rate_filter)]
    print("Number of samples after filtering sample call rate: " + str(len(df_filtered['sample'].unique())))
    
    return df_filtered


def filter_variants(df, min_af = 0.1, min_call_rate=0.95):
    
    print("Number of variants: " + str(len(df.feature_id.unique())))
        
    # Filter AF
    df_filtered = df[df['AF'] >= min_af]
    print("Number of variants after filtering AF: " + str(len(df_filtered.feature_id.unique())))
    
    # Filter call rate
    df_filtered['called'] = df_filtered.call_GT!=-1
    call_rate_filter = df_filtered.groupby('feature_id').called.agg(['sum', 'count'])
    call_rate_filter['rate'] = call_rate_filter.iloc[:,0]/call_rate_filter.iloc[:,1]
    call_rate_filter = call_rate_filter.rate > min_call_rate
    call_rate_filter = call_rate_filter[call_rate_filter].index
    df_filtered = df_filtered[df_filtered['feature_id'].isin(call_rate_filter)]
    print("Number of variants after filtering call rate: " + str(len(df_filtered.feature_id.unique())))
    
    return df_filtered