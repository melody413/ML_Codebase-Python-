import pandas as pd, csv, scipy.stats as ss, seaborn as sns, numpy as np, sys
sys.path.insert(0,'/ds/data/ve2/dtl/aixx/phi/no_gbd/r000/work/pocep_ds/pocep_ds')
from config.PARAMETERS_global import *
from utils.BASIC_input_output import *

def select_dtype(df, dtypes):
    df = df.select_dtypes(include=dtypes)
    return df

def remove_dtype(df, dtypes):
    df = df.select_dtypes(include=dtypes)
    return df

def df_correlation(df, columns):
    for col in columns:
        print(col)
        tmp = df[columns].apply(lambda x: x.corr(df[col]))
        #print(tmp)
        print(tmp.loc[tmp >= 0.5])
        print("Processing ended for column: {}".format(col))
    return
    
def df_skewed(df, columns):
    for col in columns:
        print(col)
        tmp = df[col].value_counts().head(20)/len(df)
        print(tmp.loc[tmp >= 0.8])
        #print("Processing ended for column: {}".format(col))
    return

def df_unique(df, columns):
    for col in columns:
        print(col, len(df[col].unique()))
    return

def df_outlier(df, columns):
    for col in columns:
        print(col)
        tmp = df[col].value_counts().tail(20)/len(df)
        print(tmp.loc[tmp <= 0.01])
        #print("Processing ended for column: {}".format(col))
    return

def generate_temporal_vars(df, cols):
    for col in cols:
        df[col] =  df[col].astype('datetime64')
        df['fe_' + str(col)+'_weekday'] = df[col].dt.weekday
        df['fe_' + str(col)+'_month'] = df[col].dt.month
        df['fe_' + str(col)+'_year'] = df[col].dt.year
        df['fe_' + str(col)+'_day'] = df[col].dt.day   
    return df

def df_count_nan(df):
    tmp = df.apply(lambda x : x.isnull().sum(axis=0))
    print(tmp.loc[tmp > 0])
    return
              
def df_strip_values(df):
    object_cols = select_dtype(df, ['object']).columns.values.tolist()
    df[object_cols] =  df[object_cols].apply(lambda x : x.str.strip())
    return df

def df_number_stats(df):
    df_int = select_dtype(df, ['int64', 'float64'])
    for col in df_int:
        print(col)
        print(round(df_int[col].describe(),2))
    return

def df_object_describe(df):
    df_obj = select_dtype(df, ['object'])
    for col in df_obj:
        print(col)
        print((df_obj[col].describe()))
    return

'''
Cramer's V method to calculate categorical correlation
'''
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

'''
Correlation among categorical variables
'''
def categorical_correlation(df, exceptions):
    df_obj = select_dtype(df, ['object'])
    object_cols = df_obj.columns.values.tolist()
    object_cols = difference(object_cols,exceptions)
    corr = {}
    for col1 in object_cols:
        for col2 in object_cols:
            try:
                correlation = cramers_v(df[col1],df[col2])
            except ValueError:
                print("Value error occurred for columns {} and {}".format(col1, col2))
            corr[str(col1) + "-" + str(col2)] = correlation
            if((col1!=col2) & (correlation >= 0.5)):
                print(col1, col2, corr[str(col1) + "-" + str(col2)])
    return

'''
Correlation of categoriacal features with a categorical target
'''
def categorical_correlation_w_target(df, exceptions, target =  target_aa):
    df_obj = select_dtype(df, ['object'])
    object_cols = df_obj.columns.values.tolist()
    object_cols = difference(object_cols,exceptions)
    for col in object_cols:
        print(col)
        print(cramers_v(df[col],df[target]))
    return

'''
Returns a df where modifier code is not * and does not match procedure modifier code
'''
def modifier_analysis(df, var1, var2):
    df_temp = df.loc[df[var1] != df[var2]][[var1,var2]]
    df_temp['combined'] = df_temp[var1] + df_temp[var2]
    df_temp_ = df_temp.loc[df_temp[var2]  != '* ']
    print(len(df_temp_), len(df_temp))
    return df_temp_
