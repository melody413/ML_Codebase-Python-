#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Input: Merged dataset (at DTL or HDR level)
Outptut: Folder with X (sparse matrix), Y (csv), col_dict (csv)
'''


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

#Import packages
import pandas as pd, sys, os, glob
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from category_encoders.target_encoder import TargetEncoder

sys.path.insert(0,'/ds/data/ve2/dtl/aixx/phi/no_gbd/r000/work/pocep_ds/pocep_ds')
sys.path.insert(0,'/ds/data/ve2/dtl/aixx/phi/no_gbd/r000/work/pocep_ds/pocep_ds/code')

from config.PARAMETERS_global import *
from config.LOOKUP_objects import *
from utils.BASIC_input_output import *
from models.CREATE_X_y import *
from utils.FUNCTIONS_feature_encoding import *

pd.options.display.max_rows = 100


# In[3]:


get_ipython().run_line_magic('reload_ext', 'autoreload')


# In[4]:


model_aapm_version


# In[5]:


'''
MODEL INPUTS
'''
model = 'aa' #aa - allowed amount, ltr - LTR
#model_name = model_aapm_version ## This will be the name of the folder
model_name = 'model_aapm_basic_subset_ap_prof_claims_5MM_num_cap_y_dates_n_binning_n_fe_proc_cd_grp_nbr_2020_3_months'
filename_master_data = '/' + date.today().strftime("%Y%m%d") + '_ver' + model_name +'.csv.gz'


# In[6]:


filename_master_data = '/20200914_vermodel_aapm_basic_subset_ap_prof_claims_5MM_num_cap_y_dates_n_binning_n_fe_proc_cd_grp_nbr_5MM.csv.gz'


# In[7]:


target  = target_aa if(model=='aa') else target_ltr
merge  = pd.read_csv(path_data_master + filename_master_data, compression = 'gzip')
merge = merge.loc[merge['CLM_PAYMNT_ACTN_1_CD']!='R'] if(model=='aa') else merge


# In[10]:


merge_ = merge.sample(n=2000000, random_state  = 42)


# In[12]:


merge = merge_.copy(deep = True)


# In[13]:


merge['binned_DTL_LINE_NBR'] = pd.cut(x = merge['DTL_LINE_NBR'], bins = [0,1,2,3,4,50])

keys = merge['binned_DTL_LINE_NBR'].value_counts().index.tolist()
values = ['1','2','3','4','4+']

values_dict = dict(zip(keys, values))
merge['binned_DTL_LINE_NBR'] = merge['binned_DTL_LINE_NBR'].map(values_dict)


# In[14]:


merge.drop(columns = ['KEY_CHK_DCN_ITEM_CD', 'DTL_LINE_NBR'], inplace = True)


# In[15]:


get_ipython().run_cell_magic('time', '', 'X_train, X_test, y_train, y_test = train_test_split_ratio(merge, target)')


# In[16]:


print(X_train.shape, X_test.shape)


# In[17]:


model_name


# ###### OHE Encoding

# In[18]:


key_ = key_dtl if(model=='aa') else key_hdr
dtypes = dict(dtl_dtypes).update(hdr_dtypes)
thresh = 0.001 #all the values which are less than  size*thresh will be dropped
train = pd.concat([X_train, y_train], axis = 1)
test = pd.concat([X_test, y_test], axis = 1)


# In[19]:


print(train.shape, test.shape)


# In[20]:


path = path_data_output + '/encoders/ohe/' + model_name
path_train = path + '/train'
path_test  = path + '/test'
path_pre_process_train = model_name + '/train/pre_process'
path_pre_process_test = model_name + '/test/pre_process'


# In[21]:


##Create folder structure for new model
try:
    os.mkdir(path)
    os.mkdir(path + '/train')
    os.mkdir(path + '/test')
    os.mkdir(path + '/train/pre_process')
    os.mkdir(path + '/test/pre_process')
except:
    print('Directory already present')
else:
    print('Directory created')


# Train Processing

# In[22]:


get_ipython().run_cell_magic('time', '', "n_claims = train.shape[0] # Number of rows\ntrain['row'] = (range(n_claims)) # add a row column with s.no.")


# In[23]:


get_ipython().run_cell_magic('time', '', "remove_num_cols = [key_common[0], target, 'row','HCFA_PT_CD','PAT_MBR_CD']  # -- Exceptions for numeric processing\n# remove_num_cols = [key_common[0], target, 'row', 'dtl_fe_month_MBR_CNTRCT_END_DT','dtl_fe_month_MBR_CNTRCT_EFCTV_DT',\n#                    'dtl_fe_month_SRVC_FROM_DT','hdr_fe_year_ILNS_ONSET_DT','dtl_fe_month_CLM_CMPLTN_DT',\n#                    'hdr_fe_month_PAT_BRTH_DT','hdr_fe_year_PAT_BRTH_DT','dtl_fe_year_MBR_CNTRCT_EFCTV_DT',\n#                    'hdr_fe_month_SRVC_FROM_DT','dtl_fe_year_CLM_CMPLTN_DT','HCFA_PT_CD','hdr_fe_month_SRVC_THRU_DT',\n#                    'dtl_fe_month_SRVC_TO_DT','dtl_fe_year_MBR_CNTRCT_END_DT','dtl_fe_year_SRVC_TO_DT',\n#                    'hdr_fe_year_SRVC_THRU_DT','dtl_fe_year_SRVC_FROM_DT','hdr_fe_year_SRVC_FROM_DT','PAT_MBR_CD',\n#                    'hdr_fe_year_CLM_CMPLTN_DT','hdr_fe_month_CLM_CMPLTN_DT','hdr_fe_month_ILNS_ONSET_DT']  # -- Exceptions for numeric processing\nnum_processed, numeric_cols = process_numerical(train, ['row'], dtypes, remove_num_cols)\ntarget_processed = process_target(train, ['row'], target)")


# In[24]:


get_ipython().run_cell_magic('time', '', "columns_to_remove = [key_common[0], target, 'row']\ncolumns_to_remove = columns_to_remove + numeric_cols + [target]\ncat_size = pre_process_categorical(train, ['row'], path_pre_process_train , dtypes, columns_to_remove)")


# In[25]:


get_ipython().run_cell_magic('time', '', "cat_processed  = process_categorical(train, cat_size, n_claims*thresh, path_train + '/pre_process', dtypes, columns_to_remove)")


# In[26]:


get_ipython().run_cell_magic('time', '', "merge_to_model_ready(target_processed, num_processed, cat_processed, path_train, 'row')")


# In[27]:


train_X_path =  '/ds/data/ve2/dtl/aixx/phi/no_gbd/r000/work/pocep_ds/pocep_ds/data/output/encoders/ohe/'+model_name+'/train/X.npz'
df_train = sparse.load_npz(train_X_path)
df_train.shape


# In[28]:


#Select the datset to delete
delete = 'train'


# In[29]:


#Remove pre_process files and directory
files = glob.glob(path + '/' + delete + '/pre_process/ohe_*.pkl')
for f in files:
    os.remove(f)
os.rmdir(path + '/' + delete + '/pre_process/')

#Remove all other files 
# files = glob.glob(path + '/' + delete + '/*')
# for f in files:
#     os.remove(f)
# os.rmdir(path + '/' + delete)


# Test Processing

# In[30]:


get_ipython().run_cell_magic('time', '', "#test.drop_duplicates(subset = key_, inplace  = True)  # df which has key_hdr values have been de-duplicated\nn_claims = test.shape[0] # Number of rows\ntest['row'] = (range(n_claims)) # add a row column with s.no.")


# In[31]:


get_ipython().run_cell_magic('time', '', "num_processed, numeric_cols = process_numerical(test, ['row'], dtypes, remove_num_cols)\ntarget_processed = process_target(test, ['row'], target)")


# In[32]:


get_ipython().run_cell_magic('time', '', "col_dict = pd.read_csv(path_train +  '/col_dict.csv')\ncat_size = pre_process_categorical_test(test, ['row'], path_pre_process_test , dtypes, columns_to_remove, col_dict)")


# In[33]:


get_ipython().run_cell_magic('time', '', "cat_processed  = process_categorical(test,cat_size, n_claims*thresh, path_test + '/pre_process', dtypes, columns_to_remove)")


# In[34]:


get_ipython().run_cell_magic('time', '', "merge_to_model_ready_test(target_processed, num_processed, cat_processed, path_test, path_train, 'row')")


# In[35]:


test_X_path =  '/ds/data/ve2/dtl/aixx/phi/no_gbd/r000/work/pocep_ds/pocep_ds/data/output/encoders/ohe/'+model_name+'/test/X.npz'
df_test = sparse.load_npz(test_X_path)
df_test.shape


# In[36]:


#Select the datset to delete
delete = 'test'


# In[37]:


#Remove pre_process files and directory
files = glob.glob(path + '/' + delete + '/pre_process/ohe_*.pkl')
for f in files:
    os.remove(f)
os.rmdir(path + '/' + delete + '/pre_process/')

#Remove all other files 
# files = glob.glob(path + '/' + delete + '/*')
# for f in files:
#     os.remove(f)
# os.rmdir(path + '/' + delete)


# In[41]:


#Remove main directory
# os.rmdir(path)


# In[96]:





# In[ ]:





# ######  Label Encoding

# In[9]:


thresh = 0.0001 #determines the count for 'rare' classification 


# In[10]:


get_ipython().run_cell_magic('time', '', 'label_encoding_fit(X_train, thresh)')


# In[11]:


get_ipython().run_cell_magic('time', '', 'X_train_  = label_encoding_transform(X_train)')


# In[12]:


get_ipython().run_cell_magic('time', '', 'X_test_  = label_encoding_transform(X_test)')


# ###### Target Encoding

# In[7]:


get_ipython().run_cell_magic('time', '', 'target_encoding_fit(X_train, y_train, model)')


# In[6]:


get_ipython().run_cell_magic('time', '', 'X_train_ = target_encoding_transform(X_train, model)')


# In[7]:


get_ipython().run_cell_magic('time', '', 'X_test_ = target_encoding_transform(X_test, model)')


# In[10]:


model_ready = path_data_model_ready + '/' + model_name + '/'
save_csv(X_train_, model_ready + 'target_encoded_X_train.csv.gz', compression = 'gzip')
save_csv(X_test_, model_ready + 'target_encoded_X_test.csv.gz', compression = 'gzip')
save_csv(y_train, model_ready + 'target_encoded_y_train.csv.gz', compression = 'gzip')
save_csv(y_test, model_ready + 'target_encoded_y_test.csv.gz', compression = 'gzip')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def sparse_binning(df, columns, thresh):
    '''takes columns to bin using the threshold as the % of sparsity over which the variables will be binned'''
    for col in columns:
        sparse_name =  str(col) + str('_sparse')
        value_counts = df[col].value_counts()*100/len(df)
        values = value_counts.loc[value_counts<thresh].index.values.tolist()
        values_dict = dict(zip(values, [sparse_name]*len(values)))
        df[col] = df[col].map(values_dict).fillna(df[col])
        print('Binning complete for column: {}'.format(col))
    return df

list(binning_variables['icd_codes']) + list(binning_variables['zip_codes'])

df = merge
columns = ['ICD_A_CD',	'ICD_A_POA_CD',	'ICD_B_CD',	'ICD_B_POA_CD',	'ICD_C_CD',	'ICD_C_POA_CD',	'ICD_D_CD',	'ICD_D_POA_CD',	'ICD_E_CD',	'ICD_E_POA_CD',	'MDFR_1_CD',	'MDFR_2_CD',	'MDFR_3_CD']
thresh = 0.01
merge = sparse_binning(df, columns, thresh) 

merge['binned_DTL_LINE_NBR'] = pd.cut(x = merge['DTL_LINE_NBR'], bins = [0,1,2,3,4,50])

keys = merge['binned_DTL_LINE_NBR'].value_counts().index.tolist()
values = ['1','2','3','4','4+']

values_dict = dict(zip(keys, values))
merge['binned_DTL_LINE_NBR'] = merge['binned_DTL_LINE_NBR'].map(values_dict)

#-- Remove repetetive columns 
dates = ['ILNS_ONSET_DT', 'MBR_CNTRCT_EFCTV_DT', 'MBR_CNTRCT_END_DT',  'PAT_BRTH_DT', 'SRVC_TO_DT', 'SRVC_THRU_DT', 
         'SRVC_FROM_DT', 'CLM_CMPLTN_DT']
repeats = ['DTL_LINE_NBR',  'hdr_fe_norm_TOTL_CHRG_AMT', 'dtl_fe_norm_BILLD_CHRGD_AMT', 'dtl_fe_norm_BILLD_SRVC_UNIT_QTY',
           'dtl_fe_norm_TOTL_UNITS_PRCD_CNT', 'dtl_fe_norm_UNITS_OCR_NBR']

merge.shape

merge.drop(columns = dates + repeats, inplace = True)

merge.shape

##rough end


# In[ ]:





# In[ ]:





# ######  Label Encoding

# In[9]:


thresh = 0.0001 #determines the count for 'rare' classification 


# In[10]:


get_ipython().run_cell_magic('time', '', 'label_encoding_fit(X_train, thresh)')


# In[11]:


get_ipython().run_cell_magic('time', '', 'X_train_  = label_encoding_transform(X_train)')


# In[12]:


get_ipython().run_cell_magic('time', '', 'X_test_  = label_encoding_transform(X_test)')


# ###### Target Encoding

# In[7]:


get_ipython().run_cell_magic('time', '', 'target_encoding_fit(X_train, y_train, model)')


# In[6]:


get_ipython().run_cell_magic('time', '', 'X_train_ = target_encoding_transform(X_train, model)')


# In[7]:


get_ipython().run_cell_magic('time', '', 'X_test_ = target_encoding_transform(X_test, model)')


# In[10]:


model_ready = path_data_model_ready + '/' + model_name + '/'
save_csv(X_train_, model_ready + 'target_encoded_X_train.csv.gz', compression = 'gzip')
save_csv(X_test_, model_ready + 'target_encoded_X_test.csv.gz', compression = 'gzip')
save_csv(y_train, model_ready + 'target_encoded_y_train.csv.gz', compression = 'gzip')
save_csv(y_test, model_ready + 'target_encoded_y_test.csv.gz', compression = 'gzip')


# In[ ]:





# In[ ]:





# In[ ]:




