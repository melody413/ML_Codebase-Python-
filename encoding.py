import pandas as pd, sys, numpy as np, os
from scipy import sparse
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import LabelEncoder
sys.path.insert(0,'/ds/data/ve2/dtl/aixx/phi/no_gbd/r000/work/pocep_ds/pocep_ds')
from config.PARAMETERS_global import *
from config.LOOKUP_objects import *
from utils.BASIC_input_output import *
from utils.EDA_helpers import *

'''Target (mean) Encoding'''
def target_encoding_fit(X,y,model):
    object_cols = select_dtype(X,['object', 'datetime64']).columns.values.tolist()
    object_cols = difference(object_cols,key_common) #Remove ID
    folder = 'target/aa/target' if model == 'a' else 'target/ltr/target'
    for col in object_cols:
        print("Encoding for column: {}".format(col))
        encoder = TargetEncoder(cols = [col])
        encoder.fit(X[col], y)
        write_encoder(encoder, folder, col)
    return

def target_encoding_transform(df,model):
    ## Apply the encoder to test data
    object_cols = select_dtype(df,['object', 'datetime64']).columns.values.tolist()
    object_cols = difference(object_cols,key_common) #Remove ID
    folder = 'target/aa/target' if model == 'a' else 'target/ltr/target'
    for col in object_cols:
        #Load encoder
        print("Encoding for column: {}".format(col))
        try:
            encoder = read_encoder(folder, col)
        except:
            print('Error occurred when reading column {}'.format(col))
        df[col] = encoder.transform(df[col])
    return df

'''Label Encoding'''
def label_encoding_fit(df, thresh):
    object_cols = select_dtype(df,['object']).columns.values.tolist()
    object_cols = difference(object_cols,key_common) #Remove ID
    for col in object_cols:
        print("Encoding for column: {}".format(col))
        ## Identify rare values in train dataset
        df[col] = df[col].replace('*', str(col) + '_none')
        col_df = df[col]
        df[col][col_df.groupby(col_df).transform('count').lt(thresh*len(df))] = 'rare'
        encoder = 'le_'  + str(col)
        encoder = LabelEncoder()
        try:
            df[col] = df[col].apply(lambda x :  str(x))
            df[col]  = encoder.fit_transform(df[col])
        except ValueError:
            print("Val error occurred for " + str(col))
        except re.error:
            print("Re error occurred for " + str(col))
        except TypeError:
            print("TypeError occurred for " + str(col))
        write_encoder(encoder, 'label/le', col)
    return

def label_encoding_transform(df):
    ## Apply the encoder to test data
    object_cols = select_dtype(df,['object']).columns.values.tolist()
    object_cols = difference(object_cols,key_common) #Remove ID    
    for col in object_cols:
        #Load encoder
        print("Encoding for column: {}".format(col))
        df[col] = df[col].replace('*', str(col) + '_none')
        encoder = LabelEncoder()
        try:
            encoder = read_encoder('label/le', col)
        except:
            print('Error occured when reading column {}'.format(col))
        print(encoder.classes_)

        #Identify and assign values
        df[col][~df[col].isin(encoder.classes_)] = 'rare'
        df[col] = encoder.transform(df[col])
    return df

def process_numerical(df, key, dtypes, remove_num_cols = None):
    cols = select_dtype(df,['int64','float64']).columns.values.tolist()
    cols = difference(cols,remove_num_cols) if(remove_num_cols) else cols
    df_num = df[key + cols]
    return df_num, cols

def process_target(df, key, target):
    cols = [target]
    df_target = df[key + cols]
    return df_target

def pre_process_categorical(df, key, path, dtypes, remove_cat_cols = None):
    p_size = 0
    object_cols = df.columns.values.tolist()
    object_cols = difference(object_cols, remove_cat_cols)
    write_path = 'ohe/'+path+'/ohe'
    for col in object_cols:
        if(df[col].dtype!='object'):
            df[col] = df[col].astype('str')
        temp_df = df[key  + [col]].drop_duplicates().rename(columns={col: "VAR"})
        temp_df['VAR'] = temp_df['VAR'].str.strip()
        temp_df.replace({'*': 'none'}, inplace = True)
        #temp_df = temp_df.loc[temp_df['VAR']!='*']
        temp_df['VAR'] = temp_df['VAR'].apply(lambda x : str(col) + '_' + str(x))
        write_encoder(temp_df, write_path, col)
        p_size = p_size + temp_df.shape[0]
        print("Finished processing column : {}".format(col))
    return p_size

def pre_process_categorical_test(df, key, path, dtypes, remove_cat_cols, col_dict):
    p_size = 0
    object_cols = df.columns.values.tolist()
    object_cols = difference(object_cols, remove_cat_cols)
    write_path = 'ohe/'+path+'/ohe'
    for col in object_cols:
        if(df[col].dtype!='object'):
            df[col] = df[col].astype('str')
        temp_df = df[key  + [col]].drop_duplicates().rename(columns={col: "VAR"})
        temp_df['VAR'] = temp_df['VAR'].str.strip()
        temp_df.replace({'*': 'none'}, inplace = True)
        #temp_df = temp_df.loc[temp_df['VAR']!='*']
        temp_df['VAR'] = temp_df['VAR'].apply(lambda x : str(col) + '_' + str(x))
        temp_seen_df = temp_df.loc[temp_df.VAR.isin(col_dict.VAR.values.tolist())]
        write_encoder(temp_seen_df, write_path , col)
        p_size = p_size + temp_df.shape[0]
        print("Finished processing column : {}".format(col))
    print(p_size)
    return p_size

def process_categorical(df, cat_size, thresh, path, dtypes, columns_to_remove):
    df_key = pd.DataFrame({'row': [0], 'VAR': ['No_VAR']})
    df_col = pd.DataFrame(np.repeat(df_key.values, cat_size, axis=0), columns=['row', 'VAR'])
    start_index = 0
    object_cols = df.columns.values.tolist()
    object_cols = difference(object_cols, columns_to_remove)
    #object_cols = dtype_dict_find_strings(dtypes)
    #object_cols = intersection(df.columns.values.tolist(),object_cols)
    #object_cols = difference(object_cols, columns_to_remove)    
    for col in object_cols:                                                      
        temp_df = pd.read_pickle(path + '/ohe_' + col + '.pkl')
        n = temp_df.shape[0]
        end_index = start_index + n
        df_col['VAR'][start_index:end_index] = temp_df['VAR'][:]
        df_col['row'][start_index:end_index] = temp_df['row'][:]
        start_index = end_index
        print("Finished processing column : {}".format(col))
    err_cnt = df_col.groupby(['VAR'], as_index=False)['row'].count().rename(columns={'row': 'count'})
    keep_err = err_cnt['VAR'][err_cnt['count'] > thresh]
    print(len(keep_err.unique()), " variables processed")
    df_col = df_col[df_col['VAR'].isin(keep_err)]
    #write_encoder(df_num, 'ohe/cat_processed', '')
    return df_col

def merge_to_model_ready(target, num, cat, path, key):
    #target = pd.read_pickle(path + 'target_processed.pkl')
    #num = pd.read_pickle(path + 'num_processed.pkl')
    #cat = pd.read_pickle(path + 'cat_processed.pkl')

    model_data = target[key]
    n_claims = model_data.shape[0]
    model_data = pd.merge(model_data, cat, on=key, how='left')
    model_data.fillna('No_VAR', inplace=True)
    n = model_data.shape[0]
    col_dict = model_data[['VAR']].drop_duplicates().sort_values(by=['VAR'])
    n_vars = col_dict.shape[0]
    col_dict['col'] = (range(n_vars))
    model_data = pd.merge(model_data, col_dict, on='VAR', how='left')
    vals = np.ones(n, dtype=float)
    rows = model_data['row']
    cols = model_data['col']

    for name in num.columns.values.tolist():
        if name != 'row':
            vals = np.concatenate((vals, num[name]))
            rows = np.concatenate((rows, num['row']))
            cols = np.concatenate((cols, n_vars*np.ones(len(num[name]), dtype=int)))
            new_col = pd.DataFrame({'VAR': [name], 'col': [n_vars]})
            col_dict = pd.concat([col_dict, new_col])
            n_vars = n_vars + 1

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n_claims, n_vars)) 

    sparse.save_npz(path + "/X.npz", X)
    save_csv(target, path + '/Y.csv')
    save_csv(col_dict, path + '/col_dict.csv')
    return

def merge_to_model_ready_test(target, num, cat, path_test, path_train, key):
    #target = pd.read_pickle(path + 'target_processed.pkl')
    #num = pd.read_pickle(path + 'num_processed.pkl')
    #cat = pd.read_pickle(path + 'cat_processed.pkl')

    model_data = target[key]
    n_claims = model_data.shape[0]
    model_data = pd.merge(model_data, cat, on=key, how='left')
    model_data.fillna('No_VAR', inplace=True)
    ## Variable differences
    col_dict_train = pd.read_csv(path_train + '/col_dict.csv')
    vars_train = col_dict_train.VAR.unique()
    vars_test = model_data.VAR.unique()
    vars_not_in_train = difference(vars_test,vars_train)
    vars_not_in_test = difference(vars_train,vars_test)
    vars_not_in_test_and_num = difference(vars_not_in_test,num.columns.values.tolist())

    ## Remove variables present in test but not in train
    size =  model_data.shape[0]
    model_data = model_data.loc[~model_data.VAR.isin(vars_not_in_train)]
    print("{} rows removed for variables which were present in test but not in train".
         format(size-model_data.shape[0]))

    # Add variables present in train but not in test ## DEFER
    model_data = model_data.append(pd.DataFrame({'row':[model_data.row.max()+1]*len(vars_not_in_test_and_num),  'VAR':vars_not_in_test_and_num}))

    col_dict = model_data[['VAR']].drop_duplicates().sort_values(by=['VAR'])
    n_vars = col_dict.shape[0]
    col_dict['col'] = (range(n_vars))
    model_data = pd.merge(model_data, col_dict, on='VAR', how='left')
    n = model_data.shape[0]
    vals = np.ones(n, dtype=float)
    rows = model_data['row']
    cols = model_data['col']

    for name in num.columns.values.tolist():
        if name != 'row':
            vals = np.concatenate((vals, num[name]))
            rows = np.concatenate((rows, num['row']))
            cols = np.concatenate((cols, n_vars*np.ones(len(num[name]), dtype=int)))
            new_col = pd.DataFrame({'VAR': [name], 'col': [n_vars]})
            col_dict = pd.concat([col_dict, new_col])
            n_vars = n_vars + 1

    n_claims += 1

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n_claims, n_vars)) 
    sparse.save_npz(path_test + "/X.npz", X)
    save_csv(target, path_test + '/Y.csv')
    save_csv(col_dict, path_test + '/col_dict.csv')
    return
