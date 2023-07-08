#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Read dataset
Train_data_1 = pd.read_csv("new_123_789.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
Train_data_2 = pd.read_csv("Train_Data2.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
Train_data_3 = pd.read_csv("Train_data3.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
Train_data_4 = pd.read_csv("Train_data4.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
dataset = pd.concat([Train_data_1, Train_data_2, Train_data_3, Train_data_4], ignore_index = True)


# In[3]:


# Split train, test and validation
target ='ELGBL_EXPNS_AMT'
X = dataset.drop(columns = target)
y = dataset[target]


# In[6]:


same_value_columns = same_values(X, 0.975)


# In[7]:


same_value_columns


# In[8]:


X['MDFR_1_CD'].value_counts()


# In[ ]:





# In[47]:


def pre_process_data(df):
    #Columns to remove based on business logic
    #BILLD_CHRGD_AMT
    columns_to_remove = ['PAYMNT_AMT', 'NOT_CVRD_AMT', 'BSIC_CPAYMNT_AMT', 'MM_CPAYMNT_AMT',
                         'MM_DDCTBL_AMT', 'CPAYMNT_AMT', 'CPAYMNT_TYPE_AMT','BSIC_DDCTBL_AMT', 'PN_ID', 'PN_VRTN_ID',
                         'MEM_RESP', 'AUTO_ADUJ', 'COB_SGMNT_CNT', 'MEDCR_CNT', 'DTL_SGMNT_CNT', 'EOB_DNL_CD']
    exception_cols = ['BILLD_CHRGD_AMT']
    df.drop(columns = columns_to_remove, inplace = True)
    
    #Columns which have the same value for 97.5% of the rows 
    same_value_columns = same_values(df, 0.975)
    df.drop(columns = same_value_columns, inplace = True)
    
    # Convert to int (Manual identification)
    df['TOTL_UNITS_PRCD_CNT'] = df['TOTL_UNITS_PRCD_CNT'].astype('float64')
    
    # Create variables and convert to string
    df['yr'] = pd.DatetimeIndex(df['SRVC_THRU_DT']).year.astype('str')
    df['mnth'] = pd.DatetimeIndex(df['SRVC_THRU_DT']).month.astype('str')
    df['day_of_week'] = pd.DatetimeIndex(df['SRVC_THRU_DT']).dayofweek.astype('str')
   # Drop date variables
    df.drop(columns = ['SRVC_THRU_DT','SRVC_FROM_DT'], inplace = True)
    
    # String columns with less than or equal to 15 unique values (for OHE)
    unique_cols = unique_counts(df,15)
    
    # Convert to OHE
    df = ohe(df,unique_cols,exception_cols)
    
    # Columns which are highly correlated above a certain threshold. (manually identify one variable to keep)
    columns_highly_correlated = correlation(df,0.85)
    df.drop(columns = columns_highly_correlated, inplace = True)
    
    return (df)


# In[48]:


X_ = pre_process_data(X)


# In[50]:


train_X, test_X_, train_Y, test_Y_ = train_test_split(X_, y, test_size=0.4, random_state=42)
test_X, val_X, test_Y, val_Y = train_test_split(test_X_, test_Y_, test_size=0.5, random_state=42)


# In[51]:


from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=5, random_state=42)
regr.fit(train_X, train_Y)


# In[88]:


test_pred = regr.predict(test_X)


# In[89]:


mse = mean_squared_error(test_pred, c)
rmse = np.sqrt(mse)
print(rmse)


# In[92]:


regr.score(test_X,test_Y)


# In[93]:


model.score(test_X,test_Y)


# In[83]:


for i,col in enumerate(test_X.columns.values.tolist()):
    print(col,'{0:.4f}'.format(regr.feature_importances_[i]))


# In[98]:


model = DecisionTreeRegressor(max_depth=5)


# In[99]:


model.fit(train_X, train_Y)


# In[100]:


test_pred = model.predict(test_X)


# In[101]:


mse = mean_squared_error(test_pred, test_Y)
rmse = np.sqrt(mse)
print(rmse)


# In[90]:


for i,col in enumerate(test_X.columns.values.tolist()):
    print(col,'{0:.4f}'.format(model.feature_importances_[i]))


# In[18]:


from sklearn.metrics import roc_auc_score
# Actual class predictions
rf_predictions = model.predict(test_X)
# Probabilities for each class
rf_probs = model.predict_proba(test_X)[:, 1]


# In[36]:


test_pred.value_counts()


# In[35]:


test_Y.value_counts()


# In[81]:


metrics.accuracy_score(test_Y_.tolist(),test_pred)


# In[ ]:





# In[4]:


def same_values(df, threshold):
    cols = []
    for col in df.columns.values.tolist():
        null_pct = len(df.loc[df[col].isna() == True])/len(df)
        if(null_pct >= threshold):
            cols.append(col)
        else:
            same_pct = df[col].value_counts()[0]/len(df)
            if(same_pct >= threshold):
                cols.append(col)
    return cols


# In[38]:


def unique_counts(df, threshold):
    unique_cols= []
    for col in df.columns.values.tolist():
        if (df[col].nunique() <= threshold):
            unique_cols.append(col)
    return (unique_cols)


# In[12]:


def correlation(df, threshold):
    corr = df.corr()
    correlations = []
    for col in corr.columns.values.tolist():
        for col_row in corr.index.values.tolist():
            if (col != col_row):
                if (corr[col][col_row] >= threshold):
                    correlations.append(col)
                    correlations.append(col_row)
    return list(set(correlations))


# In[45]:


def ohe(df,col_list):
    columns_list = df.columns.values.tolist()
    columns_list.remove('BILLD_CHRGD_AMT')
    for col in columns_list:
        if col in col_list:
            ohe = pd.DataFrame()
            if(df[col].dtypes == 'object'):
                ohe = pd.get_dummies(df[col])
                ohe.columns = [col + "_" + ohe_col for ohe_col in ohe.columns.values.tolist()]
                df = df.join(ohe)
        df.drop(col,axis = 1, inplace = True)
    return df


# #### BTS

# In[ ]:


## Finding variables which are NULL 


# In[3]:


## Finding variables which are highly correlated
corr = dataset.corr()                
for col in dataset.columns.values.tolist():
    for col_row in corr.index.values.tolist():
        if (col != col_row):
            if (corr[col][col_row] > 0.84):
                print(str(corr[col][col_row]), col, col_row)
correlations = []
for col in corr.columns.values.tolist():
    for col_row in corr.index.values.tolist():
        correlation = []
        if (col != col_row):
            if (corr[col][col_row] >= 0.8):
                correlation.append(col)
                correlation.append(col_row)
                correlations.append(correlation)


# In[4]:


## Find string variables with less than 16 unique values
for col in dataset.columns.values.tolist():
    if (dataset[col].nunique() <= 15):
        print ("\'" + col  +"\',")


# In[ ]:


# Convert to OHE - def 


# In[ ]:


test_X = tes


# In[5]:


dataset = pd.concat([Train_data_1, Train_data_2, Train_data_3, Train_data_4], ignore_index = True)


# In[6]:


# Remove columns 
dataset.drop(columns = columns_to_remove, inplace = True)
dataset.drop(columns = columns_97_5, inplace = True)


# In[7]:


# Convert to int
dataset['TOTL_UNITS_PRCD_CNT'] = dataset['TOTL_UNITS_PRCD_CNT'].astype('float64')


# In[8]:


# Create variables 
dataset['yr'] = pd.DatetimeIndex(dataset['SRVC_THRU_DT']).year
dataset['mnth'] = pd.DatetimeIndex(dataset['SRVC_THRU_DT']).month
dataset['day_of_week'] = pd.DatetimeIndex(dataset['SRVC_THRU_DT']).dayofweek


# In[9]:


# Convert to string 
dataset['yr'] = dataset['yr'].astype('str')
dataset['mnth'] = dataset['mnth'].astype('str')
dataset['day_of_week'] = dataset['day_of_week'].astype('str')


# In[10]:


# Drop date variables
dataset.drop(columns = ['SRVC_THRU_DT','SRVC_FROM_DT'], inplace = True)


# In[11]:


# Drop string variables with more than 50 values. Go into detail of each one and try to club them to create indicators


# In[12]:


columns_under_15 = ['PROV_TAX_ID','PROV_NM','PROV_STR_ADRS','ROV_ZIP_5_CD','PROV_PAYENT_LCTN_CD','MX_PRCG_VRTN_CD',
                    'SCRN_FRMT_CD','MIXER_PARG_IND','CLM_TYP','NUM_LINES','HCFA_PT_CD','CLM_TYPE_CD','TELEHEALTH',
                    'PROD_DESC','NEW_CLM_TYP','UM_RQRD_IND','CLM_PAYMNT_ACTN_1_CD','yr','mnth','day_of_week']


# In[13]:


columns_highly_correlated = ['PROV_NM', 'PROV_STR_ADRS', 'ROV_ZIP_5_CD', 'PROV_PAYENT_LCTN_CD', 'CLM_TYP', 
                             'UM_RQRD_IND', 'MX_PRCG_VRTN_CD',  'MIXER_PARG_IND', 'HCFA_PT_CD',  'TELEHEALTH']


# In[14]:


dataset.drop(columns = columns_highly_correlated, inplace = True)


# In[15]:


train_X = dataset.drop(columns = target)
train_Y = dataset[target]


# In[16]:


# Convert to OHE
for col in train_X.columns.values.tolist():
    if col in columns_under_15:
        ohe = pd.DataFrame()
        if(train_X[col].dtypes == 'object'):
            ohe = pd.get_dummies(train_X[col])
            ohe.columns = [col + "_" + ohe_col for ohe_col in ohe.columns.values.tolist()]
            train_X = train_X.join(ohe)
    train_X.drop(col,axis = 1, inplace = True)


# In[17]:


dataset.shape


# In[18]:


# Find correlation of all columns with each other
corr = dataset.corr()


# In[19]:


for col in dataset.columns.values.tolist():
    for col_row in corr.index.values.tolist():
        if (col != col_row):
            if (corr[col][col_row] > 0.84):
                print(str(corr[col][col_row]), col, col_row)


# In[20]:


# train without correlation


# In[21]:


# Find correlation and create a list of list
correlations = []
for col in corr.columns.values.tolist():
    for col_row in corr.index.values.tolist():
        correlation = []
        if (col != col_row):
            if (corr[col][col_row] >= 0.8):
                correlation.append(col)
                correlation.append(col_row)
                correlations.append(correlation)


# In[19]:


model = RandomForestClassifier(n_estimators=100,  bootstrap = True)


# In[20]:


model.fit(train_X, train_Y)


# In[21]:





# In[ ]:


from sklearn.metrics import roc_auc_score
# Actual class predictions
rf_predictions = model.predict(test)
# Probabilities for each class
rf_probs = model.predict_proba(test)[:, 1]


# In[14]:


Process_path = 'Train'


# In[15]:


if Process_path == "Train":
    ### Training Dataset ############
    Train_data_1 = pd.read_csv("new_123_789.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
    Train_data_2 = pd.read_csv("Train_Data2.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
    Train_data_3 = pd.read_csv("Train_data3.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
    Train_data_4 = pd.read_csv("Train_data4.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
    Train_data_4 = Train_data_4[:9203]
    dataset = pd.concat([Train_data_1, Train_data_2, Train_data_3, Train_data_4], ignore_index = True)
    print(dataset.shape)
    dataset.insert(0, 'New_ID', range(800000000, 800000000 + len(dataset)))
    dataset.drop(dataset[dataset['CLM_PAYMNT_ACTN_1_CD'] > 'P'].index, inplace = True)
    dataset1 = dataset[['New_ID',"DAIG1","PROC_CD","PRCG_ZIP_ST_CD","PROV_TAX_ID","PROV_ST_CD","PROV_SPCLTY_CD","BILLG_NPI","RNDRG_NPI","PROV_PAYENT_LCTN_CD","SRVC_FCLTY_LCTN_ID","SRVC_FCLTY_LCTN_NPI","POT_CD","PN_ID","PN_VRTN_ID","MBR_CNTRCT_CD","MBR_CVRG_PRCG_VRTN_CD","MBR_PROD_CD","HCFA_PT_CD","CLM_TYPE_CD","RNDRG_LINE_1_ADRS","RNDRG_CITY_NM","SRC_CD","PROV_RGN_CD","UM_RQRD_IND","NEW_CLM_TYP_1","CASE_NBR","ROV_ZIP_5_CD","PROV_PRC_ZIP_4_CD"]]
    dataset2 = dataset[["BILLD_CHRGD_AMT","ELGBL_EXPNS_AMT","TOTL_UNITS_PRCD_CNT"]]
    New_dataset1 = handle_non_numerical_data(dataset1)
    New_dataset2 = dataset2
    Final_Dataset = pd.concat([New_dataset1, New_dataset2], axis=1, join='inner')
    X_train = Final_Dataset[['New_ID',"DAIG1","PROC_CD","PRCG_ZIP_ST_CD","PROV_TAX_ID","PROV_ST_CD","PROV_SPCLTY_CD","BILLG_NPI","RNDRG_NPI","PROV_PAYENT_LCTN_CD","SRVC_FCLTY_LCTN_ID","SRVC_FCLTY_LCTN_NPI","POT_CD","PN_ID","PN_VRTN_ID","MBR_CNTRCT_CD","MBR_CVRG_PRCG_VRTN_CD","MBR_PROD_CD","HCFA_PT_CD","CLM_TYPE_CD","RNDRG_LINE_1_ADRS","RNDRG_CITY_NM","SRC_CD","PROV_RGN_CD","UM_RQRD_IND","NEW_CLM_TYP_1","CASE_NBR","ROV_ZIP_5_CD","PROV_PRC_ZIP_4_CD","BILLD_CHRGD_AMT","TOTL_UNITS_PRCD_CNT"]].values
    y_train = Final_Dataset["ELGBL_EXPNS_AMT"].values
    #### LINEAR Regression Model#### (Doestn't work with different types of data, need to change the New_dataset2 to Float)
    #model = LinearRegression()  
    #model.fit(X_train, y_train)

    ## Descision Model ### (Getting 100% Accuracy Lower accuracy as test Dataset increses, Able to handle larger data set)
    #model = DecisionTreeClassifier(random_state=RSEED)
    #model = DecisionTreeClassifier(criterion="entropy", max_depth=25)
    model = DecisionTreeClassifier(criterion="entropy")
    #model = DecisionTreeClassifier(criterion="entropy",max_depth=25,random_state = 100,max_features = "auto", min_samples_leaf = 50)
    model.fit(X_train, y_train)

    ######## Random Forest Model ## (getting Around 90% of accuracy)
    #########################################
    #### Create the model with 100 trees
    #model = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')
    #model.fit(X_train, y_train)

    #######Logestic Regression Model ################ (Only getting 10% Accuracy)
    #model = LogisticRegression(C=0.7,random_state=42)
    #model.fit(X_train, y_train)

    #### Load Model to the drive #######
    pickle.dump(model, open(filename, 'wb'))


# In[16]:


### Training Dataset ############
Train_data_1 = pd.read_csv("new_123_789.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
Train_data_2 = pd.read_csv("Train_Data2.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
Train_data_3 = pd.read_csv("Train_data3.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
Train_data_4 = pd.read_csv("Train_data4.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
Train_data_4 = Train_data_4[:9203]
dataset = pd.concat([Train_data_1, Train_data_2, Train_data_3, Train_data_4], ignore_index = True)
print(dataset.shape)
dataset.insert(0, 'New_ID', range(800000000, 800000000 + len(dataset)))
dataset.drop(dataset[dataset['CLM_PAYMNT_ACTN_1_CD'] > 'P'].index, inplace = True)
dataset1 = dataset[['New_ID',"DAIG1","PROC_CD","PRCG_ZIP_ST_CD","PROV_TAX_ID","PROV_ST_CD","PROV_SPCLTY_CD","BILLG_NPI","RNDRG_NPI","PROV_PAYENT_LCTN_CD","SRVC_FCLTY_LCTN_ID","SRVC_FCLTY_LCTN_NPI","POT_CD","PN_ID","PN_VRTN_ID","MBR_CNTRCT_CD","MBR_CVRG_PRCG_VRTN_CD","MBR_PROD_CD","HCFA_PT_CD","CLM_TYPE_CD","RNDRG_LINE_1_ADRS","RNDRG_CITY_NM","SRC_CD","PROV_RGN_CD","UM_RQRD_IND","NEW_CLM_TYP_1","CASE_NBR","ROV_ZIP_5_CD","PROV_PRC_ZIP_4_CD"]]
dataset2 = dataset[["BILLD_CHRGD_AMT","ELGBL_EXPNS_AMT","TOTL_UNITS_PRCD_CNT"]]
New_dataset1 = handle_non_numerical_data(dataset1)
New_dataset2 = dataset2
Final_Dataset = pd.concat([New_dataset1, New_dataset2], axis=1, join='inner')


# In[51]:


columns_train = ['New_ID',"DAIG1","PROC_CD","PRCG_ZIP_ST_CD","PROV_TAX_ID","PROV_ST_CD","PROV_SPCLTY_CD","BILLG_NPI","RNDRG_NPI","PROV_PAYENT_LCTN_CD","SRVC_FCLTY_LCTN_ID","SRVC_FCLTY_LCTN_NPI","POT_CD","PN_ID","PN_VRTN_ID","MBR_CNTRCT_CD","MBR_CVRG_PRCG_VRTN_CD","MBR_PROD_CD","HCFA_PT_CD","CLM_TYPE_CD","RNDRG_LINE_1_ADRS","RNDRG_CITY_NM","SRC_CD","PROV_RGN_CD","UM_RQRD_IND","NEW_CLM_TYP_1","CASE_NBR","ROV_ZIP_5_CD","PROV_PRC_ZIP_4_CD","TOTL_UNITS_PRCD_CNT", 'BILLD_CHRGD_AMT']


# In[52]:


for col in columns_train:
    print(col)


# In[44]:


X_train = Final_Dataset[columns_train]


# In[41]:


Final_Dataset['BILLD_CHRGD_AMT'] = Final_Dataset['BILLD_CHRGD_AMT'].astype('float')
Final_Dataset['TOTL_UNITS_PRCD_CNT'] = Final_Dataset['TOTL_UNITS_PRCD_CNT'].astype('float')
Final_Dataset['ELGBL_EXPNS_AMT'] = Final_Dataset['ELGBL_EXPNS_AMT'].astype('str')


# In[42]:


for col in X_train:
#     print(col,type(Final_Dataset[col][0]))
   print(col, Final_Dataset['ELGBL_EXPNS_AMT'].corr(Final_Dataset[col]))


# In[ ]:





# In[ ]:





# In[46]:


Final_Dataset['ELGBL_EXPNS_AMT'] = Final_Dataset['ELGBL_EXPNS_AMT'].astype('str')


# In[47]:



    
    y_train = Final_Dataset["ELGBL_EXPNS_AMT"].values
    #### LINEAR Regression Model#### (Doestn't work with different types of data, need to change the New_dataset2 to Float)
    #model = LinearRegression()  
    #model.fit(X_train, y_train)

    ## Descision Model ### (Getting 100% Accuracy Lower accuracy as test Dataset increses, Able to handle larger data set)
    #model = DecisionTreeClassifier(random_state=RSEED)
    #model = DecisionTreeClassifier(criterion="entropy", max_depth=25)
    model = DecisionTreeClassifier(criterion="entropy")
    #model = DecisionTreeClassifier(criterion="entropy",max_depth=25,random_state = 100,max_features = "auto", min_samples_leaf = 50)
    model.fit(X_train, y_train)

    ######## Random Forest Model ## (getting Around 90% of accuracy)
    #########################################
    #### Create the model with 100 trees
    #model = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')
    #model.fit(X_train, y_train)

    #######Logestic Regression Model ################ (Only getting 10% Accuracy)
    #model = LogisticRegression(C=0.7,random_state=42)
    #model.fit(X_train, y_train)

    #### Load Model to the drive #######
    pickle.dump(model, open(filename, 'wb'))


# In[ ]:





# In[ ]:





# In[48]:


Process_path = 'Test'


# In[49]:


if Process_path == "Test":
  ###### Testing Dataset ############
  #testset = pd.read_csv("Test_Date.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
#    testset = pd.read_csv("Test_Data_3.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
    Train_data_4 = pd.read_csv("Train_data4.csv",sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
    testset = Train_data_4[:500]
    testset.insert(0, 'New_ID', range(900000000, 900000000 + len(testset)))
    #testset.drop(testset[testset['CLM_PAYMNT_ACTN_1_CD'] > 'P'].index, inplace = True)
    testset1 = testset[['New_ID',"DAIG1","PROC_CD","PRCG_ZIP_ST_CD","PROV_TAX_ID","PROV_ST_CD","PROV_SPCLTY_CD","BILLG_NPI","RNDRG_NPI","PROV_PAYENT_LCTN_CD","SRVC_FCLTY_LCTN_ID","SRVC_FCLTY_LCTN_NPI","POT_CD","PN_ID","PN_VRTN_ID","MBR_CNTRCT_CD","MBR_CVRG_PRCG_VRTN_CD","MBR_PROD_CD","HCFA_PT_CD","CLM_TYPE_CD","RNDRG_LINE_1_ADRS","RNDRG_CITY_NM","SRC_CD","PROV_RGN_CD","UM_RQRD_IND","NEW_CLM_TYP_1","CASE_NBR","ROV_ZIP_5_CD","PROV_PRC_ZIP_4_CD"]]
    testset2 = testset[["BILLD_CHRGD_AMT","ELGBL_EXPNS_AMT","TOTL_UNITS_PRCD_CNT"]]
    #testset1 = testset[["DAIG1","PROC_CD","PROV_TAX_ID"]]
    New_testset1 = handle_non_numerical_data_Test(testset1)
    New_testset2 = testset2
    Final_testset = pd.concat([New_testset1, New_testset2], axis=1, join='inner')
    X_test = Final_testset[['New_ID',"DAIG1","PROC_CD","PRCG_ZIP_ST_CD","PROV_TAX_ID","PROV_ST_CD","PROV_SPCLTY_CD","BILLG_NPI","RNDRG_NPI","PROV_PAYENT_LCTN_CD","SRVC_FCLTY_LCTN_ID","SRVC_FCLTY_LCTN_NPI","POT_CD","PN_ID","PN_VRTN_ID","MBR_CNTRCT_CD","MBR_CVRG_PRCG_VRTN_CD","MBR_PROD_CD","HCFA_PT_CD","CLM_TYPE_CD","RNDRG_LINE_1_ADRS","RNDRG_CITY_NM","SRC_CD","PROV_RGN_CD","UM_RQRD_IND","NEW_CLM_TYP_1","CASE_NBR","ROV_ZIP_5_CD","PROV_PRC_ZIP_4_CD","TOTL_UNITS_PRCD_CNT"]].values
    y_test = Final_testset["ELGBL_EXPNS_AMT"].values
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=RSEED)
    #### Read Model from Drive ########
    model = pickle.load(open(filename, 'rb'))
    
    #Actual Pridiction
    y_pred = model.predict(X_test)
    new_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print("Test Data Accuracy: ({0:.4f})".format(metrics.accuracy_score(y_test,y_pred)))
    ##print("Test Data Accuracy: ({0:.4f})".format(metrics.accuracy_score(y_test,y_pred)))
    new_df1 = pd.DataFrame({'New_ID': [i[0] for i in X_test],'Actual': y_test, 'Predicted': y_pred})
    ##Final_Resultset = pd.merge(testset, new_df1, on='New_ID') 
    Final_Resultset = pd.merge(testset, new_df1, on='New_ID')
    Final_Resultset1 = Final_Resultset[["New_ID","KEY_CHK_DCN_NBR","KEY_CHK_DCN_ITEM_CD","KEY_CHK_DCN_CENTRY_CD","BILLD_CHRGD_AMT","Actual","Predicted"]]
    #print(Final_Resultset1)

Final_Resultset1.head(20)


# In[ ]:





# #### Misc

# In[ ]:


strings_vars = ['KEY_CHK_DCN_NBR','DAIG1', 'DAIG2', 'DAIG3', 'DAIG4', 'DAIG5', 'PROC_CD', 'PROV_SCNDRY_NM',
                'PROV_PRC_ZIP_4_CD', 'RNDRG_NPI', 'PROV_SPCLTY_CD', 'SRVC_FCLTY_LCTN_NPI', 'MBR_CNTRCT_CD',
                'MBR_CVRG_PRCG_VRTN_CD', 'MBR_PROD_CD', 'RNDRG_LINE_1_ADRS', 'RNDRG_CITY_NM', 'CLM_PAYMNT_ACTN_2_6_CD',
                'CASE_NBR', 'SRVC_FROM_DT', 'SRVC_THRU_DT']
categorical_vars = ['KEY_CHK_DCN_ITEM_CD', 'KEY_CHK_DCN_CENTRY_CD', 'MDFR_1_CD', 'MDFR_2_CD', 'MDFR_3_CD', 
                    'PRCG_ZIP_ST_CD', 'PROV_TAX_ID', 'PROV_NM', 'PROV_STR_ADRS', 'ROV_ZIP_5_CD', 'PROV_ST_CD',
                    'BILLG_NPI', 'PROV_PAYENT_LCTN_CD', 'SRVC_FCLTY_LCTN_ID', 'BSIC_DDCTBL_AMT', 'POT_CD',
                    'MX_PRCG_VRTN_CD', 'MX_PROV_PRCG_PROD_CD', 'PN_ID', 'PN_VRTN_ID', 'SCRN_FRMT_CD', 'MIXER_PARG_IND',
                    'MEM_RESP', 'CLM_TYP', 'NUM_LINES', 'HCFA_PT_CD', 'CLM_TYPE_CD', 'AUTO_ADUJ', 'HCPCS_MDFR_CD',
                    'PAY_AUTHRZN_CD', 'COB_SGMNT_CNT', 'MEDCR_CNT', 'DTL_SGMNT_CNT', 'PROV_GROUP', 'RNDRG_LINE_2_ADRS', 
                    'TELEHEALTH', 'SRC_CD', 'PROD_DESC', 'NEW_CLM_TYP', 'PROV_RGN_CD', 'UM_RFRL_TYPE_RQRD_IND',
                    'UM_RQRD_IND', 'NEW_CLM_TYP_1', 'CLM_PAYMNT_ACTN_1_CD', 'EOB_DNL_CD']


# In[ ]:


def pre_process_data(df):
    #Columns to remove based on business logic
    columns_to_remove = ['BILLD_CHRGD_AMT',  'PAYMNT_AMT', 'NOT_CVRD_AMT', 'BSIC_CPAYMNT_AMT', 
                    'MM_CPAYMNT_AMT', 'MM_DDCTBL_AMT', 'CPAYMNT_AMT', 'CPAYMNT_TYPE_AMT','BSIC_DDCTBL_AMT', 
                    'PN_ID', 'PN_VRTN_ID', 'MEM_RESP', 'AUTO_ADUJ', 'COB_SGMNT_CNT', 'MEDCR_CNT', 'DTL_SGMNT_CNT', 
                    'EOB_DNL_CD']
    #Columns which have the same value for 97.5% of the rows 
    same_value_columns = same_values(dataset, 0.975)
    df.drop(columns = columns_to_remove, inplace = True)
    df.drop(columns = same_value_columns, inplace = True)
    # Convert to int (Manual identification)
    df['TOTL_UNITS_PRCD_CNT'] = df['TOTL_UNITS_PRCD_CNT'].astype('float64')
    # Create variables 
    df['yr'] = pd.DatetimeIndex(df['SRVC_THRU_DT']).year
    df['mnth'] = pd.DatetimeIndex(df['SRVC_THRU_DT']).month
    df['day_of_week'] = pd.DatetimeIndex(df['SRVC_THRU_DT']).dayofweek
    # Convert to string 
    df['yr'] = df['yr'].astype('str')
    df['mnth'] = df['mnth'].astype('str')
    df['day_of_week'] = df['day_of_week'].astype('str')
   # Drop date variables
    df.drop(columns = ['SRVC_THRU_DT','SRVC_FROM_DT'], inplace = True) 
    # String columns with less than 16 unique values (for OHE)
    unique_cols = ['PROV_TAX_ID','PROV_NM','PROV_STR_ADRS','ROV_ZIP_5_CD','PROV_PAYENT_LCTN_CD','MX_PRCG_VRTN_CD',
                    'SCRN_FRMT_CD','MIXER_PARG_IND','CLM_TYP','NUM_LINES','HCFA_PT_CD','CLM_TYPE_CD','TELEHEALTH',
                    'PROD_DESC','NEW_CLM_TYP','UM_RQRD_IND','CLM_PAYMNT_ACTN_1_CD','yr','mnth','day_of_week']
    columns_highly_correlated = ['PROV_NM', 'PROV_STR_ADRS', 'ROV_ZIP_5_CD', 'PROV_PAYENT_LCTN_CD', 'CLM_TYP', 
                             'UM_RQRD_IND', 'MX_PRCG_VRTN_CD',  'MIXER_PARG_IND', 'HCFA_PT_CD',  'TELEHEALTH']
    dataset.drop(columns = columns_highly_correlated, inplace = True)
    ohe


# #### Questions

# In[37]:


#. 1. Service through and from date are same for all values
dataset['duration_of_treatment'] = pd.to_datetime(dataset['SRVC_THRU_DT']) - pd.to_datetime(dataset['SRVC_FROM_DT'])
dataset['date_check'] = dataset['SRVC_THRU_DT']==dataset['SRVC_FROM_DT']


# In[ ]:


# 2. DTL_LINE_NBR - number 01 vs 1 - is there any difference?
# Note - I think this can be an integer

