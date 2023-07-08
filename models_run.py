#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import pandas as pd, numpy as np,  sys, os 
from sklearn import metrics
from scipy import sparse
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import csv, warnings, time, pickle

sys.path.insert(0,'/ds/data/ve2/dtl/aixx/phi/no_gbd/r000/work/pocep_ds/pocep_ds')
sys.path.insert(0,'/ds/data/ve2/dtl/aixx/phi/no_gbd/r000/work/pocep_ds/pocep_ds/code')
sys.path.insert(0,'/ds/data/ve2/dtl/aixx/phi/no_gbd/r000/work/pocep_ds/pocep_ds/code/models')

from config.PARAMETERS_global import *
from config.LOOKUP_objects import *
from utils.BASIC_input_output import *
from utils.MODEL_basics import *

warnings.filterwarnings('ignore')
pd.options.display.max_rows = 100


# In[2]:


get_ipython().run_line_magic('reload_ext', 'autoreload')


# In[3]:


model_name = 'model_aapm_basic_subset_ap_prof_claims_1MM_num_cap_y_dates_n_binning_n_fe_proc_cd_grp_nbr'


# In[4]:


# model_name = model_aapm_version ## This will be the name of the folder
path = path_data_output + '/encoders/ohe/' + model_name + '/'
path_train = path + 'train/'
path_test  = path + 'test/'
X_train = sparse.load_npz(path_train + 'X.npz')
y_train = pd.read_csv(path_train + 'Y.csv')[target_aa]
X_test = sparse.load_npz(path_test + 'X.npz')
y_test = pd.read_csv(path_test + 'Y.csv')[target_aa]
col_dict = pd.read_csv(path_train + 'col_dict.csv')


# In[5]:


print(X_test.shape, y_test.shape)
X_test_ = X_test[:len(y_test)] #remove last row
X_train.shape[0] + X_test_.shape[0]


# #### Linear Regression

# In[6]:


from sklearn.linear_model import LinearRegression


# In[7]:


get_ipython().run_cell_magic('time', '', 'reg = LinearRegression().fit(X_train, y_train)')


# In[8]:


get_ipython().run_cell_magic('time', '', 'test_pred = reg.predict(X_test_)\ntrain_pred = reg.predict(X_train)')


# In[9]:


get_ipython().run_cell_magic('time', '', 'calculate_performance(y_test, test_pred)\ncalculate_performance(y_train, train_pred) ')


# In[10]:


calculate_performance_for_hypo_testing(y_test, test_pred)
generate_deviation_stats_for_hypo_testing(test_pred, y_test)


# In[121]:


generate_deviation_stats(train_pred, y_train)


# In[122]:


generate_deviation_stats(test_pred, y_test)


# In[123]:


for i,col in enumerate(col_dict.VAR):
    print(col,'{0:.4f}'.format(reg.coef_[i]))


# In[124]:


calculate_performance_for_hypo_testing(y_test, test_pred)
generate_deviation_stats_for_hypo_testing(test_pred, y_test)


# In[87]:


calculate_performance_for_hypo_testing(y_test, test_pred)
generate_deviation_stats_for_hypo_testing(test_pred, y_test)


# In[40]:


calculate_performance_for_hypo_testing(y_test, test_pred)
generate_deviation_stats_for_hypo_testing(test_pred, y_test)


# #### GBM

# In[11]:


from sklearn.ensemble import GradientBoostingRegressor


# In[28]:


gbr = GradientBoostingRegressor(n_estimators = 50, max_depth=7, learning_rate = 0.1, max_features = 'sqrt', random_state=42)


# In[29]:


get_ipython().run_cell_magic('time', '', 'gbr.fit(X_train, y_train)')


# In[30]:


preds_train = gbr.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
r2_train = r2_score(y_train, preds_train)
preds_test = gbr.predict(X_test_)
rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
r2_test = r2_score(y_test, preds_test)


# In[31]:


calculate_performance(y_test, preds_test)
calculate_performance(y_train, preds_train) 


# In[32]:


generate_deviation_stats(preds_train, y_train)


# In[33]:


generate_deviation_stats(preds_test, y_test)


# In[35]:


for i,col in enumerate(col_dict.VAR):
    print(col,'{0:.4f}'.format(gbr.feature_importances_[i]))


# In[20]:


top_10_features(gbr)


# In[ ]:


top_10_features(gbr)


# In[34]:


calculate_performance_for_hypo_testing(y_test, preds_test)
generate_deviation_stats_for_hypo_testing(preds_test, y_test)


# In[16]:


calculate_performance_for_hypo_testing(y_test, preds_test)
generate_deviation_stats_for_hypo_testing(preds_test, y_test)


# In[133]:


calculate_performance_for_hypo_testing(y_test, preds_test)
generate_deviation_stats_for_hypo_testing(preds_test, y_test)


# In[71]:


calculate_performance_for_hypo_testing(y_test, preds_test)
generate_deviation_stats_for_hypo_testing(preds_test, y_test)


# In[51]:


calculate_performance_for_hypo_testing(y_test, preds_test)
generate_deviation_stats_for_hypo_testing(preds_test, y_test)


# #### GBM - 2 (Hyperparameters changed)

# In[11]:


from sklearn.ensemble import GradientBoostingRegressor


# In[12]:


gbr = GradientBoostingRegressor(n_estimators = 500, max_depth=7, learning_rate = 0.1, random_state=42)


# In[13]:


get_ipython().run_cell_magic('time', '', 'gbr.fit(X_train, y_train)')


# In[14]:


preds_train = gbr.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
r2_train = r2_score(y_train, preds_train)
preds_test = gbr.predict(X_test_)
rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
r2_test = r2_score(y_test, preds_test)


# In[15]:


calculate_performance(y_test, preds_test)
calculate_performance(y_train, preds_train) 


# In[16]:


generate_deviation_stats(preds_train, y_train)


# In[17]:


generate_deviation_stats(preds_test, y_test)


# In[18]:


for i,col in enumerate(col_dict.VAR):
    print(col,'{0:.4f}'.format(gbr.feature_importances_[i]))


# In[19]:


top_10_features(gbr)


# In[20]:


calculate_performance_for_hypo_testing(y_test, preds_test)
generate_deviation_stats_for_hypo_testing(preds_test, y_test)


# ### XG Boost

# In[6]:


import xgboost
from xgboost import plot_importance


# In[16]:


model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.09,
                 max_depth=7,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit(X_train, y_train)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'preds_train = model.predict(X_train)\npreds_test = model.predict(X_test_)')


# In[ ]:


calculate_performance_for_hypo_testing(y_test, preds_test)
generate_deviation_stats_for_hypo_testing(preds_test, y_test)


# In[ ]:





# In[27]:


model2 = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=6,
                 min_child_weight=1,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 


# In[28]:


get_ipython().run_cell_magic('time', '', 'model2.fit(X_train, y_train)')


# In[29]:


preds_train = model2.predict(X_train)
preds_test = model2.predict(X_test_)


# In[30]:


calculate_performance_for_hypo_testing(y_test, preds_test)
generate_deviation_stats_for_hypo_testing(preds_test, y_test)


# In[31]:


model3 = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=10,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 


# In[32]:


get_ipython().run_cell_magic('time', '', 'model3.fit(X_train, y_train)')


# In[33]:


preds_train = model3.predict(X_train)
preds_test = model3.predict(X_test_)


# In[34]:


calculate_performance_for_hypo_testing(y_test, preds_test)
generate_deviation_stats_for_hypo_testing(preds_test, y_test)


# In[43]:


pickle.dump(model3, open('xgboost_best_perf.pkl', 'wb'))


# In[ ]:





# In[37]:


model4 = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=1,                 
                 learning_rate=0.07,
                 max_depth=10,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 


# In[38]:


get_ipython().run_cell_magic('time', '', 'model4.fit(X_train, y_train)')


# In[39]:


preds_train = model4.predict(X_train)
preds_test = model4.predict(X_test_)


# In[40]:


calculate_performance_for_hypo_testing(y_test, preds_test)
generate_deviation_stats_for_hypo_testing(preds_test, y_test)


# In[41]:


calculate_performance_for_hypo_testing(y_train, preds_train)
generate_deviation_stats_for_hypo_testing(preds_train, y_train)


# In[44]:





# In[46]:


feats = model4.feature_importances_


# In[50]:


feats_df = pd.DataFrame(feats)


# In[53]:


feats_df.to_csv('feats.csv', index = False)


# In[52]:


feats_df['col'] = col_dict.VAR.tolist()


# In[45]:


for i,col in enumerate(col_dict.VAR):
    print(col,'{0:.4f}'.format(model4.feature_importances_[i]))


# In[54]:


feats_df['col']


# #### RF

# In[34]:


from sklearn.ensemble import RandomForestRegressor


# In[63]:


regr = RandomForestRegressor(n_estimators = 500, max_depth=5,max_features = 'auto', random_state=42)


# In[64]:


get_ipython().run_cell_magic('time', '', 'regr.fit(X_train, y_train)')


# In[65]:


preds_train = regr.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
r2_train = r2_score(y_train, preds_train)

preds_test = regr.predict(X_test_)
rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
r2_test = r2_score(y_test, preds_test)


# In[66]:


calculate_performance(y_test, preds_test)
calculate_performance(y_train, preds_train) 


# In[67]:


generate_deviation_stats(preds_train, y_train)


# In[68]:


generate_deviation_stats(preds_test, y_test)


# In[27]:


for i,col in enumerate(col_dict.VAR):
    print(col,'{0:.4f}'.format(regr.feature_importances_[i]))


# #### Ridge

# In[92]:


from sklearn.linear_model import Ridge


# In[99]:


get_ipython().run_cell_magic('time', '', 'reg = Ridge(alpha=1).fit(X_train, y_train)')


# In[100]:


get_ipython().run_cell_magic('time', '', 'test_pred = reg.predict(X_test_)\ntrain_pred = reg.predict(X_train)')


# In[101]:


get_ipython().run_cell_magic('time', '', 'calculate_performance(y_test, test_pred)\ncalculate_performance(y_train, train_pred) ')


# In[102]:


for i,col in enumerate(col_dict.VAR):
    print(col,'{0:.4f}'.format(reg.coef_[i]))


# In[103]:


generate_deviation_stats(train_pred, y_train)


# In[104]:


generate_deviation_stats(test_pred, y_test)


# In[ ]:





# #### Lasso

# In[105]:


from sklearn.linear_model import Lasso


# In[106]:


get_ipython().run_cell_magic('time', '', 'reg = Lasso(alpha = 1).fit(X_train, y_train)')


# In[107]:


get_ipython().run_cell_magic('time', '', 'test_pred = reg.predict(X_test_)\ntrain_pred = reg.predict(X_train)')


# In[108]:


get_ipython().run_cell_magic('time', '', 'calculate_performance(y_test, test_pred)\ncalculate_performance(y_train, train_pred) ')


# In[66]:


for i,col in enumerate(col_dict.VAR):
    print(col,'{0:.4f}'.format(reg.coef_[i]))


# In[109]:


generate_deviation_stats(train_pred, c)


# In[110]:


generate_deviation_stats(test_pred, y_test)


# In[12]:


round(y_train.describe(),2)


# In[ ]:




