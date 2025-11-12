#!/usr/bin/env python
# coding: utf-8

# # Bank Customer Churn Prediction
# 
# ![23000.jpg](attachment:23000.jpg)
# 
# 
# 
# 
# ## Introduction
# 
# **Customer Churn prediction** means knowing which customers are likely to leave or unsubscribe from your service. For many companies, this is an important prediction. This is because acquiring new customers often costs more than retaining existing ones. Once you’ve identified customers at risk of churn, you need to know exactly what marketing efforts you should make with each customer to maximize their likelihood of staying.
# 
# Customers have different behaviors and preferences, and reasons for cancelling their subscriptions. Therefore, it is important to actively communicate with each of them to keep them on your customer list. You need to know which marketing activities are most effective for individual customers and when they are most effective.
# 
# ## Impact of customer churn on businesses
# 
# A firm with a high churn rate loses a lot of members, which results in lower growth rates and a bigger impact on sales and earnings. Customers may be retained by businesses with low turnover rates.
# 
# ## Why is Analyzing Customer Churn Prediction Important?
# 
# Because it costs more to attract new customers than it does to sell to existing ones, customer turnover is crucial. This measure determines whether a firm succeeds or fails. Effective customer retention raises the average lifetime value of the customer, increasing the value of all subsequent sales and boosting unit profits.
# 
# Increasing income through recurring subscriptions and dependable repeat business is frequently a better use of a company's resources than spending money on attracting new clients. It's lot simpler to expand and weather financial difficulty if you can keep your existing clients than it is to spend money bringing in new ones to replace the ones who have departed.
# 
# 

# ![174948746-5dc3418a-8296-4cc8-9561-f8f12ca9a0a4.png](attachment:174948746-5dc3418a-8296-4cc8-9561-f8f12ca9a0a4.png)
# 
# ## Problem Statement :
# 
# Customer churn or customer attrition is a tendency of clients or customers to abandon a brand and stop being a paying client of a particular business or organization. The percentage of customers that discontinue using a company’s services or products during a specific period is called a customer churn rate. Several bad experiences (or just one) are enough, and a customer may quit. And if a large chunk of unsatisfied customers churn at a time interval, both material losses and damage to reputation would be enormous.
# 
# A reputed bank “ABC BANK” wants to predict the Churn rate. Create a model by using different machine learning approaches that can predict the best result. 
# 
# ## Dataset Description :
# 
# This is a public dataset, The dataset format is given below.
#  
# Inside the dataset, there are 10000 rows and 14 different columns.
# 
# The target column here is **Exited** here.
# 
# The details about all the columns are given in the following data dictionary -
# 
# | Variable | Definition |
# | ------------- | ------------- |
# | RowNumber | Unique Row Number |
# | CustomerId | Unique Customer Id |
# | Surname | Surname of a customer |
# | CreditScore | Credit Score of each Customer  |
# | Geography | Geographical Location of Customers |
# | City_Category | Category of the City (A,B,C) |
# | Gender | Sex of Customers |
# | Age | Age of Each Customer |
# | Tenure | Number of years |
# | Balance | Current Balance of Customers |
# | NumOfProducts | Number of Products |
# | HasCrCard | If a customer has a credit card or not |
# | IsActiveMember | If a customer is active or not |
# | EstimatedSalary | Estimated Salary of each Customer |
# | **Exited** | **Customer left the bank or Not (Target Variable)** |
# 
# 
# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
get_ipython().run_line_magic('matplotlib', 'inline')
import time
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import keras
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dropout, Dense, LeakyReLU
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import initializers

plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')


# ### Importing the Dataset

# In[2]:


data = pd.read_csv("Churn_Modelling.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# ### Summary of Data
# 

# In[5]:


data.info()


# In[6]:


data.describe().T


# ### Total Unique value

# In[7]:


data.nunique()


# ### Total Missing values

# In[8]:


data.isnull().sum()


# * We can see there is **no missing values** in any column.
# 
# 
# 
# ### Deleting Unnecessary Information
# 
# The columns **RowNumber**, **CustomerId** and **Surname** are related to personal data of the customers. 
# These columns do not have any quantitative impact on any calculations whatsoever. 
# 
# Hence, we can avoid these extra columns of information by removing them from the data.

# In[9]:


# deleting the unnecessary columns (RowNumber, CustomerId, Surname)

data.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1, inplace = True)

data.head()


# ## Exploratory Data Analysis
# ### Data Visualization

# In[10]:


data.hist(bins = 100, figsize = (25, 25))
plt.suptitle('Histograms of Numerical Columns', fontsize = 35)
plt.show()


# In[11]:


labels = 'Exited', 'Retained'
sizes = [data.Exited[data['Exited'] == 1].count(), data.Exited[data['Exited'] == 0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize = (10, 8))
ax1.pie(sizes, explode = explode, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size = 20)
plt.show()


# So about **20% of the customers have churned.** So the baseline model could be to predict that 20% of the customers will churn. 
# * Given 20% is a small number, we need to ensure that the chosen model does predict with great accuracy this 20% as it is of interest to the bank to identify and keep this bunch as opposed to accurately predicting the customers that are retained.
# 
# * Here, we can see that our data is **Mild Imbalanced** we need to tackle this

# In[12]:


labels = 'Female', 'Male'
sizes = [data.Exited[data['Gender'] == "Female"].count(), data.Exited[data['Gender'] == "Male"].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize = (5,6))
ax1.pie(sizes, explode = explode, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90)
ax1.axis('equal')
plt.title("Proportion of Customer", size = 20)
plt.show()


# * **Gender:** We can clearly see the **Male customers** had more number than the female customers.

# In[13]:


fig, ax = plt.subplots(2, 3, figsize = (25, 15))

sns.countplot(x = 'Geography',      hue = 'Exited', data = data, palette = 'Set2', ax = ax[0][0])
sns.countplot(x = 'Gender',         hue = 'Exited', data = data, palette = 'Set2', ax = ax[0][1])
sns.countplot(x = 'HasCrCard',      hue = 'Exited', data = data, palette = 'Set2', ax = ax[0][2])
sns.countplot(x = 'IsActiveMember', hue = 'Exited', data = data, palette = 'Set2', ax = ax[1][0])
sns.countplot(x = 'NumOfProducts',  hue = 'Exited', data = data, palette = 'Set2', ax = ax[1][1])
sns.countplot(x = 'Tenure',         hue = 'Exited', data = data, palette = 'Set2', ax = ax[1][2])

plt.show()


# * **Geography:** We can see that majority of the data is about people **France**. Ideally for an evenly-distributed data, if the amount of people from a place is the majority, then the majority of churning should also be within that group. However, it is not so in this case as we see that number of exited people who belong to **Germany is almost equal to the number of exits from France.**
# 
# 
# * **Gender:** We can clearly see the **Female customers had more exits than the male customers.**
# 
# 
# * **Credit cards:** It is generally expected that people who have more interactions and products of the bank, would likely be retained for a longer time. However, we can see that people who have credit cards have more exits than those who do not own credit cards.
# 
# 
# * **Active Member:** This is an expected observation. We can see that **inactive members** have been churned more than members who are active.
# 
# 
# * **Number of Products:** This is also an expected observation, where we see that customers who own **more products from the bank are likely to be retained** for a longer time than those who own less products.
# 
# 
# * **Tenure:** We see that the tenure of a customer does not really tell us much if that customer is likely to be churned or not. Initially, it looks like new joinees and older people (10 years) have been churned less. However, on a closer analysis we can see that the overall number of retained customer are significantly less in both these cases. As a result, we can probably conclude that new joinees and older customers may be more likely to be churned as their churn rate (percentage) is likely to be higher than other tenure rates.

# In[14]:


fig, ax = plt.subplots(2, 3, figsize = (25, 15))

sns.boxplot(data = data, x = 'Exited', y = 'CreditScore',     hue = 'Exited', ax = ax[0][0])
sns.boxplot(data = data, x = 'Exited', y = 'Age',             hue = 'Exited', ax = ax[0][1])
sns.boxplot(data = data, x = 'Exited', y = 'Balance',         hue = 'Exited', ax = ax[0][2])
sns.boxplot(data = data, x = 'Exited', y = 'EstimatedSalary', hue = 'Exited', ax = ax[1][0])
sns.boxplot(data = data, x = 'Exited', y = 'NumOfProducts',   hue = 'Exited', ax = ax[1][1])
sns.boxplot(data = data, x = 'Exited', y = 'Tenure',          hue = 'Exited', ax = ax[1][2])

plt.show()


# * **Credit Score:** We can see that Credit Score **does not have much effect** on the customer churn.
# 
# 
# * **Age:** Here we can see that the **older customers are more likely to be churned** from the bank. This is most probably to keep a younger manpower in the organization.
# 
# 
# * **Balance:** When it comes to Balance, we see that the bank is losing a significant number of customers with **high balance** in their accounts. This is likely to affect the bank's capital as well.
# 
# 
# * **Estimated Salary:** Estimated Salary **does not seem to affect** the customer churn much.
# 
# 
# * **Number of Products:** We see that the number of products also **does not seem to affect** the customer churn.
# 
# 
# * **Tenure:** For tenure, as we can see here too, customer belonging more to the two extreme tenure groups **(new joinees and older ones)** are more likely to be churned.

# ## Data Preprocessing / Data Preparation

# In[15]:


continuous_vars  = ['CreditScore',  'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
categorical_vars = ['HasCrCard', 'IsActiveMember', 'Geography', 'Gender']


# In[16]:


data_train = data.sample(frac = 0.8, random_state = 100)
data_test  = data.drop(data_train.index)


# In[17]:


print('Number of rows in train data: ', len(data_train))
print('Number of rows in test data : ', len(data_test))


# In[18]:


data_train = data_train[['Exited'] + continuous_vars + categorical_vars]
data_train.head()


# * Here, we change the **0 values** of the variable columns **HasCrCard and IsActiveMember into -1**. 
# 
# **This will allow us to include a negative relation in the modeling.**

# In[19]:


# turning 0 values of numerical categorical features into -1 to introduce negative relation in the calculations

data_train.loc[data_train.HasCrCard == 0, 'HasCrCard'] = -1
data_train.loc[data_train.IsActiveMember == 0, 'IsActiveMember'] = -1

data_train.head()


# * Now we'll doing one-hot encode the remaining text categorical variables **Geography and Gender**.

# In[20]:


# list of categorical variables

var_list = ['Geography', 'Gender']

# turning the categorical variables into one-hot vectors

for var in var_list:
    for val in data_train[var].unique():
        data_train[var + '_' + val] = np.where(data_train[var] == val, 1, -1)

data_train = data_train.drop(var_list, axis = 1)

data_train.head()


# In[21]:


sns.pairplot(data_train, hue = "Exited", corner = True)


# In[ ]:





# ## Scaling

# In[22]:


min_values = data_train[continuous_vars].min()
max_values = data_train[continuous_vars].max()

data_train[continuous_vars] = (data_train[continuous_vars] - min_values) / (max_values - min_values)
data_train.head()


# In[23]:


data_train.head()


# In[24]:


data_test.head()


# In[25]:


print(data_train.shape)
print(data_test.shape)


# ## Machine Learning Models
# 
# ## Modeling
# 
# ### 1. Logistic Regression classifier

# In[26]:


def best_model(model):
    print(model.best_score_)
    print(model.best_params_)
    print(model.best_estimator_)


# In[27]:


start_time = time.time()

parameters = {'C': [0.1, 0.5, 1, 5, 10, 50, 100],
              'max_iter': [50, 100, 200, 300], 
              'fit_intercept':[True],
              'intercept_scaling':[1],
              'penalty':['l2'],
              'tol':[0.00001, 0.0001, 0.000001]}

LR_grid_model = GridSearchCV(LogisticRegression(), 
                             param_grid = parameters, 
                             cv = 10, 
                             refit = True, 
                             verbose = 0)

LR_grid_model.fit(data_train.loc[:, data_train.columns != 'Exited'], data_train.Exited)

print('Time taken: %.1f seconds.\n' % (time.time() - start_time))

best_model(LR_grid_model)


# In[28]:


lr_model = LogisticRegression(C = 0.1, class_weight = None, dual = False, fit_intercept = True, intercept_scaling = 1, 
                              l1_ratio = None, max_iter = 50, multi_class = 'auto', n_jobs = None, penalty = 'l2', 
                              random_state = None, solver = 'lbfgs', tol = 1e-05, verbose = 0, warm_start = False)

lr_model.fit(data_train.loc[:, data_train.columns != 'Exited'], data_train.Exited)


# In[29]:


print(classification_report(data_train.Exited, lr_model.predict(data_train.loc[:, data_train.columns != 'Exited'])))


# In[30]:


print("Testing accuracy  :", accuracy_score(lr_model.predict(data_train.loc[:, data_train.columns != 'Exited']), 
                                            data_train.Exited))


# ### 2. Random Forest Classifier

# In[31]:


start_time = time.time()

parameters = {'max_depth': [6, 7, 8, 9, 10], 
              'max_features': [5, 6, 7, 8, 9],
              'n_estimators':[10, 50, 100],
              'min_samples_split': [3, 5, 6, 7]}

RF_grid_model = GridSearchCV(RandomForestClassifier(),
                             parameters,
                             cv = 10,
                             refit = True,
                             verbose = 0)

RF_grid_model.fit(data_train.loc[:, data_train.columns != 'Exited'], data_train.Exited)

print('Time taken: %.1f seconds.\n' % (time.time() - start_time))

best_model(RF_grid_model)


# In[32]:


rf_model = RandomForestClassifier(bootstrap = True, ccp_alpha = 0.0, class_weight = None, criterion = 'gini', max_depth = 9, 
                                  max_features = 9, max_leaf_nodes = None, max_samples = None, min_impurity_decrease = 0.0, 
                                  min_impurity_split = None, min_samples_leaf = 1, min_samples_split = 7, 
                                  min_weight_fraction_leaf = 0.0, n_estimators = 50, n_jobs = None, oob_score = False, 
                                  random_state = None, verbose = 0, warm_start = False)

rf_model.fit(data_train.loc[:, data_train.columns != 'Exited'], data_train.Exited)


# In[33]:


print(classification_report(data_train.Exited, rf_model.predict(data_train.loc[:, data_train.columns != 'Exited'])))


# In[34]:


print("Testing accuracy  :", accuracy_score(rf_model.predict(data_train.loc[:, data_train.columns != 'Exited']), 
                                            data_train.Exited))


# ### 3. Support Vector Machines
#  #### 3.1 Support Vector Machines with RBF kernel

# In[35]:


start_time = time.time()

parameters = {'C': [1, 10, 50, 100],
              'gamma': [0.1, 0.01, 0.001],
              'probability': [True],
              'kernel': ['rbf']}

SVM_rbf_grid_model = GridSearchCV(SVC(), 
                                  parameters, 
                                  cv = 5, 
                                  refit = True, 
                                  verbose = 0)

SVM_rbf_grid_model.fit(data_train.loc[:, data_train.columns != 'Exited'], data_train.Exited)

print('[INFO] Time taken: %.1f seconds.\n' % (time.time() - start_time))

best_model(SVM_rbf_grid_model)


# In[36]:


svm_rbf_model = SVC(C=100, break_ties = False, cache_size = 200, class_weight = None, coef0 = 0.0, 
                    decision_function_shape = 'ovr', degree = 3, gamma = 0.1, kernel = 'rbf', max_iter = -1, probability = True,
                     random_state = None, shrinking = True, tol = 0.001, verbose = False)

svm_rbf_model.fit(data_train.loc[:, data_train.columns != 'Exited'], data_train.Exited)


# In[37]:


print(classification_report(data_train.Exited, svm_rbf_model.predict(data_train.loc[:, data_train.columns != 'Exited'])))


# In[38]:


print("Testing accuracy  :", accuracy_score(svm_rbf_model.predict(data_train.loc[:, data_train.columns != 'Exited']), 
                                            data_train.Exited))


# #### 3.2 Support Vector Machines with Poly kernel

# In[39]:


start_time = time.time()

parameters = {'C': [1, 10, 50, 100],
              'gamma': [0.1, 0.01, 0.001],
              'probability': [True],
              'kernel': ['poly'],
              'degree': [2, 3]}

SVM_poly_grid_model = GridSearchCV(SVC(), 
                                   parameters, 
                                   cv = 5, 
                                   refit = True, 
                                   verbose = 0)

SVM_poly_grid_model.fit(data_train.loc[:, data_train.columns != 'Exited'], data_train.Exited)

print('[INFO] Time taken: %.1f seconds.\n' % (time.time() - start_time))

best_model(SVM_poly_grid_model)


# In[40]:


svm_poly_model = SVC(C = 100, break_ties = False, cache_size = 200, class_weight = None, coef0 = 0.0, 
                     decision_function_shape = 'ovr', degree = 2, gamma = 0.1, kernel = 'poly', max_iter = -1, 
                     probability = True, random_state = None, shrinking = True, tol = 0.001, verbose = False)

svm_poly_model.fit(data_train.loc[:, data_train.columns != 'Exited'], data_train.Exited)


# In[41]:


print(classification_report(data_train.Exited, svm_poly_model.predict(data_train.loc[:, data_train.columns != 'Exited'])))


# In[42]:


print("Testing accuracy  :", accuracy_score(svm_poly_model.predict(data_train.loc[:, data_train.columns != 'Exited']), 
                                            data_train.Exited))


# ### 4. Stochastic Gradient Descent (SGD) classifier

# In[43]:


start_time = time.time()

parameters = {'loss': ['hinge', 'log'],
              'max_iter': [50, 100, 200, 300], 
              'fit_intercept':[True],
              'penalty':['l2'],
              'tol':[0.00001, 0.0001, 0.000001]}

SGD_grid_model = GridSearchCV(SGDClassifier(), 
                              param_grid = parameters, 
                              cv = 10, 
                              refit = True, 
                              verbose = 0)

SGD_grid_model.fit(data_train.loc[:, data_train.columns != 'Exited'], data_train.Exited)

print('[INFO] Time taken: %.1f seconds.\n' % (time.time() - start_time))

best_model(SGD_grid_model)


# In[44]:


sgd_model = SGDClassifier(alpha = 0.0001, average = False, class_weight = None, early_stopping = False, epsilon = 0.1, 
                          eta0 = 0.0, fit_intercept = True, l1_ratio = 0.15, learning_rate = 'optimal', loss = 'log', 
                          max_iter = 300, n_iter_no_change = 5, n_jobs = None, penalty = 'l2', power_t = 0.5, 
                          random_state = None, shuffle = True, tol = 1e-06, validation_fraction = 0.1, verbose = 0, 
                          warm_start = False)

sgd_model.fit(data_train.loc[:, data_train.columns != 'Exited'], data_train.Exited)


# In[45]:


print(classification_report(data_train.Exited, sgd_model.predict(data_train.loc[:, data_train.columns != 'Exited'])))


# In[46]:


print("Testing accuracy  :", accuracy_score(sgd_model.predict(data_train.loc[:, data_train.columns != 'Exited']), 
                                            data_train.Exited))


# ### 4. Extreme Gradient Boost (XGB) classifier

# In[47]:


start_time = time.time()

parameters = {'max_depth': [5, 6, 7, 8],
              'gamma': [0.01, 0.001, 0.001],
              'min_child_weight': [1, 5, 10],
              'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
              'n_estimators': [5, 10, 20, 100]}

XGB_grid_model = GridSearchCV(XGBClassifier(), 
                              parameters, 
                              cv = 10, 
                              refit = True, 
                              verbose = 0)

XGB_grid_model.fit(data_train.loc[:, data_train.columns != 'Exited'], data_train.Exited)

print('[INFO] Time taken: %.1f seconds.\n' % (time.time() - start_time))

best_model(XGB_grid_model)


# In[48]:


xgb_model = XGBClassifier(base_score = 0.5, booster = 'gbtree', colsample_bylevel = 1, colsample_bynode = 1, 
                          colsample_bytree = 1, gamma = 0.001, learning_rate = 0.1, max_delta_step = 0, max_depth = 5, 
                          min_child_weight = 1, missing = 1, n_estimators = 100, n_jobs = 1, nthread = None, 
                          objective = 'binary:logistic', random_state = 0, reg_alpha = 0, reg_lambda = 1, scale_pos_weight = 1, 
                          seed = None, silent = None, subsample = 1, verbosity = 1)

xgb_model.fit(data_train.loc[:, data_train.columns != 'Exited'], data_train.Exited)


# In[49]:


print(classification_report(data_train.Exited, xgb_model.predict(data_train.loc[:, data_train.columns != 'Exited'])))


# In[50]:


print("Testing accuracy  :", accuracy_score(xgb_model.predict(data_train.loc[:, data_train.columns != 'Exited']), 
                                            data_train.Exited))


# ### Receiver Operating Characteristic (ROC)

# In[51]:


def get_roc(y, predict_vals, prob_values):
    roc_score = roc_auc_score(y, predict_vals)
    false_positives, true_positives, _ = roc_curve(y, prob_values)
    return (roc_score, false_positives, true_positives)


# In[52]:


y = data_train.Exited
X = data_train.loc[:, data_train.columns != 'Exited']


roc_lr,       false_lr,       true_lr = get_roc(y, lr_model.predict(X), lr_model.predict_proba(X)[:, 1])
roc_rf,       false_rf,       true_rf = get_roc(y, rf_model.predict(X), rf_model.predict_proba(X)[:, 1])
roc_svm_rbf,  false_svm_rbf,  true_svm_rbf = get_roc(y, svm_rbf_model.predict(X), svm_rbf_model.predict_proba(X)[:, 1])
roc_svm_poly, false_svm_poly, true_svm_poly = get_roc(y, svm_poly_model.predict(X), svm_poly_model.predict_proba(X)[:, 1])
roc_sgd,      false_sgd,      true_sgd = get_roc(y, sgd_model.predict(X), sgd_model.predict_proba(X)[:, 1])
roc_xgb,      false_xgb,      true_xgb = get_roc(y, xgb_model.predict(X), xgb_model.predict_proba(X)[:, 1])


# In[53]:


plt.figure(figsize = (15, 8), linewidth = 2)


plt.plot(false_lr,       true_lr,       label = 'LR: ' + str(round(roc_lr, 4)))
plt.plot(false_rf,       true_rf,       label = 'RF: ' + str(round(roc_rf, 4)))
plt.plot(false_svm_rbf,  true_svm_rbf,  label = 'SVM (RBF): ' + str(round(roc_svm_rbf, 4)))
plt.plot(false_svm_poly, true_svm_poly, label = 'SVM (Poly): ' + str(round(roc_svm_poly, 4)))
plt.plot(false_sgd,      true_sgd,      label = 'SGD: ' + str(round(roc_sgd, 4)))
plt.plot(false_xgb,      true_xgb,      label = 'XGB: ' + str(round(roc_xgb, 4)))

plt.plot([0, 1], [0, 1], 'k--', label = 'Random: 0.5')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc = 'best')

plt.show()


# * From the above graph, we can clearly see that the **Random Forest (RF) classifier has the highest ROC score (0.7705)** and hence covers the **highest area under curve as well.** 
# 
# * From this, we finally choose the **Random Forest classifier** as our final machine learning classifier model. 
# 
# Let us now try to use this model with our test data and see how it works out.

# In[54]:


data_test = data_test[['Exited'] + continuous_vars + categorical_vars]

# Change the 0 in categorical variables to -1

data_test.loc[data_test.HasCrCard == 0, 'HasCrCard'] = -1
data_test.loc[data_test.IsActiveMember == 0, 'IsActiveMember'] = -1

# One hot encode the categorical variables

var_list = ['Geography', 'Gender']

for var in var_list:
    for val in data_test[var].unique():
        data_test[var + '_' + val] = np.where(data_test[var] == val, 1, -1)

data_test = data_test.drop(var_list, axis = 1)

# Ensure that all one hot encoded variables that appear in the train data appear in the subsequent data

columns_list = list(set(data_train.columns) - set(data_test.columns))

for column in columns_list:
    data_test[str(column)] = -1

# MinMax scaling of the continuous variables based on min and max from the train data

data_test[continuous_vars] = (data_test[continuous_vars] - min_values) / (max_values - min_values)

# Ensure that The variables are ordered in the same way as was ordered in the train set

data_test = data_test[data_train.columns]


# In[55]:


# mask infinite values and delete not available or missing values from columns

data_test = data_test.mask(np.isinf(data_test))
data_test = data_test.dropna()

print(data_test.shape)


# In[56]:


print(classification_report(data_test.Exited,  rf_model.predict(data_test.loc[:, data_test.columns != 'Exited'])))


# In[57]:


print("Testing accuracy  :", accuracy_score(rf_model.predict(data_train.loc[:, data_train.columns != 'Exited']), 
                                            data_train.Exited))


# In[58]:


roc_rf_test, false_rf_test, true_rf_test = get_roc(data_test.Exited, 
                                                   rf_model.predict(data_test.loc[:, data_test.columns != 'Exited']), 
                                                   rf_model.predict_proba(data_test.loc[:, data_test.columns != 'Exited'])[:,1])


# In[59]:


plt.figure(figsize = (15,8), linewidth = 2)

plt.plot(false_rf_test, true_rf_test,label = 'RF: ' + str(round(roc_rf_test, 4)))

plt.plot([0, 1], [0, 1], 'k--', label = 'Random: 0.5')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc = 'best')

plt.show()


# * The precision of the model on previousy unseen test data is slightly higher with regard to predicting 1's i.e. those customers that churn. However, in as much as the model has a high accuracy, it still misses about half of those who end up churning. 
# 
# * This could be imprved by providing retraining the model with more data over time while in the meantime working with the model to save the 61% that would have churned :-)

# ## Deep Learning Models

# In[60]:


data.head()


# ## PreProcessing Data

# In[61]:


X = data.drop(['Exited'], axis = 1)
y = data['Exited']


# In[62]:


X.head()


# In[63]:


y.head()


# ## One-hot Encoding

# In[64]:


labelEncoder = LabelEncoder()

X['Geography'] = labelEncoder.fit_transform(X['Geography'])
X['Gender'] = labelEncoder.fit_transform(X['Gender'])

X.head()


# ## Splitting the dataset
# 

# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)


# ## Scaling

# In[66]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('X train shape:\t', X_train.shape)
print('y train shape:\t', y_train.shape)
print('X test shape:\t' , X_test.shape)
print('y test shape:\t' , y_test.shape)


# ### 1. Neural Network classifier

# In[67]:


nn_model = Sequential()

nn_model.add(Dense(500, activation = 'relu', input_dim = X.shape[1]))
nn_model.add(Dense(1, activation = 'sigmoid'))


# In[68]:


nn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[69]:


# fitting the neural model to the training data 

history = nn_model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 100, verbose = 1)


# In[70]:


# evaluating the model's performance on the train data and test data

_, train_acc = nn_model.evaluate(X_train, y_train, verbose = 1)
_, test_acc = nn_model.evaluate(X_test, y_test, verbose = 1)

print()

# print the train accuracy and test accuracy

print('Train accuracy: %.3f %%' % (train_acc * 100))
print('Test accuracy:\t%.3f %%' % (test_acc * 100))


# In[71]:


# predict the test values to get the confusion matrix 

y_pred = nn_model.predict_classes(X_test)

print(confusion_matrix(y_test, y_pred))


# In[72]:


print(classification_report(y_test, y_pred))


# In[73]:


# plot the accuracy and loss graphs for train and test data

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 10))
ax1.plot(history.history['loss'],     label = 'train loss')
ax1.plot(history.history['val_loss'], label = 'test loss')

ax2.plot(history.history['accuracy'],     label = 'train accuracy')
ax2.plot(history.history['val_accuracy'], label = 'test accuracy')

ax1.legend()
ax2.legend()
plt.show()


# ### 2. Neural Network classifier with 1 Hidden layer - with Early Stopping

# In[74]:


nn_model_es = Sequential()

nn_model_es.add(Dense(500, activation = 'relu', input_dim = X.shape[1]))
nn_model_es.add(Dense(1, activation = 'sigmoid'))


# In[75]:


nn_model_es.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[76]:


# setting up the Early Stopping criterion on validation loss

earlyStopping = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1)


# In[77]:


# fitting the neural model to the training data 

history = nn_model_es.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 500, verbose = 1, 
                          callbacks = [earlyStopping])


# In[78]:


# evaluating the model's performance on the train data and test data

_, train_acc = nn_model.evaluate(X_train, y_train, verbose = 1)
_, test_acc = nn_model.evaluate(X_test, y_test, verbose = 1)

print()

# print the train accuracy and test accuracy

print('Train accuracy: %.3f %%' % (train_acc * 100))
print('Test accuracy:\t%.3f %%' % (test_acc * 100))


# In[79]:


# predict the test values to get the confusion matrix 

y_pred = nn_model.predict_classes(X_test)

print(confusion_matrix(y_test, y_pred))


# In[80]:


print(classification_report(y_test, y_pred))


# In[81]:


# plot the accuracy and loss graphs for train and test data

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 10))
ax1.plot(history.history['loss'],     label = 'train loss')
ax1.plot(history.history['val_loss'], label = 'test loss')

ax2.plot(history.history['accuracy'],     label = 'train accuracy')
ax2.plot(history.history['val_accuracy'], label = 'test accuracy')

ax1.legend()
ax2.legend()
plt.show()


# ### 3. Neural Network Architecture with multiple Hidden layers

# In[82]:


X = data.iloc[:, 3:13].values
y = data['Exited'].values   

# removing unnecessary columns

#dataset = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)
dataset = data
dataset


# In[83]:


x = dataset.apply(LabelEncoder().fit_transform)

# one-hot encoding

OneHotEncode = OneHotEncoder(handle_unknown = 'ignore')
enc_df = pd.DataFrame(OneHotEncode.fit_transform(x[['Geography']]).toarray())

x = x.join(enc_df)
x = x.drop(['Geography', 'Exited'], axis = 1)
x = x.rename(columns = {0: 'France', 1:'Spain', 2:'Germany'})
z = x
x.head()


# In[84]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[85]:


X_train = StandardScaler().fit_transform(X_train)
X_test  = StandardScaler().fit_transform(X_test)


# In[86]:


def create_model():

    initializer = 'glorot_normal'

    # create model
    model = Sequential()
    model.add(Dense( input_dim = 12, units = 21, kernel_initializer = initializer))
    model.add(LeakyReLU(alpha = 0.25))
    model.add(Dropout(rate = 0.2))

    # Adding the 2nd layer
    model.add(Dense( units = 21, kernel_initializer = initializer))
    model.add(LeakyReLU(alpha = 0.25))
    model.add(Dropout(rate = 0.2))
    
    # Adding the 3rd layer
    model.add(Dense( units = 21, kernel_initializer = initializer))
    model.add(LeakyReLU(alpha = 0.25))
    model.add(Dropout(rate = 0.2))
    
    # Adding the fourth hidden layer
    model.add(Dense( units = 21, kernel_initializer = initializer))
    model.add(LeakyReLU(alpha = 0.25))
    model.add(Dropout(rate = 0.2))
    
    # Adding the fifth hidden layer
    model.add(Dense( units = 21, kernel_initializer = initializer))
    model.add(LeakyReLU(alpha = 0.25))
    
    # Adding the output layer
    model.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform')) 
    
    # Compile model
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

# fitting the model

model = create_model()
history = model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 16, epochs = 100, verbose = 1)


# In[87]:


# show network structure
keras.utils.plot_model(model, "network.png", show_shapes = True)


# In[88]:


# evaluating the model's performance on the train data and test data

_, train_acc = model.evaluate(X_train, y_train, verbose = 1)
_, test_acc = model.evaluate(X_test, y_test, verbose = 1)

print()

# print the train accuracy and test accuracy

print('Train accuracy: %.3f %%' % (train_acc * 100))
print('Test accuracy:\t%.3f %%' % (test_acc * 100))


# In[89]:


# predict the test values to get the confusion matrix 

y_pred = model.predict_classes(X_test)

print(confusion_matrix(y_test, y_pred))


# In[90]:


print(classification_report(y_test, y_pred))


# In[91]:


# plot the accuracy and loss graphs for train and test data

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 10))
ax1.plot(history.history['loss'],     label = 'train loss')
ax1.plot(history.history['val_loss'], label = 'test loss')

ax2.plot(history.history['accuracy'],     label = 'train accuracy')
ax2.plot(history.history['val_accuracy'], label = 'test accuracy')

ax1.legend()
ax2.legend()
plt.show()


# ### 4. Neural Network Architecture with Early Stopping

# In[92]:


earlyStopping = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1)


# In[93]:


# fitting the model

model = create_model()
history = model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 16, epochs = 100, verbose = 1, 
                    callbacks = [earlyStopping])


# In[94]:


# evaluating the model's performance on the train data and test data

_, train_acc = model.evaluate(X_train, y_train, verbose = 1)
_, test_acc = model.evaluate(X_test, y_test, verbose = 1)

print()

# print the train accuracy and test accuracy

print('Train accuracy: %.3f %%' % (train_acc * 100))
print('Test accuracy:\t%.3f %%' % (test_acc * 100))


# In[95]:


# predict the test values to get the confusion matrix 

y_pred = model.predict_classes(X_test)

print(confusion_matrix(y_test, y_pred))


# In[96]:


print(classification_report(y_test, y_pred))


# In[97]:


# plot the accuracy and loss graphs for train and test data

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 10))
ax1.plot(history.history['loss'],     label = 'train loss')
ax1.plot(history.history['val_loss'], label = 'test loss')

ax2.plot(history.history['accuracy'],     label = 'train accuracy')
ax2.plot(history.history['val_accuracy'], label = 'test accuracy')

ax1.legend()
ax2.legend()
plt.show()


# ## Conclusion
# 
# Churn prediction is important to predict the customer churn rate so that the management is able to develop customer retention strategies to retain the loyal customers. For the customers churn analytics, it is important to get a low false positive rates. This is because if the false positive rate is high, it means that the system predicts that the customer is churn but indeed the customer is not churn. As a result, Bank may face severe losses due to the wrong prediction given by the system because the promotion will be given to the customers who are actually not churning. Thus, precision is the most important evaluation element as high precision relates to the low false positive rate.
# 
# ### Accuracy
# 
# Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations. The highest accuracy score in the models is **Random Forest Classifier model**, the **accuracy 89.53%** accurate.
# 
# <table style='border:1px solid black'>
#   <tr style='font-size: 14px;'>
#       <th style='text-align: center;border: 1px solid black;'>Models</th>
#       <th style='text-align: center;border: 1px solid black;'>Training Accuracy Score</th>
#       <th style='text-align: center;border: 1px solid black;'>Testing Accuracy Score</th>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Logistic Regression</td>
#       <td style='text-align: left;border: 1px solid black;'>88.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>81.41 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Random Forest</td>
#       <td style='text-align: left;border: 1px solid black;'>94.42 %</td>
#       <td style='text-align: left;border: 1px solid black;'>89.53 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Support Vector Machine (SVM) (RBF kernel)</td>
#       <td style='text-align: left;border: 1px solid black;'>90.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>86.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Support Vector Machine (SVM) (Poly kernel)</td>
#       <td style='text-align: left;border: 1px solid black;'>91.53 %</td>
#       <td style='text-align: left;border: 1px solid black;'>85.32 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Stochastic Gradient Descent (SGD)</td> 
#       <td style='text-align: left;border: 1px solid black;'>90.47 %</td>
#       <td style='text-align: left;border: 1px solid black;'>81.78 %</td>
#   </tr>
#     <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>XGBoost Classifier</td> 
#       <td style='text-align: left;border: 1px solid black;'>96.45 %</td>
#       <td style='text-align: left;border: 1px solid black;'>88.81 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network classifier</td> 
#       <td style='text-align: left;border: 1px solid black;'>89.67 %</td>
#       <td style='text-align: left;border: 1px solid black;'>84.35 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network classifier with Early Stopping</td> 
#       <td style='text-align: left;border: 1px solid black;'>89.67 %</td>
#       <td style='text-align: left;border: 1px solid black;'>84.35 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network Architecture with multiple Hidden layers</td> 
#       <td style='text-align: left;border: 1px solid black;'>86.67 %</td>
#       <td style='text-align: left;border: 1px solid black;'>86.20 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network Architecture with Early Stopping</td> 
#       <td style='text-align: left;border: 1px solid black;'>86.65 %</td>
#       <td style='text-align: left;border: 1px solid black;'>86.23 %</td>
#   </tr>
# </table>
# 
# 
# ### Precision
# 
# Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. The highest precision score in the models is **Random Forest Classifier model**, the **precision approximate to the 90.00% which is pretty good. High precision relates to the low false positive rate.
# 
# <table style='border:1px solid black'>
#   <tr style='font-size: 14px;'>
#       <th style='text-align: center;border: 1px solid black;'>Models</th>
#       <th style='text-align: center;border: 1px solid black;'>Precision for Retained (0)</th>
#       <th style='text-align: center;border: 1px solid black;'>Precision for Exited (1)</th>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Logistic Regression</td>
#       <td style='text-align: left;border: 1px solid black;'>82.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>69.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Random Forest</td>
#       <td style='text-align: left;border: 1px solid black;'>90.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>89.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Support Vector Machine (SVM) (RBF kernel)</td>
#       <td style='text-align: left;border: 1px solid black;'>86.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>84.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Support Vector Machine (SVM) (Poly kernel)</td>
#       <td style='text-align: left;border: 1px solid black;'>86.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>81.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Stochastic Gradient Descent (SGD)</td> 
#       <td style='text-align: left;border: 1px solid black;'>83.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>67.00 %</td>
#   </tr>
#     <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>XGBoost Classifier</td> 
#       <td style='text-align: left;border: 1px solid black;'>89.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>85.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network classifier</td> 
#       <td style='text-align: left;border: 1px solid black;'>87.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>69.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network classifier with Early Stopping</td> 
#       <td style='text-align: left;border: 1px solid black;'>87.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>69.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network Architecture with multiple Hidden layers</td> 
#       <td style='text-align: left;border: 1px solid black;'>88.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>75.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network Architecture with Early Stopping</td> 
#       <td style='text-align: left;border: 1px solid black;'>87.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>83.00 %</td>
#   </tr>
# </table>
# 
# 
# ### Recall
# 
# Recall is the ratio of correctly predicted positive observations to the all observations in actual class. The highest recall score in the model is **Random Forest Classifier model**, the **recall score** which is approximate to the **98.00**.
# 
# <table style='border:1px solid black'>
#   <tr style='font-size: 14px;'>
#       <th style='text-align: center;border: 1px solid black;'>Models</th>
#       <th style='text-align: center;border: 1px solid black;'>Recall for Retained (0)</th>
#       <th style='text-align: center;border: 1px solid black;'>Recall for Exited (1)</th>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Logistic Regression</td>
#       <td style='text-align: left;border: 1px solid black;'>98.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>15.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Random Forest</td>
#       <td style='text-align: left;border: 1px solid black;'>98.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>55.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Support Vector Machine (SVM) (RBF kernel)</td>
#       <td style='text-align: left;border: 1px solid black;'>98.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>38.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Support Vector Machine (SVM) (Poly kernel)</td>
#       <td style='text-align: left;border: 1px solid black;'>98.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>36.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Stochastic Gradient Descent (SGD)</td> 
#       <td style='text-align: left;border: 1px solid black;'>98.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>19.00 %</td>
#   </tr>
#     <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>XGBoost Classifier</td> 
#       <td style='text-align: left;border: 1px solid black;'>98.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>55.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network classifier</td> 
#       <td style='text-align: left;border: 1px solid black;'>95.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>43.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network classifier with Early Stopping</td> 
#       <td style='text-align: left;border: 1px solid black;'>95.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>43.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network Architecture with multiple Hidden layers</td> 
#       <td style='text-align: left;border: 1px solid black;'>96.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>48.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network Architecture with Early Stopping</td> 
#       <td style='text-align: left;border: 1px solid black;'>98.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>43.00 %</td>
#   </tr>
# </table>
# 
# 
# ### F1 Score
# 
# F1 Score is the weighted average of Precision and Recall.This score takes both false positives and false negatives into account. The highest F1 score in the models is **Random Forest Classifier model**, the **F1 score** approximate to the **94.00**.
# 
# <table style='border:1px solid black'>
#   <tr style='font-size: 14px;'>
#       <th style='text-align: center;border: 1px solid black;'>Models</th>
#       <th style='text-align: center;border: 1px solid black;'>F1 Score for Retained (0)</th>
#       <th style='text-align: center;border: 1px solid black;'>F1 Score for Exited (1)</th> 
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Logistic Regression</td>
#       <td style='text-align: left;border: 1px solid black;'>89.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>25.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Random Forest</td>
#       <td style='text-align: left;border: 1px solid black;'>94.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>68.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Support Vector Machine (SVM) (RBF kernel)</td>
#       <td style='text-align: left;border: 1px solid black;'>92.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>53.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Support Vector Machine (SVM) (Poly kernel)</td>
#       <td style='text-align: left;border: 1px solid black;'>91.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>50.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Stochastic Gradient Descent (SGD)</td> 
#       <td style='text-align: left;border: 1px solid black;'>90.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>30.00 %</td>
#   </tr>
#     <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>XGBoost Classifier</td> 
#       <td style='text-align: left;border: 1px solid black;'>93.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>66.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network classifier</td> 
#       <td style='text-align: left;border: 1px solid black;'>91.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>53.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network classifier with Early Stopping</td> 
#       <td style='text-align: left;border: 1px solid black;'>91.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>53.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network Architecture with multiple Hidden layers</td> 
#       <td style='text-align: left;border: 1px solid black;'>92.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>59.00 %</td>
#   </tr>
#   <tr style='font-size: 14px;'>
#       <td style='text-align: left;border: 1px solid black;'>Neural Network Architecture with Early Stopping</td> 
#       <td style='text-align: left;border: 1px solid black;'>92.00 %</td>
#       <td style='text-align: left;border: 1px solid black;'>57.00 %</td>
#   </tr>
# </table>

# In[ ]:




