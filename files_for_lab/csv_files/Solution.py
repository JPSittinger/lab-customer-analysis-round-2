#!/usr/bin/env python
# coding: utf-8

# # Round 2

# In[1]:

import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


data = pd.read_csv('marketing_customer_analysis.csv')
data


# In[3]:


data.shape


# In[4]:


data.columns


# In[5]:


data = data.drop(['Unnamed: 0', 'Customer'], axis=1)


# In[6]:


print(data.columns)


# In[7]:


data.columns = list(map(lambda el: el.lower(), data.columns))
data.head()


# In[8]:


data.dtypes


# In[9]:


#print(data._get_numeric_data())
data.select_dtypes(include= np.number).columns


# In[10]:


data.select_dtypes(include= object).columns


# In[11]:


data.isna().sum()


# In[12]:


data.duplicated().sum()


# In[13]:


data = data.drop_duplicates(keep='first')
data.shape


# In[14]:


round(data.isna().sum()/len(data),4)*100  # shows the percentage of null values in a column
nulls_df = pd.DataFrame(round(data.isna().sum()/len(data),4)*100)
nulls_df
nulls_df = nulls_df.reset_index()
nulls_df
nulls_df.columns = ['header_name', 'percent_nulls']
nulls_df


# Looks like 50% of vehicle type is missing either way, so imputing it with the most common vehicle type will likely scew the data too much. Probably best to just drop vehicle type all the way, filling in half of the data with "guesses" isn't accurate enough anymore and dropping those cases would be loss of a lot of data. So just dropping the column "vehicle type", lol.
# 
# Also, state, response, months since last claim, number of open complaints, vehicle class and vehicle size are still below 6% of missings, dropping those nA's would be a loss of valuable data so better impute it with the most common value for categorial data and the mean for the numericals (just assuming mean is the better value here, no time to check if median might be better, way too hungry).

# In[15]:


data = data.drop(['vehicle type'], axis=1)


# In[16]:


mean_months_since_last_claim = data['months since last claim'].mean()
mean_months_since_last_claim
data['months since last claim'] = data['months since last claim'].fillna(mean_months_since_last_claim)


# In[17]:


mean_number_of_open_complaints = data['number of open complaints'].mean()
mean_number_of_open_complaints
data['number of open complaints'] = data['number of open complaints'].fillna(mean_number_of_open_complaints)


# In[18]:


# Replacing null values for categorical variables
data['state'].unique()
data['state'].value_counts()

data['state'].value_counts(dropna=False)
#len(data[data['state'].isna()==True])  # number of missing values


# In[19]:


# use most common value to fillna
data['state'] = data['state'].fillna('California')
len(data[data['state'].isna()==True]) # now this number is 0
data['state'].value_counts(dropna=False)


# In[20]:


data['response'].unique()
data['response'].value_counts()

data['response'].value_counts(dropna=False)


# In[21]:


data['response'] = data['response'].fillna('No')
len(data[data['response'].isna()==True]) # now this number is 0
data['response'].value_counts(dropna=False)


# In[22]:


data['vehicle class'].unique()
data['vehicle class'].value_counts()

data['vehicle class'].value_counts(dropna=False)


# In[23]:


data['vehicle class'] = data['vehicle class'].fillna('Four-Door Car')
len(data[data['vehicle class'].isna()==True]) # now this number is 0
data['vehicle class'].value_counts(dropna=False)


# In[24]:


data['vehicle size'].unique()
data['vehicle size'].value_counts()

data['vehicle size'].value_counts(dropna=False)


# In[25]:


data['vehicle size'] = data['vehicle size'].fillna('Medsize')
len(data[data['vehicle size'].isna()==True]) # now this number is 0
data['vehicle size'].value_counts(dropna=False)


# In[26]:


round(data.isna().sum()/len(data),4)*100  # shows the percentage of null values in a column
nulls_df = pd.DataFrame(round(data.isna().sum()/len(data),4)*100)
nulls_df
nulls_df = nulls_df.reset_index()
nulls_df
nulls_df.columns = ['header_name', 'percent_nulls']
nulls_df

# Yasss, no missing values, BUT not happy with the solution for 'response', scews the data too much.


# In[27]:


data['effective to date'] = pd.to_datetime(data['effective to date'], errors='coerce')


# In[28]:


data['month'] = pd.DatetimeIndex(data['effective to date']).month
data


# In[29]:


data['month'].unique()
data['month'].value_counts()
# No March in the months, so can kinda just ignore that for the first quarter. Ayy, wait a sec, there is only January and February already, probably pre-filtered. So no need to filter anything out again.


# In[30]:


# Just for fun, this would be the code to filter for the first quarter without even needing to add the month column:
data_quarterone = data[data['effective to date'].dt.quarter == 1]


# # ROUND 3

# In[31]:


data.info(verbose=True)


# In[32]:


data.describe()


# In[33]:


data['response_numerical'] = data['response'].map(dict(Yes=1, No=0))
data


# In[34]:


sns.countplot(x=data['response'])


# In[35]:


import matplotlib.pyplot as plt
import seaborn as sns 
#get_ipython().run_line_magic('matplotlib', 'inline')
sns.barplot(x="response_numerical", y="sales channel", data=data)
plt.show()


# In[36]:


data['total claim amount_bin'] = pd.qcut(data['total claim amount'], q=10)

sns.barplot(x="response_numerical", y="total claim amount_bin", data=data)
plt.show()

# def income_bins(x):
#     if x <20000:
#         return 1
#     elif x<40000:
#         return 2
#     elif x<60000:
#         return 3
#     elif x<80000:
#         return 4
#     else:
#         return 5


# In[37]:


data['income_bin'] = pd.qcut(data['income'], q=10, duplicates='drop')

sns.barplot(x="response_numerical", y="income_bin", data=data)
plt.show()


# # ROUND 4

# In[38]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import MinMaxScaler # do not use the function Normalise() - it does something entirely different
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[39]:


data.dtypes


# In[40]:


def clean_data(df):
    
    df.columns=[e.lower().replace(' ', '_') for e in df.columns]
    
    df=df.drop(columns=['unnamed:_0'])
    
    df['vehicle_type']=df['vehicle type'].fillna('M')
    
    df['effective_to_date']=pd.to_datetime(df['effective_to_date'], errors='coerce')
    df['month'] = df['effective_to_date'].dt.month
    
    df=df.dropna()
    return df


# In[41]:


numerical = data.select_dtypes(include=np.number)
numerical


# In[42]:


categoricals = data.select_dtypes(include=['object'])
categoricals


# In[43]:


#SOLUTION USING SEABORN IS BELOW; LOOKS NICER AND WORKS BETTER
numerical.hist(column=None, by=None, grid=True, xlabelsize=None, xrot=None, ylabelsize=None, yrot=None, ax=None, sharex=False, sharey=False, figsize=(20, 20), layout=None, bins=100, backend=None, legend=False)


# In[44]:


sns.displot(numerical['customer lifetime value'])
plt.show()


# In[45]:


for column in numerical:
    sns.displot(numerical[column])


# In[46]:


# for column in numerical:
#     plt.hist(numerical[column])

# plt.hist(numerical['customer lifetime value'])

numerical.hist(bins=30, figsize=(15, 20))


# In[47]:


correlations_matrix = numerical.corr()
correlations_matrix

plt.figure(figsize = (16,5))
sns.heatmap(correlations_matrix, annot=True)
plt.show()


# Pretty much no multicollinearity between the features

# # ROUND 5

# In[48]:


# # setting y to our target
# y = numerical['total claim amount']
# # putting everything else to x
# X = numerical.drop(['total claim amount'], axis=1)


# In[49]:


# Normalize between 0 and 1

# transformer = MinMaxScaler().fit(X)
# x_normalized = transformer.transform(X)
# print(x_normalized.shape)
# x_normalized
# pd.DataFrame(x_normalized, columns=X.columns)


# In[50]:


# # scaling standard scaler: make data distributed with mean=0 and std=1
# transformer = StandardScaler().fit(X)
# x_standardized = transformer.transform(X)
# print(x_standardized.shape)
# pd.DataFrame(x_standardized, columns=X.columns)


# # ROUND 6

# In[51]:


categoricals = categoricals.drop('response', axis=1)
# categoricals


# In[52]:


df = pd.concat([categoricals, numerical], axis=1)
df


# In[53]:


# setting y to our target
y = df['total claim amount']
# putting everything else to x
X = df.drop(['total claim amount'], axis=1)


# In[54]:


# splitting X into numericals and categoricals
X_num = X.select_dtypes(include=np.number)
X_cat = X.select_dtypes(include=['object'])
print(X_num.columns)
print(X_cat.columns)


# In[55]:


# Normalize between 0 and 1

transformer = MinMaxScaler().fit(X_num)
X_normalized = transformer.transform(X_num)
print(X_normalized.shape)
X_normalized = pd.DataFrame(X_normalized,columns=X_num.columns)
pd.DataFrame(X_normalized, columns=X_num.columns)


# In[56]:


X_normalized.describe().T


# In[57]:


# summe = 0
# for i in X_cat.columns:
#     a = X_cat[i].unique()
#     for i in a:
#         summe += 1
#     summe -=1
# summe


# In[58]:


#OneHotEncoding the categorical values
from sklearn.preprocessing import OneHotEncoder
encoder2 = OneHotEncoder(drop='first').fit(X_cat)
encoded = encoder2.transform(X_cat).toarray()
encoded
cols = encoder2.get_feature_names_out(input_features=X_cat.columns)
cols
# # Note: in version 1.0 and higher of sklearn this method is called 'get_feature_names_out()'
# #cols
onehot_encoded = pd.DataFrame(encoded, columns=cols)
onehot_encoded


# In[59]:


X = pd.concat([X_normalized, onehot_encoded], axis=1)
X.head()


# ## Hold on to your seats, I'm about to train-test split!

# In[60]:


y = df['total claim amount']
y.head()


# In[ ]:


# # train test split is the way ML generates its claim to fame: 
# # we build the model on a portion of the data but we then validate it in 
# # another "fresh" portion
# # our model has no opportunity to "cheat": it must accurately guess the values 
# # in the "fresh" dataset that it never saw before
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # random_state is only here because he wants the same train and test set every time, helps him in education 
# # but otherwise you leabe it out


# In[ ]:


# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)


# In[ ]:


# #we train/fit our model like yesterday
# lm = linear_model.LinearRegression()
# lm.fit(X_train,y_train)


# In[ ]:


# from sklearn.metrics import r2_score
# predictions = lm.predict(X_train)
# r2_score(y_train, predictions)


# In[ ]:


# # But now we evaluate it in the TEST portion of the data, that we did not use for training.
# # This way we know our model is genuinely guessing our donations, not just repeating the values it has seen in the training data


# predictions_test = lm.predict(X_test)
# r2_score(y_test, predictions_test)


# In[ ]:


# from sklearn.metrics import mean_squared_error
# mse=mean_squared_error(y_test,predictions_test)
# mse


# In[ ]:


# rmse = np.sqrt(mean_squared_error(y_test,predictions_test))
# rmse


# In[62]:


import numpy as np
from sklearn.model_selection import train_test_split

"""
class LinearRegression:
    
    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range (self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr *db
        
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
        
"""

# In[63]:

#import LinearRegression
from LinearRegression import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(y_test)

#fig = plt.figure(figsize=(8,6))
#plt.scatter(X[:0], y, color ="b", marker="o", s=30)
#plt.show()

reg = LinearRegression()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

mse = mse(y_test,predictions)
print(round(mse, 2))


# In[ ]:




