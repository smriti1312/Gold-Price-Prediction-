

# #  ML PROJECT
# ## GOLD  PRICE PREDICTION 

# In[4]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[5]:


# loading the csv data to a Pandas DataFrame
gold_data = pd.read_csv('gld_price_data.csv')


# In[6]:


# print first 5 rows in the dataframe
gold_data.head()


# In[7]:


# print last 5 rows of the dataframe
gold_data.tail()


# In[9]:


# data info
gold_data.info()


# In[10]:


# checking the number of missing values
gold_data.isnull().sum()


# In[11]:


# getting the statistical measures of the data
gold_data.describe()


# In[12]:


correlation = gold_data.corr()


# In[9]:


# constructing a heatmap to understand the correlatiom
plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='Blues')


# In[13]:


# correlation values of GLD
print(correlation['GLD'])


# In[14]:


# checking the distribution of the GLD Price
sns.distplot(gold_data['GLD'],color='green')


# In[15]:


#Splitting the Features and Target
X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']


# In[13]:


print(X)


# In[16]:


print(Y)


# In[19]:


#Splitting into Training data and Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)


# In[40]:


#model training
model = RandomForestRegressor(n_estimators=100)


# In[41]:


# training the model
model.fit(X_train,Y_train)


# In[42]:


#model evaluation
# prediction on Test Data
test_data_prediction = model.predict(X_test)


# In[43]:


print(test_data_prediction)


# In[44]:


# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)


# In[47]:


#Compare the Actual Values and Predicted Values in a Plot
Y_test = list(Y_test)


# In[48]:


# visualisation of Actual price vs Predicted price
plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[49]:


#Testing scores
testing_test_data_prediction = model.score(X_test, Y_test)
print("Model Score/Performance on Testing data",testing_test_data_prediction)


# In[50]:


training__test_data_prediction = model.score(X_train, Y_train)
print("Model Score/Performance on Training data",training__test_data_prediction)


# In[51]:


# Checking working of the model
input_data=(1447.160034 , 78.470001 , 15.1800 , 1.471692)
input_array=np.asarray(input_data)
reshape_data =input_array.reshape(1,-1)
new_pred =model.predict(reshape_data)
print('Price of GOLD predicted by the model : = ')
print(new_pred)


# #                           THANK YOU
