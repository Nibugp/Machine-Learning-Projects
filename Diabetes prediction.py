#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Importing the neccassary libraries 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data Collection and Analysis
# In[5]:


# loading the dataset to a pandas dataframe

diabetes_dataset = pd.read_csv('diabetes.csv')


# In[7]:


diabetes_dataset.head()


# In[9]:


# number of rows and columns
diabetes_dataset.shape


# In[11]:


diabetes_dataset.info()


# In[12]:


# getting the statistical measure of the data
diabetes_dataset.describe()


# In[33]:


diabetes_dataset['Outcome'].value_counts()


# In[39]:


diabetes_dataset.groupby('Outcome').mean()


# In[47]:


# separating the data and labels

X = diabetes_dataset.drop(columns ='Outcome', axis =1)

Y = diabetes_dataset['Outcome']


# In[48]:


print(X)


# In[49]:


print(Y)


# In[53]:


scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)


# In[55]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[56]:


print(X)
print(Y)


# In[57]:


# split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state = 42)


# In[58]:


X.shape, X_train.shape, X_test.shape


# In[59]:


Y.shape, Y_train.shape,Y_test.shape


# In[66]:


classifier = svm.SVC(kernel = 'linear')


# In[67]:


# training the support vector machine classifier
classifier.fit(X_train,Y_train)


# In[77]:


# Model Evaluation

#accuracy score on train data

X_train_predicted = classifier.predict(X_train)

training_data_accuracy = accuracy_score(Y_train, X_train_predicted)

print('Accuracy score for the train data is', training_data_accuracy)


# In[78]:


# accuracy score on test data

X_test_predicted = classifier.predict(X_test)

test_data_accuracy = accuracy_score( Y_test, X_test_predicted)

print('Accuracy of the test data is', test_data_accuracy)


# In[88]:


# Making the predictive system

input_data = (2,197,70,45,543,30.5,0.158,53)

# Changing the input data into a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data (Assuming 'scaler' is already fitted)
std_data = scaler.transform(input_data_reshaped)
print("Standardized data:", std_data)

# Make a prediction (Assuming 'classifier' is already trained)
prediction = classifier.predict(std_data)
print("Prediction:", prediction)

if (prediction[0] == 0) :
    print("The person is non diabetic")
    
else :
    print("The person is diabetic")

