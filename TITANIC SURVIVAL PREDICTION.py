#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[6]:


df = pd.read_csv('D:\codsoft\COD/Titanic-Dataset.csv')


# # Drop unnecessary columns

# In[7]:


df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'],  axis=1)


# # Handle missing value

# In[8]:


df['Age'].fillna(df['Age'].median(),inplace=True)


# # Encode categorical variables

# In[9]:


label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])


# # Replace infinite values with NaN

# In[10]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)


# # Drop rows with NaN values

# In[11]:


df.dropna(inplace=True)


# # Split the data into features and traget

# In[12]:


X = df.drop('Survived', axis=1)
y = df['Survived']


# # Split the data into training and testing sets

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Train a decision tree classifier

# In[14]:


clf =DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# # Make predictions on the test set

# In[15]:


y_pred = clf.predict(X_test)


# # Evaluate the model

# In[16]:


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


# In[18]:


print("Accuracy:{:.2f}%".format(accuracy * 100))
print("Classification Report:\n", report)


# In[ ]:




