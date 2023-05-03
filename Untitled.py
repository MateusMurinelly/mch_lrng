#!/usr/bin/env python
# coding: utf-8

# In[23]:


from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings 
warnings.filterwarnings('ignore')


# In[3]:


boston = load_boston()


# In[4]:


x = boston.data
y = boston.target


# In[5]:


lr = LinearRegression()


# In[7]:


scores = cross_val_score(lr,x,y,cv=5)
print('scores: ',scores)
print('Média: ',scores.mean())


# In[8]:


import matplotlib.pyplot as plt
plt.bar(range(1,6),scores)
plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Validação cruzada em regressaõ linear')
plt.show()


# In[17]:


cancer = load_breast_cancer()


# In[25]:


x = cancer.data
y = cancer.target

k=12

kf= KFold(n_splits=k, shuffle=True, random_state=42)

lr= LogisticRegression()


# In[26]:


scores = []
for train_index, test_index in kf.split(x):
    x_train,x_test = x[train_index], x[test_index]
    y_train,y_test = y[train_index], y[test_index]
    
    lr.fit(x_train,y_train)
    score = lr.score(x_test,y_test)
    
    scores.append(score)
    
mean_score = np.mean(score)
print('Scores:',scores)
print('Média:',mean_score)


# In[27]:


plt.bar(range(1,13),scores)
plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Validação cruzada em regressaõ linear')
plt.show()


# In[ ]:





# In[ ]:




