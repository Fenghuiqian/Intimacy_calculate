#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


file = pd.read_csv('test_data.csv')


# In[3]:


file.head()


# ##### 主叫号码个数

# In[4]:


file['source_data.phoneNumber'].unique().shape


# ##### 被叫号码个数

# In[5]:


file['source_data.toPhoneNumber'].unique().shape


# ##### 主叫号码的总通话次数    total_num_of_calls

# In[6]:


file['total_num_of_calls'] = file.groupby('source_data.phoneNumber')['source_data.toPhoneNumber'].transform('count')


# ##### 主叫号码的总通话时长    total_duation

# In[7]:


file['total_duation'] = file.groupby('source_data.phoneNumber')['source_data.duation'].transform('sum')


# ##### 时间特征处理
# 9:00-18:00 : work_time  
# 0:00~9:00 & 18:00-0:00 :personal_time

# In[8]:


time_series = pd.to_datetime(file['source_data.eventTime'])


# In[9]:


file['personal_time'] = time_series.dt.hour.apply(lambda x: 1 if x>18 or x<9 else 0)


# In[10]:


file['work_time'] = time_series.dt.hour.apply(lambda x: 1 if 9 <= x <= 18 else 0)


# In[11]:


file.shape


# In[12]:


file.head()


# ##### 聚合特征

# In[13]:


features = file.groupby(['source_data.phoneNumber','source_data.toPhoneNumber']).agg(
                                                                                    {'source_data.duation': ['count', 'sum'], 
                                                                                     'personal_time':'sum', 
                                                                                     'work_time': 'sum', 
                                                                                     'total_duation': 'first',
                                                                                     'total_num_of_calls': 'first'})


# In[14]:


features.shape


# ###### 特征解释
# duation_count：某号码向某号码的通话次数  
# duation_sum：某号码向某号码的通话总时长  
# personal_time： 某号码向某号码，在个人时间的通话次数  
# work_time：某号码向某号码，在工作时间的通话次数  
# total_duation： 某号码的总通话时长  
# total_num_of_calls：某号码的总通话次数  

# In[15]:


features.columns = [ 'duation_count', 'duation_sum', 'personal_time', 'work_time', 'total_duation', 'total_num_of_calls']


# In[16]:


features.head()


# ##### 亲密度计算

# In[17]:


# 特征归一化
features_normalize = (features - features.min()) / (features.max() - features.min())


# ##### 特征与亲密度的关系估计 
# duation_count， duation_sum， personal_time与亲密度成正比关系  
# work_time， total_duation， total_num_of_calls与亲密度成反比关系

# In[18]:


# 权重假设
weights = {
    'duation_count': 0.4,
    'duation_sum': 0.4,
    'personal_time': 0.2,
    'work_time': -0.2,
    'total_duation': -0.1,
    'total_num_of_calls': -0.1}


# In[20]:


# 添加亲密度intimacy列
features_normalize['intimacy'] = 0
for feature in weights.keys():
    features_normalize['intimacy'] += features_normalize[feature] * weights[feature]


# ##### 亲密度 intimacy归一化

# In[23]:


features_normalize['intimacy'] = (features_normalize['intimacy'] - features_normalize['intimacy'].min()) / (features_normalize['intimacy'].max() - features_normalize['intimacy'].min())


# In[24]:


# 亲密度intimacy 排序
features_normalize.sort_values(by='intimacy', ascending=False, inplace=True) 


# In[25]:


result = features_normalize.loc[:, ['intimacy']]


# In[26]:


result.shape


# In[27]:


result.head()


# In[28]:


result.to_csv('result.csv')


# In[ ]:




