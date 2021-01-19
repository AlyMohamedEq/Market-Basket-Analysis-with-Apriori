#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing pip to get mlxtend library
import pip
pip.main(['install', 'mlxtend'])


# In[3]:


pip.__version__


# In[4]:


# importing pandas and apriori
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[5]:


# importing the needed file
df = pd.read_excel('E:\Basics of ML & Python\Online Retail.xlsx')


# In[6]:


# display metadate 
df.info()


# In[7]:


# sum the null values in every column
df.isnull().sum()


# In[9]:


# eliminating spaces at the end and beginning of strings 
df['Description'] = df['Description'].str.strip()


# In[10]:


# review first 5 rows in the date 
df.head()


# In[15]:


# filtering by quantity more than 6 
df[df['Quantity']>6]


# In[16]:


#identifing invoiceNo column as string 
df ['InvoiceNo'] = df ['InvoiceNo'].astype('str')


# In[20]:


# droping null values from invoiceNo column 
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)


# In[21]:


# selecting all values except credit 
df2 = df[~ df['InvoiceNo'].str.contains('C')]


# In[23]:


# displaying first 5 rows
df2.head()


# In[24]:


# filtering by country 'germany'
basket = df2[df2['Country'] == 'Germany']


# In[25]:


# displaying first 5 rows
basket.head()


# In[26]:


# group by invoice and description ,sum quantity and group by it , unstack to make rows colunms , fill null values with 0  
basket = basket.groupby(['InvoiceNo','Description'])['Quantity'].sum().unstack().fillna(0)


# In[28]:


# displaying first 20 rows
basket.head(20)


# In[30]:


# creating a fn return values more than or equal 1 to 1 , less than or equal 0 to 0
def encode_units(x):
    if x >= 1 :
        return 1
    if x <= 0 :
        return 0


# In[31]:


# applying fn on data 
basket_set = basket.applymap(encode_units)


# In[32]:


# display the data 
basket_set


# In[33]:


# drop postage colunm 
basket_set.drop('POSTAGE', axis=1, inplace=True)


# In[34]:


# identifing frequent itemset with min support 
frequent_items = apriori(basket_set, min_support= 0.07, use_colnames=True)


# In[36]:


# displaying the frequent items 
frequent_items.head(10)


# In[94]:


# making the rule using the frquent itemset and confidence with min 7%
rules = association_rules(frequent_items, metric='confidence', min_threshold=0.07)


# In[95]:


rules.head()


# In[ ]:




