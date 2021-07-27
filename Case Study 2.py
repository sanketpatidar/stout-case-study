#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("casestudy2.csv")


# In[3]:


df


# In[4]:


# Finding the current year my selecting the unique years and adding them into a list and finding the max so that
# everytime current year will be the most recent year in the dataset.

current_year = list(df['year'].unique())
current_year = max(current_year)
current_year


# In[5]:


df_2017 = df[df['year'] == current_year]
df_2016 = df[df['year'] == 2016]
df_2017


# # •	Total revenue for the current year

# In[6]:


total_rev_2017 = df_2017['net_revenue'].sum()
total_rev_2017


# # •	New Customer Revenue e.g. new customers not present in previous year only

# In[7]:


def new_cust_revenue(df):
    filtered_df = df[(df['year'] < current_year)]
    
    # selecting all the users who were present in previous years and created the list of users email id
    emails_ls = list(filtered_df['customer_email'])
    
    # Here selecting only those users who does not present in previous years
    inverse_boolean_series = ~(df.customer_email.isin(emails_ls))

    filtered_df1 = df[inverse_boolean_series]
    
    New_customer_revenue = filtered_df1.net_revenue.sum()
    
    return("The revenue of new customer is- {}".format(New_customer_revenue))

new_cust_revenue(df)


# # •	Existing Customer Growth. To calculate this, use the Revenue of existing customers for current year –(minus) Revenue of existing customers from the previous year

# In[8]:


sum_curr = df_2017['net_revenue'].sum()
sum_prior = df_2016['net_revenue'].sum()

cust_growth = sum_curr - sum_prior
print("The Existing custome growth is-", cust_growth)


# # •	Existing Customer Revenue Current Year

# In[9]:


sum_curr = df_2017['net_revenue'].sum()
print("Existing customer revenue current year-",sum_curr)


# # •	Existing Customer Revenue Prior Year

# In[10]:


sum_prior = df_2016['net_revenue'].sum()
print("Existing customer revenue prior year-",sum_prior)


# # •	Total Customers Current Year

# In[11]:


total_cust_curr = df_2017['customer_email'].count()
print("Total customers current year-",total_cust_curr)


# # •	Total Customers Previous Year

# In[12]:


total_cust_prev = df_2016['customer_email'].count()
print("Total customers previous year-",total_cust_prev)


# # •	New Customers

# In[13]:


filtered_df = df[(df['year'] < current_year)]
    
# selecting all the users who were present in previous years and created the list of users email id
emails_ls = list(filtered_df['customer_email'])

# Here selecting only those users who does not present in previous years
inverse_boolean_series = ~(df.customer_email.isin(emails_ls))

filtered_df1 = df[inverse_boolean_series]

print("New customers-")
filtered_df1

