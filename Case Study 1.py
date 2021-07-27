#!/usr/bin/env python
# coding: utf-8

# # Case Study 1

# In[92]:


import numpy as np 
import pandas as pd


# In[93]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as pl2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,classification_report,roc_auc_score,precision_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense,Input,BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier


# In[94]:


df = pd.read_csv("casestudy1.csv")


# In[95]:


df


# In[96]:


# Checking if the dataset contains null values
df.isnull().sum()


# In[97]:


#Total number of unique customers
print(f"Total number of unique customers are {df.nameOrig.nunique()}")
print(f"Total number of unique customers are {df.nameDest.nunique()}")
print(f"Average no. of transactions per customer are {df.shape[0]/df.nameOrig.nunique()}")
print(f"Average no. of transactions per recipient are {df.shape[0]/df.nameDest.nunique()}")


# # Issues with the dataset
# Simple univariate and bivariate analysis is done on all the variables to diagnose the data. There are primarily four issues with the data.
# 
# First, the number of frauds are in avery minute amount in the data. From a logical standpoint, it makes sense and also it is good for the society that there are very less number of frauds happening. But from a modelling and data point of view, less number of events is a troublesome issue and there is a need of solving it before developing the model.

# In[98]:


print(df.isFraud.value_counts())
sns.countplot(y="isFraud",data=df)
plt.show()


# Secondly, there are very big outliers in the quantitative variables. Generally, very very big outliers are removed because there is a chance that it because of wrong recording. But in this case, wrong recording is not the case because very large frauds transactions tend to happen. So we cannot remove the outliers and there is a need of treating them before analysis or modelling.

# In[99]:


fig,axs = plt.subplots(2,2,figsize=(12,6))
axs[0][0].title.set_text('Histogram of transaction amount')
axs[0][0].hist(df["amount"])
axs[0][1].title.set_text("Histogram of opening customer balance")
axs[0][1].hist(df["oldbalanceOrg"])
axs[1][0].title.set_text("Histogram of closing customer balance")
axs[1][0].hist(df["newbalanceOrig"])
axs[1][1].title.set_text("Histogram of clsoing recipient balance")
axs[1][1].hist(df["newbalanceDest"])
plt.show()


# In[100]:


sns.barplot(x=df.type.unique(),y=df.groupby("type")["isFraud"].sum())


# Last but not the least, many variables are not talking with each other. For example, take a transaction from a customer "C1900366749" to recipient "C997608398". The data states that opening customer balance is 4465, closing customer balance is 0, opening recipient balance is 10845, closing recipient balance is 157982.12 and transaction amount is 9644.94. The difference between opening and closing balance of customer, difference between opening and closing balance of recipient and transaction amount must be equal which is not happening for most of the observations. (>85%) Also, the customer's account balance remained same when TRANSFER transactions happened more than 15% of the times. There are multiple reasons this can happen - Different currencies or multiple transactions at the same time or wrong recording. We don't have solution for any of these problems.

# The below graph shows the fraud attack every hour in the 744 hours. As expected there are peaks and troughs and also a very big peak. This suggests that frauds happen in short period of time.

# In[101]:


sns.lineplot(x=list(range(1,744)),y=df.groupby("step")["isFraud"].sum())
plt.xlabel("Hour of the month")
plt.ylabel("Number of transactions per hour")
plt.show()


# The below plot shows the frauds at different hours of day. It tells that frauds happen during sleeping hours the most. Close to 20% of transactions that happen during 4 AM and 5 AM are fraud transactions.

# In[102]:


df["hour"] = df.step % 24
frauds_hour = pd.concat([df.groupby("hour")["isFraud"].sum(),df.groupby("hour")["isFraud"].count()],axis=1)
frauds_hour.columns = ["Frauds","Transactions"]
frauds_hour["fraud_rate"] = frauds_hour.Frauds/frauds_hour.Transactions
sns.barplot(x=frauds_hour.index,y=frauds_hour.fraud_rate)
plt.show()


# As mentioned in the issues sections, there is a need of outlier treatment in the quantitative variables. Two popular ways to treat the outliers are transformations and capping. For transformation, Log(1+x) is a decent one when there are zeroes present in the variable and 3xp75 is a good cut-off for capping. The box plots for one variable 'amount' are shown below pre and post transformations.

# In[103]:


fig,ax = plt.subplots(1,3,figsize=(18,6))
ax[0].title.set_text("Distribution of transaction amount pre transformations")
ax[1].title.set_text("Distribution of transaction amount post log transformation")
ax[2].title.set_text("Distribution of transaction amount post capping")
sns.boxplot(x=df.isFraud, y=df.amount,ax=ax[0])
sns.boxplot(x=df.isFraud,y=np.log1p(df.amount),ax=ax[1])
df1 = df.copy()
df1[df1.amount > df1.amount.quantile(0.75)*3]["amount"] = df1.amount.quantile(0.75)*3
sns.boxplot(x=df1.isFraud,y=(df1.amount),ax=ax[2])
plt.show()


# The below plot shows the total fraud transaction amount on a daily basis

# In[104]:


df["day"] = round(df.step/24)
sns.barplot(x=list(range(1,33)),y=df[df.isFraud==1].groupby("day")["amount"].sum())
plt.xlabel("Day")
plt.ylabel("Total Fraud transaction amount")
plt.show()


# We can also see the fraud transactions per day of the week.

# In[105]:


df["dayweek"] = df.day % 7
sns.barplot(x=list(range(1,8)),y=df[df.isFraud==1].groupby("dayweek")["amount"].mean())
plt.xlabel("Day of the week")
plt.ylabel("Average Fraud transaction amount")
plt.show()


# In[106]:


df1["hourday"] = df1.step % 24
df1["hourweek"] = df1.step % (24*7)
df1["day"] = round(df1.step/24)
df1["dayweek"] = df1.day % 7
df1["daymonth"] = df1.day % 30


# In[107]:


df1 = pd.get_dummies(df1,columns=["type"])
df1.head()


# In[108]:


df1["logamount"] = np.log1p(df1["amount"])
df1["logoldbalanceOrg"] = np.log1p(df1["oldbalanceOrg"])
df1["lognewbalanceOrig"] = np.log1p(df1["newbalanceOrig"])
df1["logoldbalanceDest"] = np.log1p(df1["oldbalanceDest"])
df1["lognewbalanceDest"] = np.log1p(df1["newbalanceDest"])
df1["custdiff"] = df1["oldbalanceOrg"] - df1["newbalanceOrig"]
df1["destdiff"] = df1["oldbalanceDest"] - df1["newbalanceDest"]
df1["custind"] = np.where(df1["oldbalanceOrg"] - df1["newbalanceOrig"] == df1.amount,1,0)
df1["destind"] = np.where(df1["oldbalanceDest"] - df1["newbalanceDest"] == df1.amount,1,0)
df1["custrto"] = df1.oldbalanceOrg/(df1.newbalanceOrig+1)
df1["destrto"] = df1.oldbalanceDest/(df1.newbalanceDest+1)
df1["custdestrto1"] = df1.oldbalanceOrg/(df1.oldbalanceDest+1)
df1["custdestrto2"] = df1.newbalanceOrig/(df1.newbalanceDest+1)
df1["custamountrto"] = df1.oldbalanceOrg/(df1.amount+1)
df1["destamountrto"] = df1.oldbalanceDest/(df1.amount+1)


# In[109]:


df1 = df1.drop(["isFlaggedFraud","nameOrig","nameDest"],axis=1)


# In[110]:


X1,X2,y1,y2 = train_test_split(df1.drop("isFraud",axis=1),df1["isFraud"],test_size=0.75,random_state=1234,stratify = df1["isFraud"])
X_train,X_test,y_train,y_test = train_test_split(X1,y1,test_size=0.75,random_state=1234,stratify = y1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:





# Decision Tree Classifier

# In[111]:


from sklearn.tree import DecisionTreeClassifier,plot_tree
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as pl2

dt = pl2([
    ('sampler',RandomUnderSampler(random_state=1234,sampling_strategy='majority')),
    ('clf',DecisionTreeClassifier(max_depth=6))
    ])

dt.fit(X_train,y_train)


# In[113]:


plt.subplots(figsize=(18,10))
plot_tree(dt['clf'],feature_names = X_train.columns)


# Logistic Regression Classifier

# In[115]:


logreg = pl2([
    ('stdize',StandardScaler()),
    ('sampler',RandomUnderSampler(random_state=1234,sampling_strategy='majority')),
    ('clf',LogisticRegression(max_iter=1000000))
    ])
logreg.fit(X_train,y_train)


# In[116]:


pd.DataFrame({"Variable":list(X_train.columns),"Coefficient":logreg['clf'].coef_[0]})


# Gradient Boosting Classifier

# In[118]:


pl = pl2([
    ('sampler',RandomUnderSampler(random_state=1234,sampling_strategy='majority')),
    ('clf',GradientBoostingClassifier(max_features='sqrt',subsample=0.7))
    ])
parameters = {'clf__learning_rate':[0.07,0.1],
              'clf__n_estimators':[300,500],
              'clf__max_depth':[5]
    }

cv = GridSearchCV(pl,parameters,scoring="roc_auc",verbose=True,n_jobs=6)
cv.fit(X_train,y_train)
print(cv.best_params_)
print(cv.best_score_)


# In[119]:


imp = pd.DataFrame({"Variable":list(X_train.columns),"Importance":cv.best_estimator_['clf'].feature_importances_})
imp = imp.sort_values("Importance",ascending=False)
sns.barplot(x=imp.Importance.head(10),y=imp.Variable.head(10))


# Proposals to improve the model
# 
# • Run the entire process above on PySpark instead of Python which saves lot of time and memory.
# 
# • Explore other outlier treatment techniques to handle the quantitative variables in a better manner.
# 
# • Get more data and information to solve the problem of '2 variables not talking with each other' issue.
# 
# • More variables can be created as mentioned in the feature generation section.
# 
# • Sampling techniques other than Undersampling can be used and the best model can be taken.
# 
# • Hyperparameter tuning is not done on any model. There is still lot of scope of improvement by tuning the models.
# 
# • Selecting the best model based out of precision and recall rather than AUC and F-1 score

# In[ ]:




