#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score


# In[2]:


#Data 
data_path = './data/raw/train.csv'
t_data_path = './data/raw/test.csv'

df_train = pd.read_csv(data_path)
df_test = pd.read_csv(t_data_path)


# Gathering information about the data.

# In[3]:


df_train.info()


# In[4]:


df_train.describe()


# In[5]:


df_train.shape


# In[6]:


df_train.head()


# Checking for missing values

# In[7]:


df_train.isna().sum()


# Since there are missing values, it's interesting to check how much it represents for each respective column.

# In[8]:


#To evaluate how much of the missing values represent from each respective columns
total = df_train.isnull().sum().sort_values(ascending=False)
percent_1 = df_train.isnull().sum()/df_train.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(3)


# As it seens, missing values are barely noticeable within the data. Still, some preprocessing will be made rather than just dropping the missing data.
# 
# Now, some analysis will be carried about the data and what insights can be obtained from it.

# In[9]:


#Analysing Churn
df_train['Churn'].value_counts()


# As it seens, it's a imbalanced dataset, since positive churn represents less than 30% of the total data, therefore this will be a imbalanced classification analysis. 

# In[10]:


#Churn Percentage
churn_df = df_train['Churn'].value_counts()
churn_df = churn_df.set_axis(['Not Cancelled','Cancelled']) 
exp = [0.05,0.0]
plt.title('Churn Percentage')

plt.pie(churn_df, autopct = '%.02f%%',labels = churn_df.index, shadow = True, explode = exp,startangle= 90,wedgeprops = {'edgecolor':'black','linewidth':0.5,})


# Checking for correlations.
# First, analysing some categorical features that may enlight this analysis. 
# For example, how's the churn related to:
# * Having Phone Service and Multiple lines
# * Having internet service, online security and backup
# * Having Device protection and tech support
# * Streaming TV and Movies
# * Contract type
# * Presence or absense of parents and dependents
# * Being or not a senior citizen
# 
# 

# In[11]:


FacetGrid = sns.FacetGrid(df_train, row='InternetService', height=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot,  'OnlineSecurity','Churn','OnlineBackup' , order=None, hue_order=None )
FacetGrid.add_legend();


# As it seens, there's a higher chance of Churn if it'a a DSL user with Online Security, but this is inverted with Fiber optic. Basically no churn occurs with no internet service.
# 
# Proceeding to deeper analysis based on monthly charges.

# In[12]:


grid = sns.FacetGrid(df_train, col='Churn', row='InternetService', height=3.2, aspect=1.6, hue= 'OnlineSecurity')
grid.map(plt.hist, 'MonthlyCharges', alpha=.5, bins=20)
grid.add_legend();


# It is interesting to see that higher churn occurs with Fiber optic on high monthly charges and mostly with Online Security service.
# 
# Following, relation between churn and contract type was analyzed.

# In[13]:


df_train['Contract'].unique()


# In[14]:


#Relation with contract
sns.countplot(x = 'Contract', data = df_train, hue = 'Churn')


# In[15]:


sns.countplot(x = 'Contract', data = df_train)


# Month-to-month contracts are the highest in count, and also has a big churn among clients. It is interesting to contact clients with this contract type to offer discounts that could avoid these occurances.
# 
# Analyzing now the churn occurance relation with being a senior citizen.

# In[16]:


#Check if churn occurs more on senior citizens or not
sns.countplot(x = 'SeniorCitizen',data = df_train, hue = 'Churn')


# In[17]:


#Senior Churn Percentage
senior_pct = df_train.loc[(df_train['SeniorCitizen'] == 1), 'Churn'].value_counts()/df_train.loc[(df_train['SeniorCitizen'] == 1), 'Churn'].value_counts().sum()
senior_pct = senior_pct.set_axis(['Not Cancelled','Cancelled']) 
exp = [0.05,0.0]
plt.title('Senior Churn Percentage')

plt.pie(senior_pct, autopct = '%.02f%%',labels = senior_pct.index, shadow = True, explode = exp,startangle= 90,colors = ('darkblue','orange'),wedgeprops = {'edgecolor':'black','linewidth':0.5,})


# Senior citizens are minority among clients of this company. Still, churn ocurred in almost 50% of senior citizen clients. It's important to see if a relation with contracted services exists.

# In[18]:


#What services does senior citizens use the most?
senior_df = df_train[df_train['SeniorCitizen'] == 1]
fig, axs = plt.subplots(ncols= 2, nrows = 2 )
sns.countplot(x = 'PhoneService', data = senior_df, hue = 'Churn', ax = axs[0][0])
sns.countplot(x = 'InternetService', data = senior_df, hue = 'Churn', ax = axs[0][1])
sns.countplot(x = 'StreamingTV', data = senior_df, hue = 'Churn', ax = axs[1][0])
sns.countplot(x = 'TechSupport', data = senior_df, hue = 'Churn', ax = axs[1][1])
plt.tight_layout()


# Churn, as shown before, almost do not occur with costumers that decided for no internet service, and in this case for no phone service. The highest count occurs with costumers with No Tech Support. Since we're dealing with senior citizens, the absence of technical support can lead to difficulties with the use of services, what can motivate the service cancellation. 
# 
# The high churn among senior costumerss that doesn't have technical support brought a question in mind: is this a pattern over all clients? So, this analysis was done with a Categorical Plot.

# In[19]:


axes = sns.catplot(x = 'TechSupport',y = 'Churn', 
                      data=df_train, aspect = 2.5,kind = 'point');
axes.fig.suptitle('Churn relation with Technical Support Service',fontsize = 15)


# The next analysis done was if there was any relation between churn and costumer having partners/dependents, which brought that clients with no partners/dependents are more inclined to churn.
# 
# After that, the last analysis was about streaming TV and movies, to check how many costumers decided for both services.
# 

# In[20]:



fig, axes = plt.subplots(ncols = 2)
sns.countplot(x = 'Churn', data = df_train, hue = 'Partner', ax = axes[0], palette='Set2')
sns.countplot(x = 'Churn', data = df_train, hue = 'Dependents', ax = axes[1],  palette='Set2')
fig.suptitle('Churn relation with customers family')
plt.tight_layout()


# A new feature can be created to encompass if the costumer has a full family (partner and dependents) or not.

# In[21]:


plt.title('Streaming Services')
sns.countplot(x = 'StreamingTV', data = df_train, hue = 'StreamingMovies')


# As it seens, hardly someone who hires StreamingTV doesn't hire Streaming Movies, so a new feature can be created to encompass having both streaming services.
# 
# Now, the preprocessing of the data.

# In[22]:


#Missing Values will be imputed by mode for categorical and mean for numerical values
dependents_mode = df_train['Dependents'].mode()
payment_mode = df_train['PaymentMethod'].mode()
tenure_mean = df_train['tenure'].mean()


# In[23]:


df_train['Dependents'] = df_train['Dependents'].fillna(dependents_mode[0])
df_train['Dependents'].isna().sum()


# In[24]:


df_train['PaymentMethod'] = df_train['PaymentMethod'].fillna(payment_mode[0])
df_train['PaymentMethod'].isna().sum()


# In[25]:


df_train['tenure'] = df_train['tenure'].fillna(tenure_mean)
df_train['tenure'].isna().sum()


# In[26]:


#Feature Engineering
#Creating a Full Family feature if Dependents and Partners are 1
df_train.loc[(df_train['Dependents'] == 'Yes') & (df_train['Partner'] == 'Yes'), 'Full_Family'] = 'Yes'
df_train.loc[(df_train['Partner'] == 'No') |              (df_train['Dependents'] == 'No')              ,'Full_Family'] = 'No'


# In[27]:


df_train.head()


# In[28]:


#Creating feature about full streaming service
df_train.loc[(df_train['StreamingTV'] == 'Yes') & (df_train['StreamingMovies'] == 'Yes'), 'Full_Streaming(TV/Movies)'] = 'Yes'
df_train.loc[((df_train['StreamingTV'] == 'Yes') & (df_train['StreamingMovies'] == 'No')) |              ((df_train['StreamingTV'] == 'No') & (df_train['StreamingMovies'] == 'Yes'))              | ((df_train['StreamingTV'] == 'No') & (df_train['StreamingMovies'] == 'No'))             | (df_train['StreamingTV'] == 'No internet service') | (df_train['StreamingMovies'] == 'No internet Service'),'Full_Streaming(TV/Movies)'] = 'No'


# In[29]:


#New feature based on Monthly Charges and Tenure. On previous feature importance analysis not shown here, these two had the highest values, so a combination between those could led to a strong feature for the model.
df_train['Monthly_Tenure_Ratio'] = df_train['tenure']/df_train['MonthlyCharges']


# In[30]:


df_train_n = df_train.drop(['StreamingTV','StreamingMovies','Dependents'],axis = 1)
df_train_n.head()


# In[31]:


#Preprocessing - Binary and One-Hot Encoding features. Those which have only two categories (like gender) will be binary encoded.
X = df_train_n.copy()
y = X.pop('Churn')

columns = ['PhoneService','PaperlessBilling','Full_Family','Full_Streaming(TV/Movies)']
X['gender'] = X['gender'].replace({'Male': 0, 'Female':1})
for c in columns:
    X[c] = X[c].replace({'Yes': 0, 'No':1})
X.head()


# Here starts the Machine Learning Algorithm preparation. At first, RandomForestClassifier was used, but after a comparing with LogisticRegression, the latter retrieved better results, being the chosen model. Then, a hyperparameter tuning was applied to improve the model performance.

# In[32]:


#Data Split
X_train_full,X_valid_full,y_train,y_valid = train_test_split(X,y,test_size = 0.3,random_state=42)


# In[33]:


categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()


# In[34]:


#Applying One Hot Encoding
OHE = OneHotEncoder()

scaler = StandardScaler()

RFC = RandomForestClassifier(n_estimators = 100, min_samples_leaf= 2,min_samples_split= 2,max_features=0.35,random_state=42)
LGR = LogisticRegression(random_state = 42)

transformer = ColumnTransformer([('cat_cols', OHE, categorical_cols),
                                ('num_cols', scaler, numerical_cols)])

pipe = Pipeline([("preprocessing", transformer),
                ("classifier", LGR)])

pipe.fit(X_train, y_train)
pipe.score(X_train,y_train)


# In[35]:


preds = pipe.predict(X_valid)
acc = accuracy_score(preds,y_valid)
acc


# In[36]:


#Hyperparameter Tuning
grid = {'classifier__solver':['newton-cg', 'lbfgs', 'sag', 'saga'],'classifier__C':np.linspace(0.1,5,num = 10),'classifier__max_iter': range(1000,2500,250),
        'classifier__class_weight':[None,'balanced',0.25],
        'classifier__tol':[0.0001,0.01,0.1]
}
grid_search = GridSearchCV(estimator = pipe,param_grid = grid,cv = 5, return_train_score = True)
grid_search.fit(X_train,y_train)


# In[37]:


grid_search.best_params_


# In[38]:


grid_search.cv_results_['mean_train_score'].mean()


# In[39]:


tuned_LGR = LogisticRegression(solver= 'newton-cg',C = 0.1, class_weight = None, max_iter = 1000, tol = 0.0001,random_state = 42)
new_pipe = Pipeline([("preprocessing", transformer),
                ("classifier", tuned_LGR)])

new_pipe.fit(X_train, y_train)
new_pipe.score(X_train,y_train)


# In[40]:


new_preds = new_pipe.predict(X_valid)
n_acc = accuracy_score(new_preds,y_valid)
n_acc


# Hyperparameter tuning until this moment didn't lead to any improvements over the results. Further analysis will be done to obtain higher scores.
# 
# Below, the information about the quality of the predition using classification report and confusion matrix.
# 

# In[41]:


from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_valid,new_preds))


# In[42]:


cm = confusion_matrix(y_valid, new_preds)
target_names = ['Not Cancelled','Cancelled']
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix', fontsize = 15)
plt.savefig('Confusion_Matrix.png')
plt.show(block=False)


# In[43]:


rec = recall_score(y_valid, new_preds)
rec


# Since this is a imbalanced dataset, accuracy isn't the best metric for this case. As shown above, the model shown 55% of true positives for churn in the predictions, which means that over half of cancellations can be predicted and avoided. 
# 
# Below, is shown the feature importance of the dataset for the model.

# In[44]:


ft_imp = pipe.steps[1][1].coef_[0]
#ft_imp_im = ft_imp.T.tolist()


# In[45]:



encoded_cols = list(pipe['preprocessing'].transformers_[0][1].get_feature_names_out())


# In[46]:


num_cols = list(pipe['preprocessing'].transformers_[1][1].get_feature_names_out())


# In[47]:


total_cols = encoded_cols + num_cols

Feature_imp = pd.Series(ft_imp,index= total_cols)
Feature_imp = Feature_imp.sort_values(ascending = False)
Feature_imp


# In[48]:


plt.figure(figsize = (6,14))
plt.title('Feature Importance',fontsize = 15)
Feature_imp.plot.barh()


# With the model ready, the last step was the deployment to the service. The deploy was done using Pickle.

# In[49]:


import pickle

pickle.dump(pipe, open('./models/modelo_churn.pkl', 'wb'))


# The model was implemented in a application that will recieve information of a certain client, predicting if this costumer is a potential case of churn, in order to avoid and help the marketing team to contact and offer solutions so so that the customer's satisfaction with the company's service increases.

# In[ ]:




