#!/usr/bin/env python
# coding: utf-8

# In[1]:


#claimants=car insurance claimers
#casenum=unique case claim number
#ATTORNEY=legally can he claim
#CLMSEX=sex of claimants
#CLMINSUR=identifies claimer has  insurance or not
#SEATBELT=wheather he wear the seat belt or not
#CLMAGE=age of claimant
#LOSS=loss of money of claimer for the incident
#we predict"whether the person is capable of insurance or not"


# In[39]:


#step 1:problem framing
#to predict car insurance eligibility prediction


# In[2]:


##############  step2: collecting and reading data    ##################


# In[1]:


import pandas as pd
df=pd.read_csv('claimants.csv')
df.head()


# In[4]:


df.shape


# In[40]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df['CLMINSUR'].value_counts()


# In[ ]:


###### step3:cleaning data ############


# In[2]:


df['CLMINSUR'].fillna(1.0,inplace=True)


# In[34]:


df['SEATBELT'].value_counts()


# In[3]:


df['SEATBELT'].fillna(0.0,inplace=True)


# In[4]:


df['CLMAGE'].fillna(df['CLMAGE'].mean(),inplace=True)


# In[5]:


df['CLMSEX'].fillna(1.0,inplace=True)


# In[6]:


df.isnull().sum()


# In[14]:


###########step4: visualisation-EDA   ##########


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

# Distribution plots for numerical columns
numeric_columns = ['ATTORNEY', 'CLMAGE', 'LOSS']

for col in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[10]:


# Correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[12]:


# Pairplot to visualize relationships
sns.pairplot(df)
plt.show()

# Scatter plot to check the relationship between age and loss
plt.figure(figsize=(8, 4))
sns.scatterplot(x='CLMAGE', y='LOSS', data=df)
plt.title('Relationship between CLMAGE and LOSS')
plt.show()


# In[15]:


df['CLMAGE'].hist()


# In[16]:


df['LOSS'].hist()


# In[ ]:


### step 5: oulier removal ######


# In[7]:


import numpy as np
import pandas as pd

# Function to replace outliers with the median for a single column
def replace_outliers_with_median(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    
    # Calculate IQR
    iqr = q3 - q1
    
    # Define whisker thresholds (1.5 times IQR)
    w1 = q1 - 1.5 * iqr
    w2 = q3 + 1.5 * iqr
    
    # Calculate median within whisker range
    median_value = df[(df[column] >= w1) & (df[column] <= w2)][column].median()
    
    # Replace outliers with median
    df.loc[(df[column] < w1) | (df[column] > w2), column] = median_value

# Apply the function to all columns in the DataFrame
def replace_outliers_all_columns(df):
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            replace_outliers_with_median(df, column)

# Example usage:
# Assuming 'df' is your DataFrame
replace_outliers_all_columns(df.iloc[:,2:])


# In[18]:


################ step 6:Data Transformation  ###########################


# In[7]:


df.info()


# In[8]:


df_cont=df.iloc[:,5:7]
df_cat=df.iloc[:,2:5]
df_cat.head()


# In[9]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
SS_X = scaler.fit_transform(df_cont)
SS_X = pd.DataFrame(SS_X)
SS_X.columns = df_cont.columns
SS_X.head()


# In[10]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_cat = df_cat.apply(le.fit_transform)
df_cat.head()


# In[11]:


X = pd.concat([SS_X, df_cat], axis=1)
X.head()


# In[12]:


Y=df['ATTORNEY']


# In[ ]:


###### step 7:data partition ##########


# In[ ]:


#now we have x and y and we do "data partition"


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[31]:


df.corr()


# In[ ]:


##### step 8: selecting models #######


# In[ ]:


#now i want to perform 4 models.to know the best model which gives good accuracy as it is a classification problem.the models 
#are 1)Decision trees 2)KNN 3)Logistic regression 4)svm


# In[ ]:


############(1)DECISION TREES#######################


# In[14]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy',max_depth=None)
dt.fit(X_train, Y_train)


# In[15]:


# prompt: predict the values on training and test data and calculate the accuracies for both

y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)

from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(Y_train, y_pred_train)
test_accuracy = accuracy_score(Y_test, y_pred_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


# In[16]:


# cross validation
#=========================================================

training_acc = []
test_acc = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3,random_state=i)
    dt.fit(X_train.values,Y_train)
    Y_pred_train = dt.predict(X_train.values)
    Y_pred_test  = dt.predict(X_test.values)
    training_acc.append(accuracy_score(Y_train,Y_pred_train))
    test_acc.append(accuracy_score(Y_test,Y_pred_test))

import numpy as np
print("Cross validation - Training accuracy:" ,np.mean(training_acc).round(2))
print("Cross validation - Test accuracy:" ,np.mean(test_acc).round(2))


# In[18]:


#if we provide none as a base estimator ,it takes decision tree classifier by default value
# cross validation with Bagging regressor
#=========================================================
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
training_err = []
test_err = []

for i in range(1,101):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.25,random_state=i)
    model = BaggingRegressor(estimator=None, max_samples=0.8,max_features=0.7)
    model.fit(X_train,Y_train)
    Y_pred_train = model.predict(X_train)
    Y_pred_test  = model.predict(X_test)
    training_err.append(mean_squared_error(Y_train,Y_pred_train,squared=False))
    test_err.append(mean_squared_error(Y_test,Y_pred_test,squared=False))

import numpy as np
print("Cross validation - Training error:" ,np.mean(training_err).round(2))
print("Cross validation - Test error:" ,np.mean(test_err).round(2))
print("bagging -variance :" ,(np.mean(test_err)-np.mean(training_err)).round(2))


# In[ ]:


######################## (2)KNN   ##################


# In[19]:


# cross validation with randomforest regressor
#=========================================================
from sklearn.ensemble import RandomForestRegressor

training_err = []
test_err = []

for i in range(1,101):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.25,random_state=i)
    model = RandomForestRegressor(max_samples=0.8,max_features=0.7)
    model.fit(X_train,Y_train)
    Y_pred_train = model.predict(X_train)
    Y_pred_test  = model.predict(X_test)
    training_err.append(mean_squared_error(Y_train,Y_pred_train,squared=False))
    test_err.append(mean_squared_error(Y_test,Y_pred_test,squared=False))

import numpy as np
print("Cross validation - Training error:" ,np.mean(training_err).round(2))
print("Cross validation - Test error:" ,np.mean(test_err).round(2))
print("RandomForestRegressor -variance :" ,(np.mean(test_err)-np.mean(training_err)).round(2))


# In[20]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=9)

model.fit(X_train.values,Y_train)
Y_pred_train = model.predict(X_train.values)
Y_pred_test  = model.predict(X_test.values)


# In[21]:


# cross validation with GradientBoosting
#=========================================================
from sklearn.ensemble import GradientBoostingRegressor

training_err = []
test_err = []

for i in range(1,101):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.25,random_state=i)
    model = GradientBoostingRegressor(learning_rate=0.1,n_estimators=100,max_features=0.7)
    model.fit(X_train,Y_train)
    Y_pred_train = model.predict(X_train)
    Y_pred_test  = model.predict(X_test)
    training_err.append(mean_squared_error(Y_train,Y_pred_train,squared=False))
    test_err.append(mean_squared_error(Y_test,Y_pred_test,squared=False))

import numpy as np
print("Cross validation - Training error:" ,np.mean(training_err).round(2))
print("Cross validation - Test error:" ,np.mean(test_err).round(2))
print("Gradient Boosting -variance :" ,(np.mean(test_err)-np.mean(training_err)).round(2))


# In[23]:


training_accuracy_list = []
test_accuracy_list = []

for k in range(5, 18, 2):
    training_accuracy = []
    test_accuracy = []

    for i in range(1, 100, 1):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=i)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train.values, Y_train)
        y_pred_train = knn.predict(X_train.values)
        y_pred_test = knn.predict(X_test.values)
        training_accuracy.append(accuracy_score(Y_train, y_pred_train))
        test_accuracy.append(accuracy_score(Y_test, y_pred_test))

    training_accuracy_list.append(np.mean(training_accuracy).round(2))
    test_accuracy_list.append(np.mean(test_accuracy).round(2))

print("Training Accuracies:", training_accuracy_list)
print("Test Accuracies:", test_accuracy_list)


# In[24]:


import matplotlib.pyplot as plt
plt.scatter(range(5, 18, 2),training_accuracy_list,color='blue')
plt.plot(range(5, 18, 2),training_accuracy_list,color='black')
plt.scatter(range(5, 18, 2),test_accuracy_list,color='red')
plt.plot(range(5, 18, 2),test_accuracy_list,color='black')
plt.show()


# In[ ]:


#############  3)Logistic Regression   ##############


# In[28]:


from sklearn.linear_model import LogisticRegression
Lr = LogisticRegression()
Lr.fit(X_train,Y_train)


# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assuming you already have your training data (X_train, Y_train) and testing data (X_test, Y_test)

# Create the Logistic Regression model
model = LogisticRegression()

# Train the model on training data
model.fit(X_train, Y_train)

# Make predictions on testing data
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(Y_test, y_pred)

# Print the accuracy score
print("Accuracy:", accuracy)


# In[ ]:


# in previous without doing any ensemble method,we got the accuracy as 68%.but after doing bagging classsifier we get it as 70%.
#demonstrating that 2% increase in accuracy.


# In[29]:


from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
#dt = DecisionTreeClassifier(criterion='gini')
bagging = BaggingClassifier(estimator=LogisticRegression(), n_estimators=300, max_samples=0.8, max_features=0.8,random_state=42)
bagging.fit(X_train, Y_train)

y_pred_train = bagging.predict(X_train)
y_pred_test = bagging.predict(X_test)

from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(Y_train, y_pred_train)
test_accuracy = accuracy_score(Y_test, y_pred_test)

print("Bagging - Training Accuracy:", train_accuracy)
print("Bagging - Testing Accuracy:", test_accuracy)


# In[30]:


Y_train_pred=Lr.predict(X_train)
Y_test_pred=Lr.predict(X_test)


# In[31]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y_train,Y_train_pred)
cm


# In[32]:


from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score


# In[33]:


ac1=accuracy_score(Y_train,Y_train_pred)
ac2=accuracy_score(Y_test,Y_test_pred)
print(ac1)
print(ac2)


# In[34]:


from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,dummy=roc_curve(Y_train,Lr.predict_proba(X_train)[:,1:])


# In[35]:


import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.ylabel('tpr-true positivr rate-sensitivity')
plt.xlabel('fpr-false positive rate-(1-specificity)')
plt.show()


# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Assuming you already have your training data X_train and Y_train

# Create the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, Y_train)

# Predict class probabilities (by default, LogisticRegression outputs probabilities)
predicted_proba = model.predict_proba(X_train)[:, 1:]  # Select probability for positive class

# Calculate ROC AUC score
rocvalue = roc_auc_score(Y_train, predicted_proba)

# Print the ROC AUC score
print(rocvalue)


# In[ ]:


# as area under score is 75,we can consider it as a good model


# In[37]:


Y_pred = model.predict(X)
from sklearn.metrics import log_loss
print("log loss:", log_loss(Y,Y_pred))


# In[ ]:


########   (4)SVM   ##############


# In[25]:


from sklearn.svm import SVC
#model = SVC(kernel='linear') 
#model = SVC(kernel='poly',degree=2) 
model = SVC(kernel='poly',degree=3)  #--> 91,90
#model = SVC(kernel='poly',degree=4)  #--> 52,47

#model = SVC(kernel='rbf',gamma='scale')  #--> 92,91


model.fit(X_train,Y_train)
Y_pred_train = model.predict(X_train)
Y_pred_test  = model.predict(X_test)


# In[26]:


# metrics
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy score:" ,round(ac1,3))
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy score:" , round(ac2,3))


# In[27]:


# cross validation
#=========================================================

training_acc = []
test_acc = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3,random_state=i)
    model.fit(X_train.values,Y_train)
    Y_pred_train = model.predict(X_train.values)
    Y_pred_test  = model.predict(X_test.values)
    training_acc.append(accuracy_score(Y_train,Y_pred_train))
    test_acc.append(accuracy_score(Y_test,Y_pred_test))

import numpy as np
print("Cross validation - Training accuracy:" ,np.mean(training_acc).round(2))
print("Cross validation - Test accuracy:" ,np.mean(test_acc).round(2))


# In[ ]:


#final output:

#   --Logistic Regression model is chosen as it produces high accuracy and low variance.

