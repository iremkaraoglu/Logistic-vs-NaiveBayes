
# coding: utf-8

# In[1]:


# Loading necessary libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt  # for graphics
import seaborn as sns; sns.set()   # also for colored graphs, heat maps
from sklearn.datasets import fetch_20newsgroups  # load data from sklearn dataset
from sklearn.feature_extraction.text import TfidfVectorizer  # for coverting text to vectors
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline 
from sklearn.metrics import confusion_matrix #Creating confusion matrix and heat map
from sklearn.metrics import classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
data=fetch_20newsgroups()
folders = data.target_names   # list all categories


# In[2]:


categories=data.target_names
# Training the data on these categories
train=fetch_20newsgroups(subset='train', categories=categories)
# Testing the data for these categories
test=fetch_20newsgroups(subset='test', categories=categories)


# In[3]:


# See how many training examples
print(len(train.data))


# In[4]:


# See how many test examples
print(len(test.data))


# In[5]:


tenPercentTrainSet = train.data[0:1131] #Created new train set for 10 percent of the original test set.
tenPercentTrainTarget = train.target[0:1131]
tenPercentTrainTargetNames = train.target_names[0:1331]


# In[6]:


print(tenPercentTrainSet)


# In[7]:


print(test.data[0])


# In[8]:


print(tenPercentTrainSet[0])


# In[9]:


thirtyPercentTrainSet = train.data[0:3394] #Created new train set for 30 percent of the original test set.
thirtyPercentTrainTarget = train.target[0:3394]
thirtyPercentTrainTargetNames = train.target_names[0:3394]


# In[10]:


fiftyPercentTrainSet = train.data[0:5657] #Created new train set for 50 percent of the original test set.
fiftyPercentTrainTarget = train.target[0:5657]
fiftyPercentTrainTargetNames = train.target_names[0:5657]


# In[11]:


###########  NAIVE BAYES  ##############
###### MODEL WITH 10% TRAIN SET
#Creating a model based on Multinomial Naive Bayes
model=make_pipeline(TfidfVectorizer(), MultinomialNB())
# Training the model with the train data
model.fit(tenPercentTrainSet,tenPercentTrainTarget)
#creating labels for the test data
labels=model.predict(test.data)


# In[12]:


mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False
           , xticklabels=tenPercentTrainTargetNames
           , yticklabels=tenPercentTrainTargetNames)
plt.xlabel("True labels")
plt.ylabel("Predicted labels")


# In[13]:


print('accuracy %s' % accuracy_score(labels, test.target))
print(classification_report(test.target, labels,target_names=categories))


# In[14]:


###### MODEL WITH 30% TRAIN SET
#Creating a model based on Multinomial Naive Bayes
model=make_pipeline(TfidfVectorizer(), MultinomialNB())
# Training the model with the train data
model.fit(thirtyPercentTrainSet,thirtyPercentTrainTarget)
#creating labels for the test data
labels=model.predict(test.data)


# In[15]:


#Creating confusion matrix and heat map
mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False
           , xticklabels=thirtyPercentTrainTargetNames
           , yticklabels=thirtyPercentTrainTargetNames)
plt.xlabel("True labels")
plt.ylabel("Predicted labels")


# In[16]:


print('accuracy %s' % accuracy_score(labels, test.target))
print(classification_report(test.target, labels,target_names=categories))


# In[17]:


###### MODEL WITH 50% TRAIN SET
#Creating a model based on Multinomial Naive Bayes
model=make_pipeline(TfidfVectorizer(), MultinomialNB())
# Training the model with the train data
model.fit(fiftyPercentTrainSet,fiftyPercentTrainTarget)
#creating labels for the test data
labels=model.predict(test.data)


# In[18]:


#Creating confusion matrix and heat map
mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False
           , xticklabels=fiftyPercentTrainTargetNames
           , yticklabels=fiftyPercentTrainTargetNames)
plt.xlabel("True labels")
plt.ylabel("Predicted labels")


# In[19]:


print('accuracy %s' % accuracy_score(labels, test.target))
print(classification_report(test.target, labels,target_names=categories))


# In[20]:


#Creating a model based on Multinomial Naive Bayes
model=make_pipeline(TfidfVectorizer(), MultinomialNB())
# Training the model with the train data
model.fit(train.data,train.target)
#creating labels for the test data
labels=model.predict(test.data)


# In[21]:


mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False
           , xticklabels=train.target_names
           , yticklabels=train.target_names)
plt.xlabel("True labels")
plt.ylabel("Predicted labels")


# In[22]:


print('accuracy %s' % accuracy_score(labels, test.target))
print(classification_report(test.target, labels,target_names=categories))


# In[23]:


#####LOGISTIC REGRESSION######
#Creating a model based on Logistic Regression
model=make_pipeline(TfidfVectorizer(), LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42))
# Training the model with the train data
model.fit(tenPercentTrainSet, tenPercentTrainTarget)
#creating labels for the test data
labels=model.predict(test.data)


# In[24]:


mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False
           , xticklabels=tenPercentTrainTargetNames
           , yticklabels=tenPercentTrainTargetNames)
plt.xlabel("True labels")
plt.ylabel("Predicted labels")


# In[25]:


print('accuracy %s' % accuracy_score(labels, test.target))
print(classification_report(test.target, labels,target_names=categories))


# In[26]:


# Training the model with the train data
model.fit(thirtyPercentTrainSet, thirtyPercentTrainTarget)
#creating labels for the test data
labels=model.predict(test.data)


# In[27]:


mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False
           , xticklabels=thirtyPercentTrainTargetNames
           , yticklabels=thirtyPercentTrainTargetNames)
plt.xlabel("True labels")
plt.ylabel("Predicted labels")


# In[28]:


print('accuracy %s' % accuracy_score(labels, test.target))
print(classification_report(test.target, labels,target_names=categories))


# In[29]:


# Training the model with the train data
model.fit(fiftyPercentTrainSet, fiftyPercentTrainTarget)
#creating labels for the test data
labels=model.predict(test.data)


# In[30]:


mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False
           , xticklabels=fiftyPercentTrainTargetNames
           , yticklabels=fiftyPercentTrainTargetNames)
plt.xlabel("True labels")
plt.ylabel("Predicted labels")


# In[31]:


print('accuracy %s' % accuracy_score(labels, test.target))
print(classification_report(test.target, labels,target_names=categories))


# In[32]:


# Training the model with the train data
model.fit(train.data,train.target)
#creating labels for the test data
labels=model.predict(test.data)


# In[33]:


mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False
           , xticklabels=train.target_names
           , yticklabels=train.target_names)
plt.xlabel("True labels")
plt.ylabel("Predicted labels")


# In[34]:


print('accuracy %s' % accuracy_score(labels, test.target))
print(classification_report(test.target, labels,target_names=categories))


# In[9]:


#accuracy results with respect to percentage of used train set
import matplotlib.pyplot as plt
x = [10, 30, 50, 100]
y_nb = [0.245, 0.651, 0.734, 0.773] #naive bayes
plt.plot(x, y_nb,color='m')
y_logistic = [0.706, 0.794, 0.816, 0.843] #logistic regression
plt.plot(x, y)
plt.xlabel('percentage of used train set')
plt.ylabel('accuracy')
plt.show()
("Blue line represents logistic regression and pink one is naive bayes")

