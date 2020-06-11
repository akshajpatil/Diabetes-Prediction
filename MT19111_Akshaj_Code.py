
# coding: utf-8

# In[1]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


dataSet=pd.read_csv("E:/IIIT Delhi/DMG/Project/pima-indians-diabetes-database/diabetes_csv.csv")
dataSet.head()
dataSet.describe()


# In[2]:

print(dataSet.shape)

Attributes=dataSet.columns
#print(Attributes[8])

data=pd.read_csv("E:/IIIT Delhi/DMG/Project/pima-indians-diabetes-database/diabetes_csv.csv").to_dict(orient="row")

j=0
for i in data:
    if data[j]['class']=="tested_positive":
        data[j]['class']=1
    else:
        data[j]['class']=0
    j=j+1
    
#data[4]['class']=1
#for i in data:
 #   print(i['class'])

dfObj = pd.DataFrame(data, columns = Attributes)
#print(dfObj)
#export_csv = dfObj.to_csv ('E:/IIIT Delhi/DMG/Project/pima-indians-diabetes-database/PreprocessedDiabetes.csv', index = None, header=True) 
#dataSet1=pd.read_csv("E:/IIIT Delhi/DMG/Project/pima-indians-diabetes-database/PreprocessedDiabetes.csv")




# In[63]:

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import norm
#plt.matshow(dfObj.corr()['class'][:])
#plt.show()
corr=dfObj.corr()
f,ax=plt.subplots(figsize=(9,8))
sns.heatmap(corr, ax=ax, cmap="Accent_r", linewidths = 0.1,annot=True)
#flights = sns.load_dataset(dfObj)
#ax = sns.heatmap(dfObj)
plt.show()
#print(corr)
dfObj[dfObj.columns[1:]].corr()['class'][:]


# In[64]:

train=[]
test=[]
#print(dfObj.loc[0])
for row in range(len(dfObj)):
    if dfObj['Insuline'][row] == 0:
        #print(dfObj.loc[[row]])
        temp_list=[dfObj['Pregnency'][row],dfObj['Plasma Glucose'][row],dfObj['Blood Pressure'][row],dfObj['Skin Thickness'][row],dfObj['Insuline'][row],dfObj['BMI'][row],dfObj['Pedigree'][row],dfObj['age'][row],dfObj['class'][row]]
        test.append(temp_list)
    else:
        temp_list=[dfObj['Pregnency'][row],dfObj['Plasma Glucose'][row],dfObj['Blood Pressure'][row],dfObj['Skin Thickness'][row],dfObj['Insuline'][row],dfObj['BMI'][row],dfObj['Pedigree'][row],dfObj['age'][row],dfObj['class'][row]]
        train.append(temp_list)

        
dfObj1=pd.DataFrame(train, columns = Attributes)
dfObj2=pd.DataFrame(test, columns = Attributes)


X=dfObj1.loc[:,dfObj1.columns != 'Insuline']
Y=dfObj1.loc[:,dfObj1.columns == 'Insuline']

X1=dfObj2.loc[:,dfObj2.columns != 'Insuline']
Y1=dfObj2.loc[:,dfObj2.columns == 'Insuline']


from sklearn.neighbors import KNeighborsClassifier #predictions from skikit 
knn = KNeighborsClassifier(n_neighbors = 5) 
knn.fit(X, Y) 
predict = knn.predict(X1)


for rowno in range(len(test)):
    test[rowno][4]=predict[rowno]
    
#print(test[0],"\n",test[2])

#train.append(test)
train=train+test
#print(len(test),len(train))
dfObj1 = pd.DataFrame(train, columns = Attributes)
#print(train[766])
export_csv = dfObj1.to_csv ('E:/IIIT Delhi/DMG/Project/pima-indians-diabetes-database/PreprocessedDiabetes2.csv', index = None, header=True) 




# In[65]:

train1=[]
test1=[]
#print(dfObj.loc[0])
for row in range(len(dfObj1)):
    if dfObj1['Skin Thickness'][row] == 0:
        #print(dfObj.loc[[row]])
        temp_list=[dfObj1['Pregnency'][row],dfObj1['Plasma Glucose'][row],dfObj1['Blood Pressure'][row],dfObj1['Skin Thickness'][row],dfObj1['Insuline'][row],dfObj1['BMI'][row],dfObj1['Pedigree'][row],dfObj1['age'][row],dfObj1['class'][row]]
        test1.append(temp_list)
    else:
        temp_list=[dfObj1['Pregnency'][row],dfObj1['Plasma Glucose'][row],dfObj1['Blood Pressure'][row],dfObj1['Skin Thickness'][row],dfObj1['Insuline'][row],dfObj1['BMI'][row],dfObj1['Pedigree'][row],dfObj1['age'][row],dfObj1['class'][row]]
        train1.append(temp_list)

        
dfObj1=pd.DataFrame(train1, columns = Attributes)
dfObj2=pd.DataFrame(test1, columns = Attributes)


X=dfObj1.loc[:,dfObj1.columns != 'Skin Thickness']
Y=dfObj1.loc[:,dfObj1.columns == 'Skin Thickness']

X1=dfObj2.loc[:,dfObj2.columns != 'Skin Thickness']
Y1=dfObj2.loc[:,dfObj2.columns == 'Skin Thickness']


from sklearn.neighbors import KNeighborsClassifier #predictions from skikit 
knn = KNeighborsClassifier(n_neighbors = 5) 
knn.fit(X, Y) 
predict = knn.predict(X1)


for rowno in range(len(test1)):
    test1[rowno][3]=predict[rowno]
    
#print(test[0],"\n",test[2])

#train.append(test)
train1=train1+test1
#print(len(test),len(train))
dfObjFinal = pd.DataFrame(train1, columns = Attributes)
#print(train[766])
export_csv = dfObjFinal.to_csv ('E:/IIIT Delhi/DMG/Project/pima-indians-diabetes-database/PreprocessedDiabetes2.csv', index = None, header=True) 




# In[66]:

dfObjFinal=dfObjFinal.replace(0,np.NaN)

for i in dfObjFinal.columns:
    if i != 'class' and i != 'Pregnency':
        dfObjFinal[i].fillna(dfObjFinal[i].mean(),inplace=True)

#print(dfObjFinal.head())


# In[67]:

from sklearn.preprocessing import StandardScaler 
dfObjFinal=dfObjFinal.replace(np.NaN,0)
#print(dfObjFinal.head())
#scaler = StandardScaler() 
#scaler.fit(dfObjFinal) 
export_csv = dfObjFinal.to_csv ('E:/IIIT Delhi/DMG/Project/pima-indians-diabetes-database/PreprocessedDiabetes2.csv', index = None, header=True) 
import seaborn as sns
dfObj.isin([0]).sum()
sns.boxplot(x=dfObj['Plasma Glucose'])
plt.show()
sns.boxplot(x=dfObj['Pregnency'])
plt.show()
sns.boxplot(x=dfObj['Skin Thickness'])
plt.show()
sns.boxplot(x=dfObj['Blood Pressure'])
plt.show()
sns.boxplot(x=dfObj['Insuline'])
plt.show()
sns.boxplot(x=dfObj['BMI'])
plt.show()
sns.boxplot(x=dfObj['Pedigree'])
plt.show()
sns.boxplot(x=dfObj['age'])
plt.show()


# In[27]:

dfObjFinal.drop(dfObjFinal.index[dfObjFinal['Insuline'].idxmax()])
dfObjFinal.drop(dfObjFinal.index[dfObjFinal['Skin Thickness'].idxmax()])
corr=dfObjFinal.corr()
f,ax=plt.subplots(figsize=(9,8))
sns.heatmap(corr, ax=ax, cmap="Accent_r", linewidths = 0.1,annot=True)
#flights = sns.load_dataset(dfObj)
#ax = sns.heatmap(dfObj)
plt.show()
#print(corr)
dfObjFinal[dfObjFinal.columns[1:]].corr()['class'][:]


# In[28]:

#dfObjFinal.drop(dfObjFinal.index[dfObjFinal['Insuline'].idxmax()])
#dfObjFinal.drop(dfObjFinal.index[dfObjFinal['Skin Thickness'].idxmax()])


# In[68]:

#X_train, X_test, y_train, y_test = train_test_split(dfObj1, y, test_size=0.2)
from sklearn.cross_validation import train_test_split

X=dfObjFinal.loc[:,dfObjFinal.columns != 'class']
Y=dfObjFinal.loc[:,dfObjFinal.columns == 'class']

#print(X.head()," \n",Y.head())
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
print(X_train.shape," ",X_test.shape," ",y_train.size," ",y_test.size)

withoutmean=[]


# In[69]:

from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
  
# making predictions on the testing set 
y_pred = gnb.predict(X_test) 
  
# comparing actual response values (y_test) with predicted response values (y_pred) 
from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
withoutmean.append(metrics.accuracy_score(y_test, y_pred)*100)


# In[70]:

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(criterion="gini",max_depth=5)
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
print(metrics.accuracy_score(y_test,pred))
withoutmean.append(metrics.accuracy_score(y_test, pred)*100)


# In[71]:

from sklearn.neighbors import KNeighborsClassifier #predictions from skikit 
knn = KNeighborsClassifier(n_neighbors = 20) 
knn.fit(X_train,y_train) 
predict = knn.predict(X_test)
print(metrics.accuracy_score(y_test,predict))
withoutmean.append(metrics.accuracy_score(y_test, predict)*100)


# In[72]:

from sklearn.svm import SVC 
clfsv = SVC(kernel='linear') 
clfsv.fit(X_train, y_train)
predict=clfsv.predict(X_test)
print(metrics.accuracy_score(y_test,predict))
withoutmean.append(metrics.accuracy_score(y_test, predict)*100)


# In[73]:

dfObjmean=dfObj.replace(0,np.NaN)

for i in dfObjmean.columns:
    if i != 'class' and i != 'Pregnency':
        dfObjmean[i].fillna(dfObjmean[i].mean(),inplace=True)

#print(dfObjFinal.head())
dfObjmean=dfObjmean.replace(np.NaN,0)
dfObjmean.drop(dfObjmean.index[dfObjmean['Insuline'].idxmax()])
dfObjmean.drop(dfObjmean.index[dfObjmean['Skin Thickness'].idxmax()])

from sklearn.cross_validation import train_test_split

X1=dfObjmean.loc[:,dfObjmean.columns != 'class']
Y1=dfObjmean.loc[:,dfObjmean.columns == 'class']

#print(X.head()," \n",Y.head())
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size = 0.2)
print(X1_train.shape," ",X1_test.shape," ",y1_train.size," ",y1_test.size)

withmean=[]



# In[74]:

from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X1_train, y1_train) 
  
# making predictions on the testing set 
y1_pred = gnb.predict(X1_test) 
  
# comparing actual response values (y_test) with predicted response values (y_pred) 
from sklearn import metrics 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y1_test, y1_pred)*100)
withmean.append(metrics.accuracy_score(y1_test, y1_pred)*100)


# In[75]:

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(criterion="gini",max_depth=5)
clf.fit(X1_train,y1_train)
pred=clf.predict(X1_test)
print(metrics.accuracy_score(y1_test,pred))
withmean.append(metrics.accuracy_score(y1_test, pred)*100)


# In[76]:

from sklearn.neighbors import KNeighborsClassifier #predictions from skikit 
knn = KNeighborsClassifier(n_neighbors = 20) 
knn.fit(X1_train,y1_train) 
predict = knn.predict(X1_test)
print(metrics.accuracy_score(y1_test,predict))
withmean.append(metrics.accuracy_score(y1_test, predict)*100)


# In[77]:

from sklearn.svm import SVC 
clfsv = SVC(kernel='linear') 
clfsv.fit(X1_train, y1_train)
predict=clfsv.predict(X1_test)
print(metrics.accuracy_score(y1_test,predict))
withmean.append(metrics.accuracy_score(y1_test, predict)*100)


# In[78]:

print(withoutmean,"\n",withmean)


# In[79]:

import matplotlib.pyplot as plt
import numpy as np
my_xticks = ['GaussianNB','Decision Tree','KNN','SVM']
plt.xticks([1,2,3,4], my_xticks)
plt.plot([1,2,3,4], withoutmean,label="Missing Values Predicted")
plt.plot([1,2,3,4], withmean,label="Missing values Replaced with mean")
plt.legend()
plt.show()


# In[42]:

#dfObjFinal['class'].isin([0]).sum()
#dfObjFinal['class'].isin([1]).sum()


# In[53]:

#from sklearn.preprocessing import StandardScaler 
#scaler = StandardScaler() 
  
# To scale data 
#dfObjFinalSc=scaler.fit(dfObjFinal)
#print(dfObjFinal.head())


# In[ ]:



