
# coding: utf-8

# In[97]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier


# In[98]:


# データを読み込む
dataPath = '~/desk/kaggle/ktProj/data/'
df = pd.read_csv(dataPath + "train.csv").replace("male",0).replace("female",1)

#欠損値を補完
#df["Age"].fillna(df.Age.median(), inplace=True)
                                                                   


# In[99]:


### 客室クラス(1,2,3のどれか)毎の生死の割合をヒストグラムで見てみる

split_data = []
for survived in [0,1]:
    split_data.append(df[df.Survived==survived])

temp = [i["Pclass"].dropna() for i in split_data]
plt.hist(temp, histtype="barstacked", bins=3, label=('dead', 'alive'), color=('r', 'b'))


# In[100]:


### 年齢毎の生死のヒストグラムで見てみる

temp = [i["Age"].dropna() for i in split_data]
plt.hist(temp, histtype="barstacked", bins=16, color=('r', 'b'))


# In[101]:


### 家族数を属性に追加する

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df2 = df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)


# In[102]:


### 家族数と生死に相関があるか見てみる

X_fSize = list(set(df2["FamilySize"]))
y_fSize = []

for fsize in X_fSize:
    tmp = df2[df2.FamilySize==fsize]
    y_fSize.append(len(tmp[tmp.Survived==1])/len(tmp))

plt.plot(X_fSize, y_fSize, 'o')




# In[103]:



# 学習データを作成(欠損値を含む行は削除)
train_data = df2.values[~np.isnan(df2.values).any(axis=1)]

x_train = train_data[:, 2:]
y_train = train_data[:, 1]


# In[104]:


forest = RandomForestClassifier(n_estimators = 100)

# 学習
forest = forest.fit(x_train, y_train)

test_df= pd.read_csv(dataPath + "test.csv").replace("male",0).replace("female",1)
# 欠損値の補完
test_df["Age"].fillna(df.Age.median(), inplace=True)
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
test_df2 = test_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)


# In[105]:


test_data = test_df2.values
xs_test = test_data[:, 1:]
output = forest.predict(xs_test)

print(len(test_data[:,0]), len(output))
zip_data = zip(test_data[:,0].astype(int), output.astype(int))
predict_data = list(zip_data)


# In[106]:


import csv
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])

