import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
import csv

# 訓練データ/テストデータ読み込み
df_train = pd.read_csv('~/Devel/Kaggle/Titanic/train.csv')
df_test = pd.read_csv('~/Devel/Kaggle/Titanic/test.csv')

# Ageの欠損値を性別毎の年齢平均値で補完
age_mean_train = df_train.groupby('Sex').Age.mean()
df_train.Age.fillna(df_train[df_train.Age.isnull()].apply(lambda x: age_mean_train[x.Sex],axis=1), inplace=True)

# Ageの欠損値を性別毎の年齢平均値で補完
age_mean_test = df_test.groupby('Sex').Age.mean()
df_test.Age.fillna(df_test[df_test.Age.isnull()].apply(lambda x: age_mean_test[x.Sex],axis=1),inplace=True)

# Sex と Survived のクロス集計
# pd.crosstab(df_train['Sex'], df_train['Survived'])

# Pclass と Survived のクロス集計
# pd.crosstab(df_train['Pclass'], df_train['Survived'])

# ダミー変数化
df_train['Female'] = df_train['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
df_test['Female'] = df_test['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
pclass_df_train  = pd.get_dummies(df_train['Pclass'],prefix='Class')
pclass_df_test  = pd.get_dummies(df_test['Pclass'],prefix='Class')
df_train = df_train.join(pclass_df_train)
df_test = df_test.join(pclass_df_test)
 
# いらないカラムをドロップ。Xをモデルへの入力データとする。
X = df_train.drop(['PassengerId','Survived','Pclass','Name','Sex','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)

# 期待値
y = df_train.Survived

# モデルの生成
# Grid-Search でSVMのハイパーパラメータを求める
# tuned_params = [
#         {'C':[1,10,100,1000], 'kernel':['linear']},
#         {'C':[1,10,100,1000], 'gamma':[0.001, 0.0001], 'kernel':['rbf']},
#         ]
# clf = GridSearchCV(svm.SVC(C=1), tuned_params, n_jobs=-1, cv=5)
# clf.fit(X, y)
# best = clf.best_estimator_ # 最適なカーネル関数が返る
# print("Best is %(best)s" %locals())

#clf = svm.SVC(kernel='rbf',C=1)
clf = svm.SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
scores = cross_val_score(clf, X, y, cv=5, n_jobs=1)
print( "SVM: %(scores)s" %locals() )

# clf = RandomForestClassifier()
# scores = cross_val_score(clf, X, y, cv=5, n_jobs=1)
# print( "RandomForest: %(scores)s" %locals() )

# clf = neighbors.KNeighborsClassifier()
# scores = cross_val_score(clf, X, y, cv=5, n_jobs=1)
# print( "KNeighbors: %(scores)s" %locals() )

# clf = LogisticRegression()
# scores = cross_val_score(clf, X, y, cv=5, n_jobs=1)
# print( "Logistic: %(scores)s" %locals() )

# 学習実行
clf.fit(X, y)

# スコア
#clf.score(X,y)

# モデルへ入力するために、いらいないカラムをドロップ
df_test_in = df_test.drop(['PassengerId','Pclass','Name','Sex','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)

# 学習したモデルを用いて、テストデータの生存者を予測
test_predict = clf.predict(df_test_in)

# 結果をファイルに書き込み
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(df_test['PassengerId'], test_predict):
        writer.writerow([pid, survived])

