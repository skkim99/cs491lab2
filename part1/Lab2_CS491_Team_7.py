# Drafted by Sung Kim reviewd by Connor Filipovic

import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

clf = tree.DecisionTreeClassifier(max_depth=15, min_samples_split=20)

X_train = train.loc[:,'ps_ind_01':'ps_calc_20_bin']
y_train = train.target
clf.fit(X_train, y_train)

X_test = test.loc[:,'ps_ind_01':'ps_calc_20_bin']
y_test = test.target

train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)

accuracy_score(y_train, train_predict)
accuracy_score(y_test, test_predict)

from sklearn.metrix import confusion_matrix
#confusion_matrix()