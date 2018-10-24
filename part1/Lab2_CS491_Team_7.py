# Drafted by Sung Kim reviewd by Connor Filipovic

import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.loc[:,'ps_ind_01':'ps_calc_20_bin']
y_train = train.target

X_test = test.loc[:,'ps_ind_01':'ps_calc_20_bin']
y_test = test.target

clf = tree.DecisionTreeClassifier(max_depth=15, min_samples_split=20)
clf.fit(X_train, y_train)

train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)

scores = [score[0] for score in precision_recall_fscore_support(y_train, train_predict)]
conf_matrix = confusion_matrix(y_train, train_predict)
print('DT train\nP:{}\nR:{}\nF:{}\n{}'.format(scores[0],scores[1],scores[2],conf_matrix))

scores = [score[0] for score in precision_recall_fscore_support(y_test, test_predict)]
conf_matrix = confusion_matrix(y_test, test_predict)
print('DT test\nP:{}\nR:{}\nF:{}\n{}'.format(scores[0],scores[1],scores[2],conf_matrix))

clf2 = RandomForestClassifier(n_estimators=500)
clf2.fit(X_train, y_train)

train_predict2 = clf2.predict(X_train)
test_predict2 = clf2.predict(X_test)

scores = [score[0] for score in precision_recall_fscore_support(y_train, train_predict2)]
conf_matrix = confusion_matrix(y_train, train_predict2)
print('RF train\nP:{}\nR:{}\nF:{}\n{}'.format(scores[0],scores[1],scores[2],conf_matrix))

scores = [score[0] for score in precision_recall_fscore_support(y_test, test_predict2)]
conf_matrix = confusion_matrix(y_test, test_predict2)
print('RF test\nP:{}\nR:{}\nF:{}\n{}'.format(scores[0],scores[1],scores[2],conf_matrix))
