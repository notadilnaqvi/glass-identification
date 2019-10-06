import pandas
import time
st = time.time()
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import svm


all_data = pandas.read_csv('glass.csv')
all_features = all_data.drop(['Label'], axis=1)
all_lables = all_data['Label']


train_features, test_features, train_labels, test_labels = train_test_split(all_features, all_lables, stratify = all_lables, train_size = 0.80, random_state = 66)

clf = svm.SVC(C = 1, kernel = 'rbf', gamma = 'auto')
clf.fit(train_features, train_labels)
en = time.time()

print('Accuracy: ', clf.score(test_features, test_labels))
predictions = clf.predict(test_features)
print('F1 Score:', f1_score(test_labels, predictions, average='weighted'))
print('Runtime: ', en - st)









# Stuff
# from sklearn.metrics import f1_score, classification_report, confusion_matrix
# print(classification_report(test_labels, predictions))
# print(confusion_matrix(test_labels,predictions))
