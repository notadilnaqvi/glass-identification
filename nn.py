import pandas
import time
st = time.time()
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


all_data = pandas.read_csv('glass.csv')
all_features = all_data.drop(['Label'], axis=1)
all_lables = all_data['Label']


train_features, test_features, train_labels, test_labels = train_test_split(all_features, all_lables, random_state = 93, stratify = all_lables, train_size = 0.80)

mlp = MLPClassifier(hidden_layer_sizes=(20), learning_rate_init=0.025,  activation='logistic', random_state = 93, max_iter = 1000)
mlp.fit(train_features, train_labels)
en = time.time()

print('Accuracy: ', mlp.score(test_features, test_labels))
predictions = mlp.predict(test_features)
print('F1 Score: ',f1_score(test_labels, predictions, average='weighted'))
print('Runtime: ', en - st)











# notes
# from sklearn.metrics import f1_score, classification_report, confusion_matrix
# print(classification_report(test_labels, predictions))
# print(confusion_matrix(test_labels,predictions)
