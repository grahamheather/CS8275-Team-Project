import scipy.io
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from collections import defaultdict
import h5py
from imblearn.over_sampling import SMOTE

to_exclude = {
	# patient, vowel, sample
	2: {0: [47]},
	8: {0: [4, 5]},
	10: {2: 'all'},
	16: {0: [20, 27, 29, 31, 36]},
	28: {2: 'all'},
	36: {0: [52], 2: [21]},
	38: {1: [37]},
	50: {0: [37], 2: [13]},
	51: {1: [25]},
	54: 'all',
	55: 'all',
	56: 'all',
	57: 'all'
}

def train(file_name, variable_name):
	classes = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])

	data_file = h5py.File(file_name, 'r')
	feature_data = data_file[variable_name]

	num_patients = feature_data.shape[1]
	num_vowels = feature_data.shape[0]
	num_features = data_file[feature_data[0, 0]].shape[0]

	# final metrics
	metrics = defaultdict(list)
	avg_metrics = defaultdict(int)


	excluded = 0
	# leave one out cross-validation
	for left_out in range(num_patients):
		if not left_out in to_exclude or to_exclude[left_out] != 'all':
			print("PROGRESS", left_out)

			iter_data = np.array([]).reshape(0, num_features)
			test_data = np.array([]).reshape(0, num_features)
			iter_classes = np.array([]).reshape(0, 1)
			test_classes = np.array([]).reshape(0, 1)

			for i in range(num_patients):
				if i not in to_exclude or to_exclude[i] != 'all':
					for vowel in range(num_vowels):
						if i not in to_exclude or vowel not in to_exclude[i] or to_exclude[i][vowel] != 'all':
							if i in to_exclude and vowel in to_exclude[i]:
								to_remove = to_exclude[i][vowel]
							else:
								to_remove = []

							current_data = np.delete(np.transpose(data_file[feature_data[vowel, i]]), to_remove, axis=0)
							current_classes = np.delete(np.full((data_file[feature_data[vowel, i]].shape[1], 1), classes[i]), to_remove, axis=0)
							if not i == left_out:
								iter_data = np.concatenate((iter_data, current_data))
								iter_classes = np.concatenate((iter_classes, current_classes))
							else:
								test_data = np.concatenate((test_data, current_data))
								test_classes = np.concatenate((test_classes, current_classes))

			X = iter_data
			y = iter_classes.ravel()

			# SMOTE
			rebalancing = SMOTE()
			#print(X.shape)
			#print(y.shape)
			#print(y)
			X_res, y_res = rebalancing.fit_resample(X, y)

			# train classifier
			classifier = SVC(gamma='scale')
			classifier.fit(X_res, y_res)
			predicted_classes = classifier.predict(test_data)

			# evaluate metrics
			precision = precision_score(test_classes, predicted_classes)
			recall = recall_score(test_classes, predicted_classes)
			accuracy = accuracy_score(test_classes, predicted_classes)
			
			# store metrics
			metrics['precision'].append(precision)
			metrics['recall'].append(recall)
			metrics['accuracy'].append(accuracy)
			avg_metrics['precision'] += precision / num_patients
			avg_metrics['recall'] += recall / num_patients
			avg_metrics['accuracy'] += accuracy / num_patients
		else:
			excluded = excluded + 1

	for metric in avg_metrics:
		avg_metrics[metric] = avg_metrics[metric] * num_patients / (num_patients - excluded)

	return (metrics, avg_metrics)

#metrics, avg_metrics = train('all_features', 'feature_data')
#metrics, avg_metrics = train('featuresV1.mat', 'feature_data')
metrics, avg_metrics = train('featuresV2.mat', 'feature_data')
print(metrics)
print(avg_metrics)
