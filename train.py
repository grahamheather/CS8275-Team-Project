import scipy.io
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score
from collections import defaultdict

def train(file_name, variable_name):
	classes = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])

	feature_data = scipy.io.loadmat(file_name)[variable_name]

	num_patients = feature_data.shape[0]
	num_vowels = feature_data.shape[1]
	num_features = feature_data[0, 0].shape[1]

	# final metrics
	metrics = defaultdict(list)
	avg_metrics = defaultdict(int)

	# leave one out cross-validation
	for left_out in range(num_patients):
		print("PROGRESS", left_out)

		iter_data = np.array([]).reshape(0, num_features)
		test_data = np.array([]).reshape(0, num_features)
		iter_classes = np.array([]).reshape(0, 1)
		test_classes = np.array([]).reshape(0, 1)

		for i in range(num_patients):
			for vowel in range(num_vowels):
				if not i == left_out:
					iter_data = np.concatenate((iter_data, feature_data[i, vowel]))
					iter_classes = np.concatenate((iter_classes, np.full((feature_data[i, vowel].shape[0], 1), classes[i])))
				else:
					test_data = np.concatenate((test_data, feature_data[i, vowel]))
					test_classes = np.concatenate((test_classes, np.full((feature_data[i, vowel].shape[0], 1), classes[i])))

		# train classifier
		classifier = SVC(gamma='scale')
		classifier.fit(iter_data, iter_classes.ravel())
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

	return (metrics, avg_metrics)

metrics, avg_metrics = train('all_features', 'feature_data')
print(metrics)
print(avg_metrics)
