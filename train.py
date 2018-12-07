import scipy.io
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from collections import defaultdict
import h5py
from imblearn.over_sampling import SMOTE
import pickle

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

classes = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])

def separate_data(data_file, feature_data, to_exclude, left_out, num_patients, num_vowels, num_features):
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
						
	# remove infinite values (from dividing by 0)
	iter_data[~np.isfinite(iter_data)] = 0
	test_data[~np.isfinite(test_data)] = 0
	
	return iter_data, iter_classes, test_data, test_classes
	
	
def train_model(iter_data, iter_classes):
	X = iter_data
	y = iter_classes.ravel()

	# SMOTE
	rebalancing = SMOTE()
	X_res, y_res = rebalancing.fit_sample(X, y)
	
	# train classifier
	classifier = SVC()
	classifier.fit(X_res, y_res)
	
	return classifier
	

def train(file_name, variable_name):


	data_file = h5py.File(file_name, 'r')
	feature_data = data_file[variable_name]

	num_patients = feature_data.shape[1]
	num_vowels = feature_data.shape[0]
	num_features = data_file[feature_data[0, 0]].shape[0]

	# final metrics
	metrics = defaultdict(list)
	avg_metrics = defaultdict(int)

	# train and save full model
	iter_data, iter_classes, test_data, test_classes = separate_data(data_file, feature_data, to_exclude, -1, num_patients, num_vowels, num_features)
	classifier = train_model(iter_data, iter_classes)
	pickle.dump(classifier, open("model.pickle", 'wb'))

	num_trials = 0
	num_pos_trials = 0
	
	# leave one out cross-validation
	for left_out in range(num_patients):
		if not left_out in to_exclude or to_exclude[left_out] != 'all':
			print("PROGRESS", left_out)
			
			iter_data, iter_classes, test_data, test_classes = separate_data(data_file, feature_data, to_exclude, left_out, num_patients, num_vowels, num_features)
			classifier = train_model(iter_data, test_data)
			predicted_classes = classifier.predict(test_data)

			# evaluate metrics
			precision = precision_score(test_classes, predicted_classes)
			recall = recall_score(test_classes, predicted_classes)
			accuracy = accuracy_score(test_classes, predicted_classes)
			
			# store metrics
			metrics['precision'].append(precision)
			metrics['recall'].append(recall)
			metrics['accuracy'].append(accuracy)
			avg_metrics['precision'] += precision
			avg_metrics['recall'] += recall
			avg_metrics['accuracy'] += accuracy

			num_trials = num_trials + 1
			num_pos_trials = num_pos_trials + classes[left_out]

	avg_metrics['precision'] /= num_pos_trials
	avg_metrics['recall'] /= num_pos_trials
	avg_metrics['accuracy'] /= num_trials

	return (metrics, avg_metrics)

#metrics, avg_metrics = train('all_features', 'feature_data')
#metrics, avg_metrics = train('featuresV1.mat', 'feature_data')
#metrics, avg_metrics = train('featuresV2.mat', 'feature_data')
metrics, avg_metrics = train('featuresV4.mat', 'feature_data')
print(metrics)
print(avg_metrics)
