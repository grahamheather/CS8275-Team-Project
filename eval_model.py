import argparse
import pickle
import h5py
import numpy as np

def eval_model(model_filename, data_filename):
	model = pickle.load(open(model_filename, 'rb'))
	
	data_file = h5py.File(data_filename, 'r')
	feature_data = np.transpose(data_file['feature_data'])
	
	# remove infinite values (from dividing by 0)
	feature_data[~np.isfinite(feature_data)] = 0
	
	predicted = model.predict(feature_data)
	
	return predicted
	

if __name__ == "__main__":
	# arguments
	parser = argparse.ArgumentParser(description="predict data on SVM")
	parser.add_argument('model', help='file to load model from')
	parser.add_argument('data', help='matlab data')
	parser.add_argument('output_file', help='file for predicted classes')
	args = parser.parse_args()
	
	output = eval_model(args.model, args.data)
	
	with open(args.output_file, 'w') as out:
		for pred in output:
			out.write(str(int(pred)) + '\n')