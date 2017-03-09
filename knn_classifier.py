"""
	Train & test using knn classifier 
	I am using sklearn implementation of knn algorithm along with feature extraction 
	techniques provided by sklearn
"""

import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from pandas import DataFrame


def read_data_to_matrix(file_name):
	"""
		Read data from file and return a pandas dataframe. 
		Pandas data frame is much more easier to slice & dice. 
	"""
	fp = open(file_name,'r')
	csv_reader = csv.reader(fp)

	file_records = []
	for line in csv_reader:
		if len(line) > 1:
			file_records.append({'text':line[0], 'class':line[1]})

	data_frame = DataFrame(file_records)
	return data_frame

def get_test_data():
	return csv.reader(open("test_data.txt",'r')).read()

if __name__ == "__main__":

	file_records = read_data_to_matrix('training_dataset.csv')
	print(file_records)

	# Get Frequency Count
	count_vect = CountVectorizer(stop_words="english",decode_error='ignore')
	train_counts = count_vect.fit_transform(file_records['text'].values)

	neigh = KNeighborsClassifier(n_neighbors = 5)
	targets = file_records['class'].values
	neigh.fit(train_counts,targets)

	test_data = get_test_data()
	test_counts = count_vect.transform(test_data)
	predictions = neigh.predict(test_counts)
	predict_matrix = []
	for i in predictions:
		if i == 'non-tech':
			i = 0
		else:
			i = 1
		predict_matrix.append(i)
	print(predict_matrix)

	actual_class = [0,0,0]
	print("f1: %f\n" %f1_score(actual_class,predict_matrix,average = None))
	print("accuracy: %f\n" %accuracy_score(actual_class,predict_matrix))
	




