import json
import os

from collections import defaultdict
from random import shuffle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier

directory = '../dataset/reviews'
NUM_REVIEWS = 200
PERCENT_TRAIN = 0.9

def bag_of_words(raw_review):
	featurized_review = defaultdict(int)
	for word in raw_review['text'].split():
		featurized_review[word] += 1
	return featurized_review

original_reviews = []
featurized_reviews = []
labels = []
review_data = os.listdir(directory)
shuffle(review_data)
i = 0
while len(featurized_reviews) < NUM_REVIEWS:
	filename = directory + '/' + review_data[i]
	i += 1
	with open(filename) as review_file:
		for line in review_file:
			try:
				raw_review = json.loads(line)
				if 'Mexican' not in set(raw_review['categories']) and 'Chinese' not in set(raw_review['categories']) \
				or 'Mexican' in set(raw_review['categories']) and 'Chinese' in set(raw_review['categories']):
					continue
				# 'Chinese' => 0
				# 'Mexican' => 1
				label = 1 if 'Mexican' in set(raw_review['categories']) else 0
				featurized_review = bag_of_words(raw_review)
				featurized_reviews.append(featurized_review)
				labels.append(label)
				original_reviews.append(raw_review[text])
			except:
				continue
				# print('Unable to open review in', filename)
print(str(100.0 * sum(labels) / len(labels)) + '% Mexican')

num_train_reviews = int(PERCENT_TRAIN * len(featurized_reviews))

train_reviews = featurized_reviews[:num_train_reviews]
train_labels = labels[:num_train_reviews]

test_reviews = featurized_reviews[num_train_reviews+1:]
test_original_reviews = original_reviews[num_train_reviews+1:]
test_labels = labels[num_train_reviews+1:]

v = DictVectorizer(sparse=False)
X_train = v.fit_transform(train_reviews)
X_test = v.transform(test_reviews)

model = SGDClassifier()
model.fit(X_train, train_labels)
predictions = model.predict(X_test)

print('1', test_original_reviews)
print('2', predictions)
print('3', test_labels)

numCorrect = 0

for prediction, label in zip(predictions, test_labels):
	if prediction == label:
		numCorrect += 1
print('Accuracy:', float(numCorrect / len(predictions)))



















