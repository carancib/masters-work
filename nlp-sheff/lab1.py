import os
import argparse
from sklearn.metrics import precision_score, recall_score
import random
import collections
import re
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

## creates a bag of words from a text input, with options for added bigrams and trigrams)

def bow(text, bi=0, tri=0):
    bag = collections.Counter(re.findall(r'\w+', text))
    for sword in stopwords:
        del bag[sword]
    if bi == 1:
        bi_to_add = bigrams(text)
        bag = bag + bi_to_add
    if tri == 1:
        tri_to_add = trigrams(text)
        bag = bag + tri_to_add
    if bi == 1 & tri == 1:
        bi_to_add = bigrams(text)
        tri_to_add = trigrams(text)
        bag = bag + bi_to_add + tri_to_add
    else:
        bag = bag
    return bag

## creates a counter collection of bigrams in a text

def bigrams(text):
    bag = re.findall(r'\w+', text)
    return collections.Counter(zip(bag, bag[1:]))

## creates a counter collection of trigrams in a text

def trigrams(text):
    bag = re.findall(r'\w+', text)
    return collections.Counter(zip(bag, bag[1:], bag[2:]))

# predicts label for a text given a weight matrix

def prediction(text, weights, bi=0, tri=0):
    score = 0.0
    for word, counts in bow(text, bi=bi, tri=tri).items():
        score += counts * weights[word]
    if score >= 0.0:
        return (1)
    else:
        return (-1)

# updates weight matrix using values from the features

def update_weights(w, text, label, bi=0, tri=0):
    for word, counts in bow(text, bi=bi, tri=tri).items():
        w[word] += counts * label
    return (w)

# fits the perceptron to a train dataset, returning the weight matrix

def per_fit(train_set, weights, bi=0, tri=0):
    results = []
    for i in range(len(train_set)):
        y = prediction(train_set[i], weights, bi=bi, tri=tri)
        results.append(y)
        if y != train_labels[i]:
            update_weights(weights, train_set[i], train_labels[i], bi=bi, tri=tri)
    return (weights)

# predicts labels for a perceptron test set

def per_predict(test_set, weights, bi=0, tri=0):
    results_test = []
    for i in range(len(test_set)):
        y = prediction(test_set[i], weights, bi=bi, tri=tri)
        results_test.append(y)
    return (results_test)

## main

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str, help="directory for file")
args = parser.parse_args()
directory = args.directory if args.directory else 'review_polarity'

# read negative and positive reviews
# create lists for storing dataset

neg_total = []
pos_total = []

# set directory

neg_files = os.listdir('./' + directory + '/txt_sentoken/neg/')
pos_files = os.listdir('./' + directory + '/txt_sentoken/pos/')
neg_files.sort()
pos_files.sort()

# read and add to total list

for file in neg_files:
    with open(os.path.join('./' + directory + '/txt_sentoken/neg/', file), 'r') as review:
        text = review.read()
        neg_total.append(text)

for file in pos_files:
    with open(os.path.join('./' + directory + '/txt_sentoken/pos/', file), 'r') as review:
        text = review.read()
        pos_total.append(text)

# set first 800 for train in each class, and 200 for test, create labels

train_set = neg_total[0:800] + pos_total[0:800]
train_labels = [-1] * 800 + [1] * 800
test_set = neg_total[800:1000] + pos_total[800:1000]
test_labels = [-1] * 200 + [1] * 200
total = neg_total + pos_total

# create bag of words for train data and weight matrix

total_set = set()
total_set_bw = [collections.Counter(re.findall(r'\w+', txt)) for txt in total]

for i in range(len(total_set_bw)):
    total_set.update(total_set_bw[i])

weights = {}

for i in total_set:
    weights[i] = 0

# basic perceptron with ordered dataset

basic = per_fit(train_set, weights)
basic_results = per_predict(test_set, weights)
neg_results = [test_labels[i]+basic_results[i] for i in range(200)]
pos_results = [test_labels[i]+basic_results[i] for i in range(200,400)]
precision = precision_score(test_labels, basic_results)
recall = recall_score(test_labels, basic_results)
f1 = (2*precision*recall)/(precision+recall)
print("basic perceptron with ordered dataset", "precision", precision, "recall", recall,"f1", f1)

# reset weights

weights = {}

for i in total_set:
    weights[i] = 0

# perceptron with shuffled train data

random.seed(999)
random.shuffle(train_set)
random.seed(999)
random.shuffle(train_labels)

shuffle = per_fit(train_set, weights)
shuffle_results = per_predict(test_set, weights)

neg_results = [test_labels[i]+shuffle_results[i] for i in range(200)]
pos_results = [test_labels[i]+shuffle_results[i] for i in range(200,400)]
precision = precision_score(test_labels, shuffle_results)
recall = recall_score(test_labels, shuffle_results)
f1 = (2*precision*recall)/(precision+recall)
print("shuffled dataset perceptron", "precision", precision, "recall", recall, "f1", f1)

# reset weights

weights = {}

for i in total_set:
    weights[i] = 0

# perceptron with mutiple passes
# training perceptron 10 times

iteration = []
updates = []
precision = []
recall = []
f1 = []

for f in range(10):
    update = 0
    random.seed(11 * f)
    random.shuffle(train_set)
    random.seed(11 * f)
    random.shuffle(train_labels)
    multiple = per_fit(train_set,weights)
    print("multiple passes perceptron iteration", f, "seed", 100 * f)
    iteration.append(f)
    results_test = []
    multiple_results = per_predict(test_set, weights)
    p = precision_score(test_labels, multiple_results)
    r = recall_score(test_labels, multiple_results)
    f_one = (2*p*r)/(p+r)
    print("precision", p , "recall", r, "f1", f_one)
    precision.append(p)
    recall.append(r)
    f1.append(f_one)

# print top 10 features for each class

w = collections.Counter(weights)
print("most commont positive words", w.most_common(10))
print("most common negative words", w.most_common()[:-11:-1])

## implementing bigrams

# create bag of words for train data and weight matrix (to include bigrams)

total_set = set()
total_set_bw = [collections.Counter(re.findall(r'\w+', txt)) for txt in total]
total_bigrams = [bigrams(txt) for txt in total]
total_set_bw = total_set_bw + total_bigrams

for i in range(len(total_set_bw)):
    total_set.update(total_set_bw[i])

weights = {}

for i in total_set:
    weights[i] = 0

# fitting perceptron with shuffle

random.seed(999)
random.shuffle(train_set)
random.seed(999)
random.shuffle(train_labels)

shuffle = per_fit(train_set, weights, bi=1)
shuffle_results = per_predict(test_set, weights, bi=1)

neg_results = [test_labels[i] + shuffle_results[i] for i in range(200)]
pos_results = [test_labels[i] + shuffle_results[i] for i in range(200, 400)]
precision = precision_score(test_labels, shuffle_results)
recall = recall_score(test_labels, shuffle_results)
f1 = (2 * precision * recall) / (precision + recall)
print("peceptron with added bigrams", "precision", precision, "recall", recall, "f1", f1)

## implementing bigrams and trigrams

# create bag of words for train data and weight matrix (to include bigrams and trigrams)

total_set = set()
total_set_bw = [collections.Counter(re.findall(r'\w+', txt)) for txt in total]
total_bigrams = [bigrams(txt) for txt in total]
total_trigrams = [trigrams(txt) for txt in total]
total_set_bw = total_set_bw + total_bigrams + total_trigrams

for i in range(len(total_set_bw)):
    total_set.update(total_set_bw[i])

weights = {}

for i in total_set:
    weights[i] = 0

# fitting perceptron with shuffle

random.seed(999)
random.shuffle(train_set)
random.seed(999)
random.shuffle(train_labels)

shuffle = per_fit(train_set, weights, bi=1, tri=1)
shuffle_results = per_predict(test_set, weights, bi=1, tri=1)

neg_results = [test_labels[i] + shuffle_results[i] for i in range(200)]
pos_results = [test_labels[i] + shuffle_results[i] for i in range(200, 400)]
precision = precision_score(test_labels, shuffle_results)
recall = recall_score(test_labels, shuffle_results)
f1 = (2 * precision * recall) / (precision + recall)

print("perceptron with bigrams and trigrams", "precision", precision, "recall", recall, "f1", f1)

