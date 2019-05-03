import collections
import numpy as np
import itertools
import time
from sklearn.metrics import f1_score
import random
import argparse

# main

parser = argparse.ArgumentParser()
parser.add_argument('train', type=str, help="train file")
parser.add_argument('test', type=str, help="test file")
args = parser.parse_args()
train_file = args.train if args.train else 'train.txt'
test_file = args.test if args.test else 'test.txt'

begin = time.clock()

# loading data 

def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)

train_set = load_dataset_sents(train_file)
test = load_dataset_sents(test_file)

x_train = []
y_train = []

with open(train_file) as f:
    for line in f:
        sent, tags = line.split('\t')
        words = [w.strip() for w in sent.split()]
        x_train.append(words)
        ner_tags = [w.strip() for w in tags.split()]
        y_train.append(ner_tags)

x_test = []
y_test = []

with open(test_file) as f:
    for line in f:
        sent, tags = line.split('\t')
        words = [w.strip() for w in sent.split()]
        x_test.append(words)
        ner_tags = [w.strip() for w in tags.split()]
        y_test.append(ner_tags)        


# function total counts for word tag in train
 
def total_cwcl(sentence):    
    pairs = []
    for i in range(len(sentence)):
        for j in range(len(sentence[i])):
            pairs.append(str(sentence[i][j][0] + "_" + sentence[i][j][1]))
    cw_cl_counts = collections.Counter(pairs)
    cw_cl_counts = {k: v for k, v in cw_cl_counts.items()}
    return cw_cl_counts

cw_cl_counts = total_cwcl(train_set)

# input sentence and return counts for the keys present in counts dic for WORD TAG

def phi_1(x, y, cw_cl_counts):
    sent = list(zip(x, y))
    word_tags = []
    for i in range(len(sent)):
        word_tags.append(sent[i][0] + "_" + sent[i][1])
    counted = {}
    for x in word_tags:
        if x not in counted:
            counted[x] = 0
        if x in cw_cl_counts:
            counted[x] += 1
    return counted

# total counts of features TAG TAG

def total_tag_tag(train):
    pairs2 = []
    for i in range(len(train)):
        if len(train[i]) == 1:
            for j in range(len(train[i])):
                pairs2.append('None' + "_" + str(train[i][j][1]))
        else:
            for j in range(1, len(train[i])):
                if j-1 == 0 :
                    pairs2.append(str('None' + "_" + train[i][j-1][1]))
                pairs2.append(str(train[i][j-1][1] + "_" + train[i][j][1]))               
    tag_tag = collections.Counter(pairs2)
    return tag_tag

tag_tag = total_tag_tag(train_set)

# input sentence and return counts for the keys present in counts dic tag tag

def phi_2(y, tag_tag):
    counts = []
    final_count = {}
    counts.append("None" + "_" + y[0])
    for i in range(1, len(y)):
        counts.append(y[i-1] + "_" + y[i])
    counted = collections.Counter(counts)
    for key, value in counted.items():
        if key in tag_tag:
            final_count[key] = value
    return final_count

# merging phi_1 and phi_2

def phi_onetwo(x, y, cw_cl_counts, tag_tag):
    phi1 = phi_1(x, y, cw_cl_counts)
    phi2 = phi_2(y, tag_tag)
    phi_12 = {**phi1, **phi2}
    return phi_12

# create x set and y SET for weight matrix

x_list = []

for i in range(len(train_set)):
    for j in range(len(train_set[i])):
        x_list.append(train_set[i][j][0])

x_set = set(x_list)

y_list = []

for i in range(len(train_set)):
    for j in range(len(train_set[i])):
        y_list.append(train_set[i][j][1])
        
y_set = set(y_list)

        
all_y = ['O', 'LOC', 'ORG', 'MISC', 'PER']

# total combinations of word_tag (weights for phi1)

total_combinations_wt = list(itertools.product(x_set, y_set))

for i in range(len(total_combinations_wt)):
    total_combinations_wt[i] = total_combinations_wt[i][0] + "_" + total_combinations_wt[i][1]
    
total_wt = {}
for i in range(len(total_combinations_wt)):
    total_wt[total_combinations_wt[i]] = 0
    
# total combinations of tag_tag
    
total_combinations_tt = list(itertools.product(y_set, y_set))

for i in range(len(total_combinations_tt)):
    total_combinations_tt[i] = total_combinations_tt[i][0] + "_" + total_combinations_tt[i][1]
    
tc_tt = {}
for i in range(len(total_combinations_tt)):
    tc_tt[total_combinations_tt[i]] = 0

# weights dictionary for phi1 + phi2

tc = {**total_wt, **tc_tt}

# FUNCTIONS FOR PERCEPTRON

# get_combis, generates all posible combinations for X and labels and returns the feature representation (either phi1 or pi1+phi2)

def get_combis(x, features): # features can be 1 (phi1) or 2 (phi1+phi2)
    possible_y = itertools.product(all_y, repeat=len(x))
    ys = []
    for i in possible_y:
        ys.append(i)
    combis = []
    for i in range(len(ys)):
        if features == 1 :
            combis.append(phi_1(x, ys[i], cw_cl_counts))
        if features == 2 : 
            combis.append(phi_onetwo(x, ys[i], cw_cl_counts, tag_tag))
    return combis

# get_combis_values, returns the total sum of the weighted values for each combination

def get_combis_values(combis, weights):
    values = []
    for i in range(len(combis)):
        suma = []
        for key, value in combis[i].items():
            if key not in weights:
                weights[key] = 0
            suma.append(weights[key]*value)
        values.append(sum(suma))
    return values

# get_pred_labels, finds the argmax of the values and returns the label correspondant to that location (predicted labels)

def get_pred_label(combis_values, x):
    narray = np.array(combis_values)
    loc = [np.random.choice(np.flatnonzero(narray == narray.max()))]
    possible_y = itertools.product(all_y, repeat=len(x))
    ys = []
    for i in possible_y:
        ys.append(i)
    pred = ys[loc[0]]
    return pred

# predict function

def predict(x, weights, features):
    if features == 1 :
        combis = get_combis(x, features=1)
    if features == 2 : 
        combis = get_combis(x, features=2)
    combis_values = get_combis_values(combis, weights)
    labels = get_pred_label(combis_values, x)
    return labels

# add_weights to weight matrix for correct labels    

def add_weights(weights, x, y, features):
    if features == 1:
        upd = phi_1(x, y, cw_cl_counts)
    if features == 2:
        upd = phi_onetwo(x, y, cw_cl_counts, tag_tag)
    for key, value in upd.items():
        weights[key] += value
    return weights

# substract weights to weight matrix for correct labels    

def sub_weights(weights, x, y, features):
    if features == 1:
        upd = phi_1(x, y, cw_cl_counts)
    if features == 2:
        upd = phi_onetwo(x, y, cw_cl_counts, tag_tag)
    for key, value in upd.items():
        weights[key] += -value
    return weights

 # perceptron train function

def train(x_train, y_train, weights, features, epoch=5):
	for j in range(epoch):
	    comienzo = time.clock()
	    random.seed(16*j)
	    random.shuffle(x_train)
	    random.seed(16*j)
	    random.shuffle(y_train)
	    for i in range(len(x_train)):
	        np.random.seed(i)
	        y_hat = predict(x_train[i], weights, features=features)
	        if y_train[i] != y_hat:
	            add_weights(weights, x_train[i], y_train[i], features=features)
	            sub_weights(weights, x_train[i], y_hat, features=features)
	    final = time.clock()
	    print('epoch', j+1 , "time", final - comienzo)
	return weights

# perceptron test function

def test(x_test, weights, features):
	y_predictions = []
	for i in range(len(x_test)):
		y_hat_test = predict(x_test[i], weights, features=features)
		y_predictions.append(y_hat_test)
	return y_predictions

def score(real, predicted):
	flattened_test = [y for x in real for y in x]
	flattened_predictions = [y for x in predicted for y in x]
	f1_micro = f1_score(flattened_test, flattened_predictions, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
	return f1_micro

# train and test perceptron with WORD_TAG feature (phi1)

print("#############################################################")
print("Training perceptron with word_tag feature")
print("\n")
total_wt = train(x_train, y_train, total_wt,features=1,  epoch=5)
y_predictions = test(x_test, total_wt, features=1)
f1 = score(y_test, y_predictions)
print("\n")
print("F1 SCORE, word-tag feature", f1)
print("\n")

 ## printing top 10 for each label

def print_top(label, weights):
    label_tag = [label]
    tot = list(itertools.product(x_set, label_tag))

    for i in range(len(tot)):
        tot[i] = tot[i][0] + "_" + tot[i][1]
    total_label = {}
    for i in range(len(tot)):
        total_label[tot[i]] = 0

    for key,value in total_label.items():
        if key in weights:
            total_label[key] = weights[key]
        
    label_by_value = sorted(total_label.items(), key=lambda kv: (-kv[1], kv[0]), reverse=True)
    final = list(reversed(label_by_value[-10:]))
    print("top 10", label, "label", "sorted by value and name")
    print(*final, sep="\n")

print_top('ORG', total_wt)
print("\n")
print_top('MISC', total_wt)
print("\n")
print_top('PER', total_wt)
print("\n")
print_top('LOC', total_wt)
print("\n")
print_top('O', total_wt)

### training and test perceptron with word_tag and tag_tag features (phi_one + phi_two)

print("\n")
print("#############################################################")
print("Training perceptron with word_tag feature and tag_tag feature")
print("\n")
tc = train(x_train, y_train, tc, features=2, epoch=5)
y_predictions = test(x_test, tc, features=2)
f1 = score(y_test, y_predictions)
print("\n")
print("F1 SCORE, word-tag feature", f1)
print("\n")

 ## printing top 10 for each label

def print_top(label, weights):
    label_tag = [label]
    tot = list(itertools.product(x_set, label_tag))

    for i in range(len(tot)):
        tot[i] = tot[i][0] + "_" + tot[i][1]
    total_label = {}
    for i in range(len(tot)):
        total_label[tot[i]] = 0

    for key,value in total_label.items():
        if key in weights:
            total_label[key] = weights[key]
        
    label_by_value = sorted(total_label.items(), key=lambda kv: (-kv[1], kv[0]), reverse=True)
    final = list(reversed(label_by_value[-10:]))
    print("top 10", label, "label", "sorted by value and name")
    print(*final, sep="\n")

print_top('ORG', tc)
print("\n")
print_top('MISC', tc)
print("\n")
print_top('PER', tc)
print("\n")
print_top('LOC', tc)
print("\n")
print_top('O', tc)

end = time.clock()

print("total time", end - begin)

