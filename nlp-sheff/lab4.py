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
parser.add_argument('-v', action='store_true' , help='Viterbi')
parser.add_argument('-b', action='store_true' , help='Beam Search')
args = parser.parse_args()


train_file = args.train if args.train else 'train.txt'
test_file = args.test if args.test else 'test.txt'

viterbi = False	
beam = False

if args.v == True : viterbi = True 
if args.b == True : beam = True

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

## viterbi prediction

def vit(x, w, count):
    v = np.zeros((5, len(x)))
    for j in range(len(all_y)):
        for i in range(len(x)):
            if list(phi_1([x[i]], [all_y[j]], count).keys())[0] not in w:
                w[list(phi_1([x[i]], [all_y[j]], count).keys())[0]] = 0
            v[j][i] = list(phi_1([x[i]], [all_y[j]], count).values())[0] * w[list(phi_1([x[i]], [all_y[j]], count).keys())[0]] # fill Viterbi matrix with values
    pred = []
    for h in range(len(x)):
        z = (v[:,h])
        lab = all_y[v[:,h].argmax()] # get max from each column and get the label
        pred.append(lab)
    return pred

# add_weights to weight matrix for correct labels    

def add_weights(weights, x, y,):
    upd = phi_1(x, y, cw_cl_counts)
    for key, value in upd.items():
        weights[key] += value
    return weights

# substract weights to weight matrix for correct labels    

def sub_weights(weights, x, y):
    upd = phi_1(x, y, cw_cl_counts)
    for key, value in upd.items():
        weights[key] += -value
    return weights

 # perceptron train function

def train(x_train, y_train, weights, count, epoch=5):
	for j in range(epoch):
	    comienzo = time.clock()
	    random.seed(16*j)
	    random.shuffle(x_train)
	    random.seed(16*j)
	    random.shuffle(y_train)
	    for i in range(len(x_train)):
	        np.random.seed(i)
	        if viterbi == True : y_hat = vit(x_train[i], weights, count)
	       	if beam == True : y_hat = vit(x_train[i], weights, count)  
	        if y_train[i] != y_hat:
	            add_weights(weights, x_train[i], y_train[i])
	            sub_weights(weights, x_train[i], y_hat)
	    final = time.clock()
	    print('epoch', j+1 , "time", final - comienzo)
	return weights

# perceptron test function

def test(x_test, weights, count):
	y_predictions = []
	for i in range(len(x_test)):
		y_hat_test = vit(x_test[i], weights, count)
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
if viterbi == True : print("Using Viterbi algorithm")
if beam == True : print("Using beam search k=1 , equal to greedy search (viterbi)")
print("\n")
total_wt = train(x_train, y_train, total_wt, cw_cl_counts, epoch=5)
y_predictions = test(x_test, total_wt, cw_cl_counts)
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


end = time.clock()

print("total time", end - begin)

