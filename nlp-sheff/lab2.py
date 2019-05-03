# importing packages
import os
import collections, re
import string
import collections
import numpy as np
import argparse
import time

## main
parser = argparse.ArgumentParser()
parser.add_argument('corpus', type=str, help="corpus file")
parser.add_argument('questions', type=str, help="questions file")
args = parser.parse_args()
corpus = args.corpus if args.corpus else 'news-corpus-500k.txt'
questions = args.questions if args.questions else 'questions.txt'

comienzo = time.clock()

# read questions file, lowercase and split into tokens

with open(questions, "r") as file:  
    questions = []
    answers = []
    for line in file:
        line = line.lower()
        line = line.rstrip('\n')
        line = re.findall('([^:]+)', line)
        questions.append(line[0].split())        
        questions.append(line[0].split())
        answers.append(line[1].strip())
        
for i in range(len(answers)):
    answers[i] = answers[i].split('/')

# read corpus, lowercase and split into tokens, add symbol to begining and end of each sentence

with open(corpus, "r") as corpus:
    array = []
    for line in corpus:
        line = line.lower()
        line = line.rstrip('\n')
        line = re.sub('['+string.punctuation+']', '', line)
        line = line.split()
        array.append(line)

start = ['<s>']
end = ['</s>']

for i in range(len(array)):
    array[i] = start + array[i] + end


# create total corpus counts and calculate probability of unigrams

total_corpus = []

for i in range(len(array)):
    for g in range(len(array[i])):
        total_corpus.append(array[i][g])

unigram = collections.Counter(total_corpus)
del unigram['<s>']
del unigram['</s>']

totalsum = sum(unigram.values())

for key , value in unigram.items():
    unigram[key] = value/totalsum

## unigram model implementation

print(' ### unigram model probabilities ### ')
print("sum of total probabilities ", sum(unigram.values()))
for i in range(len(answers)):
    if unigram[answers[i][0]] > unigram[answers[i][1]]:
        print('word chosen' , answers[i][0] , unigram[answers[i][0]], unigram[answers[i][1]])
    elif unigram[answers[i][0]] < unigram[answers[i][1]]:
        print('word chosen', answers[i][1] , unigram[answers[i][1]], unigram[answers[i][0]] )
    else: 
        print ('equal probabilities, choose any')

## bigram model implementation

# function for creating bigrams
def ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

# create array with bigrams for each sentence in corpus

bigrams = []
for i in range(len(array)):
    bigrams.append(ngrams(array[i],2))
    
# create list with all the bigrams in corpus

total_bigrams = []

for i in range(len(bigrams)):
    for g in range(len(bigrams[i])):
        total_bigrams.append(bigrams[i][g])

# create dictionary with probabilities for each bigrams
        
bigrams_dict = collections.Counter(tuple(item) for item in total_bigrams)

totalsum_bi = sum(bigrams_dict.values())

for key , value in bigrams_dict.items():
    bigrams_dict[key] = value/totalsum_bi

# create list with words to try for each question

a_bi = []

for i in range(len(answers)):
    for f in range(2):
        a_bi.append(answers[i][f])

# tokenize each question, replace each word-option inside question

for i in range(len(questions)):
    for f in range(len(questions[i])):
        if questions[i][f] == '____':
            questions[i][f] = a_bi[i]

# remove punctuation 

for i in range(len(questions)):
    for f in range(len(questions[i])):
        questions[i][f] = re.sub('['+string.punctuation+']', '', questions[i][f])
    del questions[i][-1]

# add symbols to questions

start = ['<s>']
end = ['</s>']

for i in range(len(questions)):
    questions[i] = start + questions[i] + end

# create array with list of bigrams for each question
    
quest_bigrams = []

for i in range(len(questions)):
    quest_bigrams.append(ngrams(questions[i],2))

# replace bigrams with their probabilities

for i in range(len(quest_bigrams)):
    for j in range(len(quest_bigrams[i])):
        quest_bigrams[i][j] = bigrams_dict[quest_bigrams[i][j]]

# calculate the product of the bigram probabilities for each question

big_results = []

for i in range(len(quest_bigrams)):
    big_results.append(np.prod(quest_bigrams[i]))

# dictionary with possible words and their total sentence probabilities
    
new_res = {}

for i in range(len(a_bi)):
    new_res[a_bi[i]] = big_results[i]

# print chosen words with their sentence probability

print('### bigram model probabilities ###')
print("sum of total probabilities bigrams ", sum(bigrams_dict.values()))

for i in range(len(answers)):
    if new_res[answers[i][0]] > new_res[answers[i][1]]:
        print('word chosen' , answers[i][0] , new_res[answers[i][0]]*1000, new_res[answers[i][1]]*1000)
    elif new_res[answers[i][0]] < new_res[answers[i][1]]:
        print('word chosen', answers[i][1] , new_res[answers[i][1]]*1000, new_res[answers[i][0]]*1000 )
    else: 
        print ('equal probabilities, choose any')

## bigrams with add 1smoothing

total_bigrams = []

for i in range(len(bigrams)):
    for g in range(len(bigrams[i])):
        total_bigrams.append(bigrams[i][g])
        
bigrams_dict = collections.Counter(tuple(item) for item in total_bigrams)

q_bigrams = []
for i in range(len(questions)):
    q_bigrams.append(ngrams(questions[i],2))

# create list of total bigrams in questions (to see if they exist in corpus)  

bibi = []
for i in range(len(q_bigrams)):
    for j in range(len(q_bigrams[i])):
        bibi.append(q_bigrams[i][j])

# if bigrams is not in dictionary, add it with value 0

for i in range(len(bibi)):
    if bibi[i] not in bigrams_dict:
        bigrams_dict[bibi[i]] = 0

# add 1 to every value

for key, value in bigrams_dict.items():
    bigrams_dict[key] = value + 1

# calculate new total of frequencies
    
totalsum_bi = sum(bigrams_dict.values())

# calculate new probability for each bigram

for key , value in bigrams_dict.items():
    bigrams_dict[key] = value/(totalsum_bi)


# create array with list of bigrams for each question

quest_bigrams = []
for i in range(len(questions)):
    quest_bigrams.append(ngrams(questions[i],2))

# replace each bigram with its probabilty

for i in range(len(quest_bigrams)):
    for j in range(len(quest_bigrams[i])):
        quest_bigrams[i][j] = bigrams_dict[quest_bigrams[i][j]]

# calculate probability of each question

big_results_s = []
for i in range(len(quest_bigrams)):
    big_results_s.append(np.prod(quest_bigrams[i]))
    
# dictionary with possible words and their total sentence probabilities

new_res = {}
for i in range(len(a_bi)):
    new_res[a_bi[i]] = big_results_s[i]
    
# print chosen words with their sentence probability

print('### bigram with 1+ smoothing  probabilities ###')
print("sum of total probabilities with smoothing ", sum(bigrams_dict.values()))

for i in range(len(answers)):
    if new_res[answers[i][0]] > new_res[answers[i][1]]:
        print('word chosen' , answers[i][0] , new_res[answers[i][0]]*1000, new_res[answers[i][1]]*1000)
    elif new_res[answers[i][0]] < new_res[answers[i][1]]:
        print('word chosen', answers[i][1] , new_res[answers[i][1]]*1000, new_res[answers[i][0]]*1000 )
    else: 
        print ('equal probabilities, choose any')

final = time.clock()

print('tiempo' , final - comienzo)
