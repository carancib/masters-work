# -*- coding: utf-8 -*-
r"""
Word Embeddings: Encoding Lexical Semantics
===========================================

Word embeddings are dense vectors of real numbers, one per word in your
vocabulary. In NLP, it is almost always the case that your features are
words! But how should you represent a word in a computer? You could
store its ascii character representation, but that only tells you what
the word *is*, it doesn't say much about what it *means* (you might be
able to derive its part of speech from its affixes, or properties from
its capitalization, but not much). Even more, in what sense could you
combine these representations? We often want dense outputs from our
neural networks, where the inputs are :math:`|V|` dimensional, where
:math:`V` is our vocabulary, but often the outputs are only a few
dimensional (if we are only predicting a handful of labels, for
instance). How do we get from a massive dimensional space to a smaller
dimensional space?

How about instead of ascii representations, we use a one-hot encoding?
That is, we represent the word :math:`w` by

.. math::  \overbrace{\left[ 0, 0, \dots, 1, \dots, 0, 0 \right]}^\text{|V| elements}

where the 1 is in a location unique to :math:`w`. Any other word will
have a 1 in some other location, and a 0 everywhere else.

There is an enormous drawback to this representation, besides just how
huge it is. It basically treats all words as independent entities with
no relation to each other. What we really want is some notion of
*similarity* between words. Why? Let's see an example.

Suppose we are building a language model. Suppose we have seen the
sentences

* The mathematician ran to the store.
* The physicist ran to the store.
* The mathematician solved the open problem.

in our training data. Now suppose we get a new sentence never before
seen in our training data:

* The physicist solved the open problem.

Our language model might do OK on this sentence, but wouldn't it be much
better if we could use the following two facts:

* We have seen  mathematician and physicist in the same role in a sentence. Somehow they
  have a semantic relation.
* We have seen mathematician in the same role  in this new unseen sentence
  as we are now seeing physicist.

and then infer that physicist is actually a good fit in the new unseen
sentence? This is what we mean by a notion of similarity: we mean
*semantic similarity*, not simply having similar orthographic
representations. It is a technique to combat the sparsity of linguistic
data, by connecting the dots between what we have seen and what we
haven't. This example of course relies on a fundamental linguistic
assumption: that words appearing in similar contexts are related to each
other semantically. This is called the `distributional
hypothesis <https://en.wikipedia.org/wiki/Distributional_semantics>`__.


Getting Dense Word Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How can we solve this problem? That is, how could we actually encode
semantic similarity in words? Maybe we think up some semantic
attributes. For example, we see that both mathematicians and physicists
can run, so maybe we give these words a high score for the "is able to
run" semantic attribute. Think of some other attributes, and imagine
what you might score some common words on those attributes.

If each attribute is a dimension, then we might give each word a vector,
like this:

.. math::

    q_\text{mathematician} = \left[ \overbrace{2.3}^\text{can run},
   \overbrace{9.4}^\text{likes coffee}, \overbrace{-5.5}^\text{majored in Physics}, \dots \right]

.. math::

    q_\text{physicist} = \left[ \overbrace{2.5}^\text{can run},
   \overbrace{9.1}^\text{likes coffee}, \overbrace{6.4}^\text{majored in Physics}, \dots \right]

Then we can get a measure of similarity between these words by doing:

.. math::  \text{Similarity}(\text{physicist}, \text{mathematician}) = q_\text{physicist} \cdot q_\text{mathematician}

Although it is more common to normalize by the lengths:

.. math::

    \text{Similarity}(\text{physicist}, \text{mathematician}) = \frac{q_\text{physicist} \cdot q_\text{mathematician}}
   {\| q_\text{\physicist} \| \| q_\text{mathematician} \|} = \cos (\phi)

Where :math:`\phi` is the angle between the two vectors. That way,
extremely similar words (words whose embeddings point in the same
direction) will have similarity 1. Extremely dissimilar words should
have similarity -1.


You can think of the sparse one-hot vectors from the beginning of this
section as a special case of these new vectors we have defined, where
each word basically has similarity 0, and we gave each word some unique
semantic attribute. These new vectors are *dense*, which is to say their
entries are (typically) non-zero.

But these new vectors are a big pain: you could think of thousands of
different semantic attributes that might be relevant to determining
similarity, and how on earth would you set the values of the different
attributes? Central to the idea of deep learning is that the neural
network learns representations of the features, rather than requiring
the programmer to design them herself. So why not just let the word
embeddings be parameters in our model, and then be updated during
training? This is exactly what we will do. We will have some *latent
semantic attributes* that the network can, in principle, learn. Note
that the word embeddings will probably not be interpretable. That is,
although with our hand-crafted vectors above we can see that
mathematicians and physicists are similar in that they both like coffee,
if we allow a neural network to learn the embeddings and see that both
mathematicians and physicisits have a large value in the second
dimension, it is not clear what that means. They are similar in some
latent semantic dimension, but this probably has no interpretation to
us.


In summary, **word embeddings are a representation of the *semantics* of
a word, efficiently encoding semantic information that might be relevant
to the task at hand**. You can embed other things too: part of speech
tags, parse trees, anything! The idea of feature embeddings is central
to the field.


Word Embeddings in Pytorch
~~~~~~~~~~~~~~~~~~~~~~~~~~

Before we get to a worked example and an exercise, a few quick notes
about how to use embeddings in Pytorch and in deep learning programming
in general. Similar to how we defined a unique index for each word when
making one-hot vectors, we also need to define an index for each word
when using embeddings. These will be keys into a lookup table. That is,
embeddings are stored as a :math:`|V| \times D` matrix, where :math:`D`
is the dimensionality of the embeddings, such that the word assigned
index :math:`i` has its embedding stored in the :math:`i`'th row of the
matrix. In all of my code, the mapping from words to indices is a
dictionary named word\_to\_ix.

The module that allows you to use embeddings is torch.nn.Embedding,
which takes two arguments: the vocabulary size, and the dimensionality
of the embeddings.

To index into this table, you must use torch.LongTensor (since the
indices are integers, not floats).

"""

# Author: Robert Guthrie
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(123)

######################################################################
# An Example: N-Gram Language Modeling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Recall that in an n-gram language model, given a sequence of words
# :math:`w`, we want to compute
#
# .. math::  P(w_i | w_{i-1}, w_{i-2}, \dots, w_{i-n+1} )
#
# Where :math:`w_i` is the ith word of the sequence.
#
# In this example, we will compute the loss function on some training
# examples and update the parameters with backpropagation.
#

sentence0 = ['START','START','The', 'mathematician', 'ran' , '.','END', 'END']
sentence1 = ['START','START','The', 'mathematician', 'ran' , 'to', 'the' , 'store', '.','END', 'END']
sentence2 = ['START','START','The', 'physicist', 'ran' , 'to', 'the' , 'store', '.','END', 'END']
sentence3 = ['START','START','The', 'philosopher', 'thought' , 'about', 'it' , '.','END', 'END']
sentence4 = ['START','START','The', 'mathematician', 'solved' , 'the', 'open' , 'problem', '.','END', 'END']
sentences = [sentence0] + [sentence1] + [sentence2] + [sentence3] + [sentence4]


vocab = sorted(list(set(sentences[0] + sentences[1] + sentences[2] + sentences[3] + sentences[4])))
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {word_to_ix[word]: word for word in word_to_ix}

def get_trigrams(sentences):
    trigrams = []
    for i in range(len(sentences)):
        for j in range(len(sentences[i])- 2):
            trigrams.append(([sentences[i][j], sentences[i][j + 1]], sentences[i][j + 2]))
    return trigrams

def predict(context):
    context_idxs = list(map(lambda w: word_to_ix[w], context))
    context_var = autograd.Variable(
        torch.LongTensor(context_idxs))
    predict = model(context_var)
    index = (torch.max(predict, 1)[1]).data.tolist()[0]
    return ix_to_word[index]

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

CONTEXT_SIZE = 2
EMBEDDING_DIM = 5
losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.02)

print("trainable parameters", sum([param.nelement() for param in model.parameters()]))
print("model layers")
print(model)

for epoch in range(100):
    total_loss = torch.Tensor([0])
    for context, target in get_trigrams(sentences):

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_var)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)
print("final loss", losses[-1])


print('######################')
print('sanity check')
sanity = get_trigrams([sentence1])
counter = 0
for i in range(len(sanity)):
    context, word = sanity[i]
    print('real' , sanity[i])
    print('pred', (context, str(predict(context))))
    if sanity[i] == (context, str(predict(context))):
    	counter +=1
if counter == len(sanity):
	print('sanity check passed')
print('######################')


## getting logprobs for next word given context

def get_logprobs(context):
    context_idxs = list(map(lambda w: word_to_ix[w], context))
    context_var = autograd.Variable(
        torch.LongTensor(context_idxs))
    logprobs = model(context_var)
    index = (torch.max(logprobs, 1)[1]).data.tolist()[0]
    voc_prob = {}
    for i in range(len(vocab)):
    	voc_prob[vocab[i]] = logprobs[0][i]
    return voc_prob

# logprobabilities for "The philosopher solved the open problem"

phil_0 = get_logprobs(['START', 'START'])['The']
phil_1 = get_logprobs(['START', 'The'])['philosopher']
phil_2 = get_logprobs(['The', 'philosopher'])['solved']
phil_3 = get_logprobs(['philosopher', 'solved'])['the']


philosopher = phil_0 + phil_1 + phil_2 + phil_3

# logprobabilities for "The physicist solved the open problem"

phys_0 = get_logprobs(['START', 'START'])['The']
phys_1 = get_logprobs(['START', 'The'])['physicist']
phys_2 = get_logprobs(['The', 'physicist'])['solved']
phys_3 = get_logprobs(['physicist', 'solved'])['the']

physicist = phys_0 + phys_1 + phys_2 + phys_3


print("prob of philosopher", np.exp(philosopher.detach().numpy()))
print("prob of physicist", np.exp(physicist.detach().numpy()))

if np.exp(philosopher.detach().numpy()) < np.exp(physicist.detach().numpy()):
	print("physicist is more likely to fill the gap")

if np.exp(philosopher.detach().numpy()) > np.exp(physicist.detach().numpy()):
	print("philosopher is more likely to fill the gap")


### getting cosine similarity

math = model.embeddings(torch.LongTensor([word_to_ix["mathematician"]]))
phys = model.embeddings(torch.LongTensor([word_to_ix["physicist"]]))
phil = model.embeddings(torch.LongTensor([word_to_ix["philosopher"]]))
cos = nn.CosineSimilarity()
print("cosine similarity for mathematician and physicist", cos(math, phys))
print("cosine similarity for mathematician and philosopher", cos(math, phil))

if cos(math, phys) > cos(math, phil):
	print("physiscist is more similar to mathematician")
else:
	print("philosopher is more similar to mathematician")


