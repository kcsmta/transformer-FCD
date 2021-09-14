from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

problem = "FLOW016"
max_length = 100 #(MNMX:2874, FLOW016: 1129, SUBINC: 1126, SUMTRIAN: 2799
vocab_size = 20000
n_class = 5

path_to_train = "./CodeChef_Data_ASM_Seq/"+problem+"_Seq_train.txt"
path_to_CV = "./CodeChef_Data_ASM_Seq/"+problem+"_Seq_CV.txt"
path_to_test = "./CodeChef_Data_ASM_Seq/"+problem+"_Seq_test.txt"

f = open(path_to_train, "r")
y_train = []
x_train = []
for line in f:
    y, x = line.split("\t\t")
    y_train.append(int(y))
    x_train.append(x)

t = Tokenizer(num_words=vocab_size)

t.fit_on_texts(x_train)

x_train = t.texts_to_sequences(x_train) # list
x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
x_train = np.asarray(x_train) # numpy array
y_train = np.asarray(y_train)

# print(type(x_train))
# print(x_train[0])
# print(x_train.shape) # (number of sentence, max_length)

f = open(path_to_CV, "r")
y_cv = []
x_cv = []
for line in f:
    y, x = line.split("\t\t")
    y_cv.append(int(y))
    x_cv.append(x)

t = Tokenizer()

t.fit_on_texts(x_cv)

x_cv = t.texts_to_sequences(x_cv) # list
x_cv = pad_sequences(x_cv, maxlen=max_length, padding='post')
x_cv = np.asarray(x_cv) # numpy array
y_cv = np.asarray(y_cv)


# print(type(x_cv))
# print(x_cv[0])
# print(x_cv.shape) # (number of sentence, max_length)

f = open(path_to_test, "r")
y_test = []
x_test = []
for line in f:
    y, x = line.split("\t\t")
    y_test.append(int(y))
    x_test.append(x)

t = Tokenizer()

t.fit_on_texts(x_test)

x_test = t.texts_to_sequences(x_test) # list
x_test = pad_sequences(x_test, maxlen=max_length, padding='post')
x_test = np.asarray(x_test) # numpy array
y_test = np.asarray(y_test)


# print(type(x_test))
# print(x_test[0])
# print(x_test.shape) # (number of sentence, max_length)

"""
Title: Text classification with Transformer
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/10
Last modified: 2020/05/10
Description: Implement a Transformer block as a Keras layer and use it for text classification.
"""
"""
## Setup
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


"""
## Implement a Transformer block as a layer
"""


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


"""
## Implement embedding layer

Two seperate embedding layers, one for tokens, one for token index (positions).
"""


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


"""
## Download and prepare dataset
"""

# vocab_size = 20000  # Only consider the top 20k words
# maxlen = 200  # Only consider the first 200 words of each movie review
# (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
# print(len(x_train), "Training sequences")
# print(len(x_val), "Validation sequences")
# x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

"""
## Create classifier model using transformer layer

Transformer layer outputs one vector for each time step of our input sequence.
Here, we take the mean across all time steps and
use a feed forward network on top of it to classify text.
"""


embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(max_length,))
embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
# 2 transformer block
x = transformer_block(x)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(n_class, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)


"""
## Train and Evaluate
"""
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=32, epochs=1, validation_data=(x_cv, y_cv), callbacks=[callback], verbose = 0
)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

print("-----------------------")
print("Results - ", problem)

results = model.predict(x_test)
y_pred = np.argmax(results, axis=1)
acc = 100*accuracy_score(y_test, y_pred)
print("Validation Accuracy = {:.3}%".format(acc))

test_f1 = f1_score(y_test, y_pred, average=None)
test_f1 = f1_score(y_test, y_pred, average='micro')
print("F1-micro = {:.3}%".format(test_f1))

test_f1 = f1_score(y_test, y_pred, average='macro')
print("F1-macro = {:.3}%".format(test_f1))

test_f1 = f1_score(y_test, y_pred, average='weighted')
print("F1-weighted = {:.3}%".format(test_f1))

# Compute ROC curve and ROC area for each class
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

y_test = to_categorical(y_test)

for i in range(n_class):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], results[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), results.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

lw=2
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_class):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_class

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

print("AUC micro = {:.3}%".format(roc_auc["micro"]))
print("AUC macro = {:.3}%".format(roc_auc["macro"]))

# # Plot all ROC curves
# plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)
#
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)
#
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_class), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))
#
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# plt.savefig("res.png")
