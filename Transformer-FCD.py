from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import Data_IO
import numpy as np
from keras import backend as K
from sklearn.metrics import accuracy_score, f1_score

prob = "FLOW016"
data_part = 0.25 # try 1.0, 0.75, 0.5, 0.25

nb_classes = 5
nb_epochs = 20

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

def positional_signal(hidden_size: int, length: int,
                      min_timescale: float = 1.0, max_timescale: float = 1e4):
    """
    Helper function, constructing basic positional encoding.
    The code is partially based on implementation from Tensor2Tensor library
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """

    if hidden_size % 2 != 0:
        raise ValueError(
            f"The hidden dimension of the model must be divisible by 2."
            f"Currently it is {hidden_size}")
    position = K.arange(0, length, dtype=K.floatx())
    num_timescales = hidden_size // 2
    log_timescale_increment = K.constant(
        (np.log(float(max_timescale) / float(min_timescale)) /
         (num_timescales - 1)),
        dtype=K.floatx())
    inv_timescales = (
            min_timescale *
            K.exp(K.arange(num_timescales, dtype=K.floatx()) *
                  -log_timescale_increment))
    scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)
    return K.expand_dims(signal, axis=0)

class AndPositionEmbedding(layers.Layer):
    """
        Injects positional encoding signal described in section 3.5 of the original
        paper "Attention is all you need". Also a base class for more complex
        coordinate encoding described in "Universal Transformers".
        """

    def __init__(self, min_timescale: float = 1.0,
                 max_timescale: float = 1.0e4, **kwargs):
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.signal = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['min_timescale'] = self.min_timescale
        config['max_timescale'] = self.max_timescale
        return config

    def build(self, input_shape):
        _, length, hidden_size = input_shape
        self.signal = positional_signal(
            hidden_size, length, self.min_timescale, self.max_timescale)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.signal


file_wordvec = './asmdata/vec_embedding_no_ops.txt'

file_train = './CodeChef_Data_ASM_Seq/' + prob + '_Seq_train.txt'
file_CV = './CodeChef_Data_ASM_Seq/' + prob + '_Seq_CV.txt'
file_test = './CodeChef_Data_ASM_Seq/' + prob + '_Seq_test.txt'

print('\nLoad training data: ' + file_train)
print('\nLoad CV data: ' + file_CV)
print('\nLoad test data: ' + file_test)

wordvec = Data_IO.loadWordEmbedding(file_wordvec)
vocab_size = len(wordvec)

y_train, X_train, maxlen_train = Data_IO.load_ASMSeqData(file_train, wordvec)
y_CV, X_CV, maxlen_CV = Data_IO.load_ASMSeqData(file_CV, wordvec)
y_test, X_test, maxlen_test = Data_IO.load_ASMSeqData(file_test, wordvec)

if nb_classes == 2:
    y_train = [x if x == 0 else 1 for x in y_train]
    y_CV = [x if x == 0 else 1 for x in y_CV]
    y_test = [x if x == 0 else 1 for x in y_test]

y_testnum = y_test

# maxlen: the length of the longest instruction sequence
maxlen = np.max([maxlen_train, maxlen_CV, maxlen_test])
if maxlen % 2 == 1:
    maxlen = maxlen + 1
print('max number of instructions: ' + str(maxlen))
# padding data
Data_IO.paddingASMSeq(X_train, maxlen)
Data_IO.paddingASMSeq(X_CV, maxlen)
Data_IO.paddingASMSeq(X_test, maxlen)

train_data = int(data_part*len(X_train))

X_train = np.array(X_train)
X_train = X_train[:train_data]
y_train = y_train[:train_data]
X_CV = np.array(X_CV)
X_test = np.array(X_test)

y_train = to_categorical(y_train, nb_classes)
y_CV = to_categorical(y_CV, nb_classes)
y_test = to_categorical(y_test, nb_classes)

embed_dim = X_train.shape[2]
print("embedding dimension:", embed_dim)

print(X_train.shape)
print(X_test.shape)
print(X_CV.shape)
print(y_train.shape)
print(y_test.shape)
print(y_CV.shape)

"""
## Create classifier model using transformer layer

Transformer layer outputs one vector for each time step of our input sequence.
Here, we take the mean across all time steps and
use a feed forward network on top of it to classify text.
"""

num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=((maxlen, embed_dim)), dtype='float32')
embedding_layer = AndPositionEmbedding()
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
# 2 transformer block
x = transformer_block(x)
# x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(nb_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

"""
## Train and Evaluate
"""
callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    X_train, y_train, batch_size=32, epochs=nb_epochs, validation_data=(X_CV, y_CV), callbacks=[callback], verbose = 1
)

print("-----------------------")
print("Results - ", prob)

results = model.predict(X_test)
y_pred = np.argmax(results, axis=1)
y_test = np.argmax(y_test, axis=1)
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
from scipy import interp

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

y_test = to_categorical(y_test)

for i in range(nb_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], results[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), results.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

lw=2
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(nb_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= nb_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

print("AUC micro = {:.3}%".format(roc_auc["micro"]))
print("AUC macro = {:.3}%".format(roc_auc["macro"]))

import matplotlib.pyplot as plt
from itertools import cycle
# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(nb_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig("res.png")
