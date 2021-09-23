import numpy as np
from matplotlib import pyplot as plt
import time
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from text_att_birnn import TextAttBiRNN

import sys
sys.path.append('../../')
import Data_IO

if len(sys.argv) !=3:
    print("run command as following:")
    print("python Transformer-FCD.py FLOW016 1.0")
    sys.exit()

prob = str(sys.argv[1])
data_part = float(sys.argv[2])

if prob not in ["FLOW016", "MNMX", "SUBINC", "SUMTRIAN"]:
    print("Dataset name {} is not valid".format(sys.argv[1]))
    sys.exit()

nb_classes = 5
epochs = 20

print('Loading data...')
file_wordvec = '../../asmdata/vec_embedding_no_ops.txt'

file_train = '../../CodeChef_Data_ASM_Seq/' + prob + '_Seq_train.txt'
file_CV = '../../CodeChef_Data_ASM_Seq/' + prob + '_Seq_CV.txt'
file_test = '../../CodeChef_Data_ASM_Seq/' + prob + '_Seq_test.txt'

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

print('Build model...')
model = TextAttBiRNN(maxlen=maxlen, class_num=nb_classes)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

print("Run training on {} of {}".format(data_part*100, prob))
start = time. time()
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
history = model.fit(X_train, y_train,
          batch_size=32,
          epochs=epochs,
          callbacks=[early_stopping],
          validation_data=(X_CV, y_CV))
end = time. time()
print("Training time: ", end - start)

print("-----------------------")
print("Results - ", prob)
from sklearn.metrics import accuracy_score, f1_score

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
