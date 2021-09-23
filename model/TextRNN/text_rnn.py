from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, LSTM


class TextRNN(Model):
    def __init__(self,
                 class_num=1,
                 last_activation='softmax'):
        super(TextRNN, self).__init__()
        self.class_num = class_num
        self.last_activation = last_activation
        self.rnn = LSTM(128)  # LSTM or GRU
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        x = self.rnn(inputs)
        output = self.classifier(x)
        return output
