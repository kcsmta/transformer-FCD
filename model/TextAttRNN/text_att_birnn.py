from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from attention import Attention


class TextAttBiRNN(Model):
    def __init__(self,
                 class_num=1,
                 last_activation='softmax'):
        super(TextAttBiRNN, self).__init__()
        self.class_num = class_num
        self.last_activation = last_activation
        self.bi_rnn = Bidirectional(LSTM(128, return_sequences=True))  # LSTM or GRU
        self.attention = Attention(self.maxlen)
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        x = self.bi_rnn(inputs)
        x = self.attention(x)
        output = self.classifier(x)
        return output
