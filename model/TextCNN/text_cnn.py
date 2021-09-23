from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout


class TextCNN(Model):

    def __init__(self,
                 kernel_sizes=[3, 4, 5],
                 class_num=1,
                 last_activation='softmax'):
        super(TextCNN, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num
        self.last_activation = last_activation
        self.convs = []
        self.max_poolings = []
        for kernel_size in self.kernel_sizes:
            self.convs.append(Conv1D(128, kernel_size, activation='relu'))
            self.max_poolings.append(GlobalMaxPooling1D())
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        # Embedding part can try multichannel as same as origin paper
        convs = []
        for i in range(len(self.kernel_sizes)):
            c = self.convs[i](inputs)
            c = self.max_poolings[i](c)
            convs.append(c)
        x = Concatenate()(convs)
        output = self.classifier(x)
        return output
