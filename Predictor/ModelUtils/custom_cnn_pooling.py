import torch as t


class CustomCnnPooling(t.nn.Module):
    def __init__(self, input_dim, hidden_size, sequence_lenth, kernel_size=3, stride=1, padding=0, norm=True, relu=True):
        super(CustomCnnPooling, self).__init__()
        self.sequence_lenth = sequence_lenth
        self.cnn = t.nn.Conv1d(in_channels=input_dim, out_channels=hidden_size, stride=stride, padding=padding, kernel_size=kernel_size)
        self.act = t.nn.Sequential()
        if norm:
            self.act.add_module('norm', t.nn.BatchNorm1d(hidden_size))
        if relu:
            self.act.add_module('relu', t.nn.ReLU())
        self.max_pooling = t.nn.MaxPool1d(sequence_lenth - 2 * (padding + 1))
        t.nn.init.xavier_normal_(self.cnn.weight)

    def forward(self, inputs):
        """
        :param inputs: B, L, D
        :return: B, D
        """
        net = self.cnn(inputs.transpose_(-2, -1))
        net = self.act(net)
        net = self.max_pooling(net).transpose_(-2, -1).squeeze(-2)
        return net



if __name__ == '__main__':

    inputs = t.randn((32, 100, 128))
    cnn = CustomCnnPooling(128, 256, 100)
    c = cnn(inputs)
    print(c.shape)
