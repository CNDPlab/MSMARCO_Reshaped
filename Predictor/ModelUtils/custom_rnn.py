import torch as t


class CustomRnn(t.nn.Module):
    def __init__(self, input_dim, hidden_size, bidirectional=True, type='GRU'):
        super(CustomRnn, self).__init__()
        assert type in ['RNN', 'GRU', 'LSTM']
        self.bidirectional = bidirectional
        self.rnn = getattr(t.nn, 'RNN')(input_dim, hidden_size, bidirectional=bidirectional, batch_first=True)
        if self.bidirectional:
            self.projection = t.nn.Linear(hidden_size * 2, hidden_size)
        t.nn.init.xavier_normal_(self.projection.weight)
        t.nn.init.orthogonal_(self.rnn.weight_hh_l0)
        t.nn.init.orthogonal_(self.rnn.weight_ih_l0)

    def forward(self, inputs):
        net, _ = self.rnn(inputs)
        if self.bidirectional:
            net = self.projection(net)
        return net


if __name__ == '__main__':
    inputs = t.randn((32, 100, 128))
    rnn = CustomRnn(128, 64)
    cc = rnn(inputs)
    print(cc.shape)

