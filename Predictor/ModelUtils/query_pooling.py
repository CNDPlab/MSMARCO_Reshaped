import torch as t
import ipdb


class AttentionPooling(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(AttentionPooling, self).__init__()
        self.projection1 = t.nn.Linear(input_size, hidden_size, bias=True)
        self.dropout = t.nn.Dropout(dropout)
        self.projection2 = t.nn.Linear(hidden_size, 1, bias=False)
        self.projection3 = t.nn.Linear(input_size, hidden_size)
        t.nn.init.xavier_normal_(self.projection1.weight)
        t.nn.init.xavier_normal_(self.projection2.weight)
        t.nn.init.xavier_normal_(self.projection3.weight)

    def forward(self, inputs, input_mask=None):
        """

        :param inputs: [B, L, E]
        :param input_mask: [B, L]
        :return: [B, E]
        """
        input_mask = input_mask.byte()
        net = t.nn.functional.tanh(self.projection1(inputs))
        # [B, L, H]
        net = self.projection2(net).squeeze(-1)
        # [B, L, 1]
        if input_mask is not None:
            net = net.masked_fill(1-input_mask, -float('inf'))
        net = t.nn.functional.softmax(net, -1).unsqueeze(-1)
        # [B, L, 1]
        net = inputs * net
        # [B, L, E]
        net = net.sum(-2)
        net = self.projection3(net)
        # [B, E]
        return net