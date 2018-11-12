import torch as t
import torch.nn.functional as F
from .query_pooling import AttentionPooling
import ipdb


class PointerNetDecoder(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(PointerNetDecoder, self).__init__()
        self.rnn_cell = t.nn.GRUCell(hidden_size, hidden_size)

        self.start_attention_pooling = AttentionPooling(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
        self.end_attention_pooling = AttentionPooling(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)

        self.C = hidden_size ** 0.5

    def forward(self, passage, query, passage_mask, query_mask):
        """

        :param inputs: B, L, H
        :param side_info: B, 1, H
        :return:
        """
        q_pooled = self.start_attention_pooling(query, query_mask)
        net = t.bmm(passage, q_pooled.unsqueeze(-1)).squeeze(-1) / self.C
        start_logits = net
        start = F.softmax(net, -1)
        start_info = start.unsqueeze(-1) * passage
        start_info = self.end_attention_pooling(start_info, passage_mask)
        end_side_info = self.rnn_cell(start_info, q_pooled)
        net = t.bmm(passage, end_side_info.unsqueeze(-1)).squeeze(-1) / self.C
        end_logits = net

        return start_logits, end_logits