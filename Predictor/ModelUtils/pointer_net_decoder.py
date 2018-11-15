import torch as t
import torch.nn.functional as F
from .query_pooling import AttentionPooling
import ipdb


class PointerNetDecoder(t.nn.Module):
    def __init__(self, model_embedding_dim, hidden_size, dropout):
        super(PointerNetDecoder, self).__init__()
        self.rnn = t.nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=False, num_layers=1)
        self.start_attention_pooling = AttentionPooling(input_size=model_embedding_dim, hidden_size=hidden_size, dropout=dropout)

        self.start_passage_linear = t.nn.Linear(hidden_size, hidden_size)
        self.start_info_linear = t.nn.Linear(hidden_size, hidden_size)
        self.start_attention_linear = t.nn.Linear(hidden_size, 1)

        self.end_passage_linear = t.nn.Linear(hidden_size, hidden_size)
        self.end_info_linear = t.nn.Linear(hidden_size * 2 if self.rnn.bidirectional else hidden_size * 1, hidden_size)
        self.end_attention_linear = t.nn.Linear(hidden_size, 1)

        t.nn.init.xavier_normal_(self.start_passage_linear.weight)
        t.nn.init.xavier_normal_(self.start_info_linear.weight)
        t.nn.init.xavier_normal_(self.start_attention_linear.weight)

        t.nn.init.xavier_normal_(self.end_passage_linear.weight)
        t.nn.init.xavier_normal_(self.end_info_linear.weight)
        t.nn.init.xavier_normal_(self.end_attention_linear.weight)
        t.nn.init.orthogonal_(self.rnn.weight_hh_l0)
        t.nn.init.orthogonal_(self.rnn.weight_ih_l0)

        self.C = hidden_size ** 0.5

    def forward(self, passage, query, passage_mask=None, query_mask=None):
        """

        :param inputs: B, L, H
        :param side_info: B, 1, H
        :return:
        """
        start_info = self.start_attention_pooling(query, query_mask)
        # B, H
        net = self.start_attention_linear(F.tanh(self.start_passage_linear(passage) + self.start_info_linear(start_info).unsqueeze(1)))
        # B, L, 1
        start_logits = net.squeeze(-1)
        start_point = t.argmax(F.softmax(start_logits, -1), -1)

        answer_recurrent, _ = self.rnn(passage, start_info.repeat(self.rnn.num_layers, 1, 1))

        end_info = t.stack([i[0][i[1]] for i in zip(answer_recurrent, start_point)], 0)
        net = self.end_attention_linear(F.tanh(self.end_passage_linear(passage) + self.end_info_linear(end_info).unsqueeze(1)))
        end_logits = net.squeeze(-1)
        end_point = t.argmax(F.softmax(end_logits, -1), -1)

        return start_logits, end_logits, start_point, end_point


if __name__ == '__main__':
    passage = t.randn((32, 100, 64))
    query = t.randn((32, 100, 128))
    pointer = PointerNetDecoder(128, 64, 0.1)

    start_logits, end_logits, start_point, end_point = pointer(passage, query)

