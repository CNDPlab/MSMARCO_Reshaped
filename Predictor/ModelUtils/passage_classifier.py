import torch as t
from .query_pooling import AttentionPooling\

class PassageClassifier(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(PassageClassifier, self).__init__()
        self.query_pooling = AttentionPooling(input_size=input_size, hidden_size=hidden_size, dropout=dropout)

    def forward(self, passage, query, passage_mask, query_mask):
        q_pooled = self.query_pooling(query, query_mask)
        # B, H
