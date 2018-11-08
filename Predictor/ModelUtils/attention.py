import torch as t
import ipdb


class MultiHeadSelfAttention(t.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, num_head):
        super(MultiHeadSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.drop = t.nn.Dropout(dropout)

        self.reshape_key = t.nn.Linear(input_size, hidden_size * num_head, bias=False)
        self.reshape_query = t.nn.Linear(input_size, hidden_size * num_head, bias=False)
        self.reshape_value = t.nn.Linear(input_size, hidden_size * num_head, bias=False)

        self.self_attention = SelfAttention(hidden_size, dropout)
        self.projection = t.nn.Linear(num_head * hidden_size, output_size, True)
        t.nn.init.xavier_normal_(self.reshape_key.weight)
        t.nn.init.xavier_normal_(self.reshape_query.weight)
        t.nn.init.xavier_normal_(self.reshape_value.weight)
        t.nn.init.xavier_normal_(self.projection.weight)

    def forward(self, query, key, value, attention_mask):
        # B, seqlenth, H
        batch_size, key_lenth, _ = key.size()
        batch_size, query_lenth, _ = query.size()

        key_ = self.reshape_key(key).view(batch_size, key_lenth, self.num_head, self.hidden_size)
        query_ = self.reshape_query(query).view(batch_size, query_lenth, self.num_head, self.hidden_size)
        value_ = self.reshape_value(value).view(batch_size, key_lenth, self.num_head, self.hidden_size)

        key_ = key_.permute(2, 0, 1, 3).contiguous().view(-1, key_lenth, self.hidden_size)
        query_ = query_.permute(2, 0, 1, 3).contiguous().view(-1, query_lenth, self.hidden_size)
        value_ = value_.permute(2, 0, 1, 3).contiguous().view(-1, key_lenth, self.hidden_size)
        key_ = self.drop(key_)
        query_ = self.drop(query_)
        value_ = self.drop(value_)

        attention_mask = attention_mask.repeat(self.num_head, 1, 1)
        output, attention_matrix = self.self_attention(query_, key_, value_, attention_mask)
        output = output.view(self.num_head, batch_size, query_lenth, self.hidden_size)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, query_lenth, -1)
        output = self.projection(output)
        return output, attention_matrix


class MultiHeadDotAttention(t.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, num_head):
        super(MultiHeadDotAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.drop = t.nn.Dropout(dropout)

        self.reshape_key = t.nn.Linear(input_size, hidden_size * num_head, bias=False)
        self.reshape_query = t.nn.Linear(input_size, hidden_size * num_head, bias=False)
        self.reshape_value = t.nn.Linear(input_size, hidden_size * num_head, bias=False)

        self.self_attention = DotAttention(hidden_size, dropout)
        self.projection = t.nn.Linear(num_head * hidden_size, output_size, True)
        t.nn.init.xavier_normal_(self.reshape_key.weight)
        t.nn.init.xavier_normal_(self.reshape_query.weight)
        t.nn.init.xavier_normal_(self.reshape_value.weight)
        t.nn.init.xavier_normal_(self.projection.weight)

    def forward(self, query, key, value, attention_mask):
        # B, seqlenth, H
        batch_size, key_lenth, _ = key.size()
        batch_size, query_lenth, _ = query.size()

        key_ = self.reshape_key(key).view(batch_size, key_lenth, self.num_head, self.hidden_size)
        query_ = self.reshape_query(query).view(batch_size, query_lenth, self.num_head, self.hidden_size)
        value_ = self.reshape_value(value).view(batch_size, key_lenth, self.num_head, self.hidden_size)

        key_ = key_.permute(2, 0, 1, 3).contiguous().view(-1, key_lenth, self.hidden_size)
        query_ = query_.permute(2, 0, 1, 3).contiguous().view(-1, query_lenth, self.hidden_size)
        value_ = value_.permute(2, 0, 1, 3).contiguous().view(-1, key_lenth, self.hidden_size)
        key_ = self.drop(key_)
        query_ = self.drop(query_)
        value_ = self.drop(value_)

        attention_mask = attention_mask.repeat(self.num_head, 1, 1)
        output, attention_matrix = self.self_attention(query_, key_, value_, attention_mask)
        output = output.view(self.num_head, batch_size, key_lenth, self.hidden_size)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, key_lenth, -1)
        output = self.projection(output)
        return output, attention_matrix.view(batch_size, self.num_head, key_lenth, query_lenth)


class SelfAttention(t.nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SelfAttention, self).__init__()
        self.C = hidden_size ** 0.5
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, query, key, value, attention_mask):
        # B, seqlenth, H
        attention = t.bmm(query, key.transpose(1, 2)) / self.C
        attention = attention.masked_fill(attention_mask, -float('inf'))
        attention = t.nn.functional.softmax(attention, -1)
        attention = attention.masked_fill(t.isnan(attention), 0)
        attention = self.dropout(attention)
        output = t.bmm(attention, value)
        return output, attention


class DotAttention(t.nn.Module):
    def __init__(self, hidden_size, dropout):
        super(DotAttention, self).__init__()
        self.C = hidden_size ** 0.5
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, query, key, value, attention_mask):
        attention = t.bmm(key, query.transpose(1, 2)) / self.C
        attention = attention.masked_fill(attention_mask, -float('inf'))
        attention = t.nn.functional.softmax(attention, -1)
        attention = attention.masked_fill(t.isnan(attention), 0)
        attention = self.dropout(attention)
        output = t.bmm(attention, query)
        return output, attention