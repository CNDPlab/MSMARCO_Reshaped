import torch as t
from Predictor.ModelUtils import MultiHeadDotAttention, MultiHeadSelfAttention, CharEncodeLayer, CustomRnn, \
    PointerNetDecoder, AttentionPooling
import ipdb
from configs import DefaultConfig
import torch.nn.functional as F


def get_input_mask(inputs):
    input_mask = inputs.data.ne(0).float()
    return input_mask


def get_sa_mask(input_mask):
    sa_mask = 1 - t.bmm(input_mask.unsqueeze(-1), input_mask.unsqueeze(-2)).byte()
    return sa_mask


def get_da_mask(query_mask, value_mask):
    da_mask = 1 - t.bmm(value_mask.unsqueeze(-1), query_mask.unsqueeze(-2)).byte()
    return da_mask


class SNet(t.nn.Module):
    def __init__(self, hidden_size, dropout, num_head, word_matrix=None, char_matrix=None, passage_num=11):
        super(SNet, self).__init__()
        self.passage_num = passage_num
        if word_matrix is None and char_matrix is None:
            self.char_embedding_dim = 14
            self.word_embedding_dim = 300
            self.word_embedding = t.nn.Embedding(310000, self.word_embedding_dim)
            self.char_embedding = t.nn.Embedding(500, self.char_embedding_dim)
            self.model_embedding_dim = self.char_embedding_dim + self.word_embedding_dim
        else:
            self.word_embedding_dim = word_matrix.shape[1]
            self.char_embedding_dim = char_matrix.shape[1]
            self.word_embedding = t.nn.Embedding(word_matrix.shape[0], self.word_embedding_dim)
            self.char_embedding = t.nn.Embedding(char_matrix.shape[0], self.char_embedding_dim)
            self.model_embedding_dim = self.word_embedding_dim + self.char_embedding_dim

        self.question_word_encoder = CustomRnn(self.model_embedding_dim, hidden_size)
        self.question_char_encoder = CharEncodeLayer(self.char_embedding_dim, DefaultConfig.word_max_lenth)
        self.passage_word_encoder = CustomRnn(self.model_embedding_dim, hidden_size)
        self.passage_char_encoder = CharEncodeLayer(self.char_embedding_dim, DefaultConfig.word_max_lenth)

        self.query_pooling = AttentionPooling(self.model_embedding_dim, hidden_size, dropout)

        self.dot_attention = MultiHeadDotAttention(self.model_embedding_dim, hidden_size, hidden_size, dropout, num_head)
        self.self_attention = MultiHeadSelfAttention(hidden_size, hidden_size, hidden_size, dropout, num_head)

        self.span_decoder = PointerNetDecoder(input_size=hidden_size, hidden_size=hidden_size)
        self.distribution_decoder = None
        self.passage_classifier = None

    def forward(self, question_word, question_char, passage_word, passage_char):

        ipdb.set_trace()
        q_mask = get_input_mask(question_word)
        p_mask = get_input_mask(passage_word)

        dot_attention_mask = get_da_mask(q_mask, p_mask)
        self_attention_mask = get_sa_mask(p_mask)

        q_w = self.word_embedding(question_word)
        p_w = self.word_embedding(passage_word)

        q_c = self.char_embedding(question_char)
        p_c = self.char_embedding(passage_char)

        q_c = self.question_char_encoder(q_c)
        q_all = t.cat([q_w, q_c], dim=-1)
        p_c = self.passage_char_encoder(p_c)
        p_all = t.cat([p_w, p_c], dim=-1)

        q_pooled = self.query_pooling(q_all, q_mask)

        net = self.dot_attention(query=q_all, key=p_all, value=p_all, attention_mask=dot_attention_mask)
        net = self.self_attention(query=net, key=net, value=net, attention_mask=self_attention_mask)

        start, end = self.span_decoder(net, q_pooled, passage_mask=p_mask)







if __name__ == '__main__':

    from Loaders import get_dataloader
    dataloader = get_dataloader('dev', 4, 4)
    for i in dataloader:
        question_word, passage_word, question_char, passage_char, start, end, passage_index = [t.from_numpy(j) for j in i]
        model = SNet(hidden_size=64, dropout=0.1, num_head=4)
        model(question_word, question_char, passage_word, passage_char)
