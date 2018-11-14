import torch as t
from Predictor.ModelUtils import MultiHeadDotAttention, MultiHeadSelfAttention, CharEncodeLayer, CustomRnn, PointerNetDecoder
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
        self.hidden_size = hidden_size
        if word_matrix is None and char_matrix is None:
            self.char_embedding_dim = 14
            self.word_embedding_dim = 300
            self.word_embedding = t.nn.Embedding(310000, self.word_embedding_dim)
            self.char_embedding = t.nn.Embedding(500, self.char_embedding_dim)
        else:
            self.word_embedding_dim = word_matrix.shape[1]
            self.char_embedding_dim = char_matrix.shape[1]
            self.word_embedding = t.nn.Embedding(word_matrix.shape[0], self.word_embedding_dim, padding_idx=0,
                                                 _weight=word_matrix)
            self.char_embedding = t.nn.Embedding(char_matrix.shape[0], self.char_embedding_dim, padding_idx=0)
            self.word_embedding.weight.requires_grad = False

        self.model_embedding_dim = self.hidden_size * 2 + self.char_embedding_dim * 3
        self.question_word_encoder = CustomRnn(self.word_embedding_dim, hidden_size, dropout, num_layers=2)
        self.question_char_encoder = CharEncodeLayer(self.char_embedding_dim, DefaultConfig.word_max_lenth)
        self.passage_word_encoder = CustomRnn(self.word_embedding_dim, hidden_size, dropout, num_layers=2)
        self.passage_char_encoder = CharEncodeLayer(self.char_embedding_dim, DefaultConfig.word_max_lenth)

        self.dot_attention = MultiHeadDotAttention(self.model_embedding_dim, hidden_size, hidden_size, dropout, num_head)
        self.self_attention = MultiHeadSelfAttention(hidden_size, hidden_size, hidden_size, dropout, num_head)

        self.span_decoder = PointerNetDecoder(model_embedding_dim=self.model_embedding_dim, hidden_size=hidden_size,
                                              dropout=dropout)
        self.distribution_decoder = None
        self.passage_classifier = None

    def forward(self, question_word, question_char, passage_word, passage_char):
        fake_batch_size, q_lenth = question_word.size()
        _, p_lenth = passage_word.size()
        batch_size = int(fake_batch_size / self.passage_num)

        q_mask = get_input_mask(question_word)
        p_mask = get_input_mask(passage_word)
        q_lens = q_mask.sum(-1)
        p_lens = p_mask.sum(-1)

        dot_attention_mask = get_da_mask(q_mask, p_mask)
        self_attention_mask = get_sa_mask(p_mask)
        q_w = self.word_embedding(question_word)
        p_w = self.word_embedding(passage_word)

        q_w, _ = self.question_word_encoder(q_w)
        q_w = q_w * q_mask.unsqueeze(-1)
        p_w, _ = self.passage_word_encoder(p_w)
        p_w = p_w * p_mask.unsqueeze(-1)

        q_c = self.char_embedding(question_char)
        p_c = self.char_embedding(passage_char)

        q_c = self.question_char_encoder(q_c)
        q_all = t.cat([q_w, q_c], dim=-1)

        p_c = self.passage_char_encoder(p_c)
        p_all = t.cat([p_w, p_c], dim=-1)

        net, _ = self.dot_attention(query=q_all, key=p_all, value=p_all, attention_mask=dot_attention_mask)
        net, _ = self.self_attention(query=net, key=net, value=net, attention_mask=self_attention_mask)

        query_info = q_all.view(batch_size, self.passage_num, q_lenth, self.model_embedding_dim)
        query_mask = q_mask.view(batch_size, self.passage_num, q_lenth)
        query_info = query_info[:, 0, :]
        query_mask = query_mask[:, 0]

        passage_info = net.view(batch_size, self.passage_num, p_lenth, self.hidden_size)
        passage_info = passage_info.view(batch_size, self.passage_num * p_lenth, self.hidden_size)
        passage_mask = p_mask.view(batch_size, self.passage_num, p_lenth)
        passage_mask = passage_mask.view(batch_size, self.passage_num * p_lenth)

        start_logits, end_logits, start_point, end_point = self.span_decoder(passage=passage_info, query=query_info, passage_mask=passage_mask, query_mask=query_mask)
        #start = start.masked_fill((1-passage_mask).byte(), -1e20)
        start = t.nn.functional.log_softmax(start_logits, -1)

        #end = end.masked_fill((1-passage_mask).byte(), -1e20)
        end = t.nn.functional.log_softmax(end_logits, -1)
        return start, end





if __name__ == '__main__':

    from tqdm import tqdm
    from Loaders import get_dataloader
    dataloader = get_dataloader('dev', 4, 4)
    for i in tqdm(dataloader):
        question_word, passage_word, question_char, passage_char, start, end, passage_index = [j for j in i]
        model = SNet(hidden_size=64, dropout=0.1, num_head=4)
        start, end = model(question_word, question_char, passage_word, passage_char)
        loss = (start + end).sum()
        loss.backward()
        ipdb.set_trace()