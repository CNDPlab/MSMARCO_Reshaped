import torch as t
from Predictor.ModelUtils.custom_cnn_pooling import CustomCnnPooling
import ipdb


class CharEncodeLayer(t.nn.Module):
    def __init__(self, input_dim, sequence_lenth):
        super(CharEncodeLayer, self).__init__()

        self.cnn_pooling = CustomCnnPooling(input_dim, input_dim, sequence_lenth)


    def forward(self, inputs):
        """

        :param inputs: Batch_size * 11, word_lenth, char_lenth, char_dim
        :return:
        """
        fake_batch_size, passage_lenth, word_max_lenth, char_embedding_dim = inputs.size()
        net = inputs.view(-1, word_max_lenth, char_embedding_dim)
        net = self.cnn_pooling(net)
        net = net.view(fake_batch_size, passage_lenth, char_embedding_dim * 3)
        return net


if __name__ == '__main__':
    inputs = t.randn((32, 100, 14, 18))
    char_encoder = CharEncodeLayer(18, 14)
    net = char_encoder(inputs)
    print(net.shape)
