import torch as t
import torch.nn.functional as F


class CustomCnnPooling(t.nn.Module):
    def __init__(self, input_dim, hidden_size, sequence_lenth):
        super(CustomCnnPooling, self).__init__()
        self.sequence_lenth = sequence_lenth
        self.cnns = t.nn.ModuleList([t.nn.Conv1d(input_dim, hidden_size, i) for i in range(3, 6)])

    def forward(self, inputs):
        """
        :param inputs: B, L, D
        :return: B, D
        """
        net = inputs.transpose(-2, -1)
        net = t.cat(
            [F.relu(cnn(net)).max(dim=2)[0] for cnn in self.cnns],
            dim=1
        )
        return net


if __name__ == '__main__':

    inputs = t.randn((32, 100, 128))
    cnn = CustomCnnPooling(128, 256, 100)
    c = cnn(inputs)
    print(c.shape)
