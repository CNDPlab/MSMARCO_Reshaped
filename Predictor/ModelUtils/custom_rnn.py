import torch as t
import ipdb
import torch.nn.functional as F


def reorder_sequence(sequence_emb, order, batch_first=True):
    """
    sequence_emb: [T, B, D] if not batch_first
    order: list of sequence length
    """
    batch_dim = 0 if batch_first else 1
    assert len(order) == sequence_emb.size()[batch_dim]

    order = t.LongTensor(order).to(sequence_emb.device)
    sorted_ = sequence_emb.index_select(index=order, dim=batch_dim)

    return sorted_

def reorder_lstm_states(lstm_states, order):
    """
    lstm_states: (H, C) of tensor [layer, batch, hidden]
    order: list of sequence length
    """
    assert isinstance(lstm_states, tuple)
    assert len(lstm_states) == 2
    assert lstm_states[0].size() == lstm_states[1].size()
    assert len(order) == lstm_states[0].size()[1]

    order = t.LongTensor(order).to(lstm_states[0].device)
    sorted_states = (lstm_states[0].index_select(index=order, dim=1),
                     lstm_states[1].index_select(index=order, dim=1))

    return sorted_states



def lstm_encoder(sequence, lstm,
                 seq_lens=None, init_states=None, embedding=None):
    """ functional LSTM encoder (sequence is [b, t]/[b, t, d],
    lstm should be rolled lstm)"""
    batch_size = sequence.size(0)
    if not lstm.batch_first:
        sequence = sequence.transpose(0, 1)
        emb_sequence = (embedding(sequence) if embedding is not None
                        else sequence)
    if seq_lens:
        assert batch_size == len(seq_lens)
        sort_ind = sorted(range(len(seq_lens)),
                          key=lambda i: seq_lens[i], reverse=True)
        seq_lens = [seq_lens[i] for i in sort_ind]
        emb_sequence = reorder_sequence(emb_sequence, sort_ind,
                                        lstm.batch_first)

    if init_states is None:
        device = sequence.device
        init_states = init_lstm_states(lstm, batch_size, device)
    else:
        init_states = (init_states[0].contiguous(),
                       init_states[1].contiguous())

    if seq_lens:
        packed_seq = t.nn.utils.rnn.pack_padded_sequence(emb_sequence,
                                                       seq_lens)
        packed_out, final_states = lstm(packed_seq, init_states)
        lstm_out, _ = t.nn.utils.rnn.pad_packed_sequence(packed_out)

        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(len(seq_lens))]
        lstm_out = reorder_sequence(lstm_out, reorder_ind, lstm.batch_first)
        final_states = reorder_lstm_states(final_states, reorder_ind)
    else:
        lstm_out, final_states = lstm(emb_sequence, init_states)

    return lstm_out, final_states


def init_lstm_states(lstm, batch_size, device):
    n_layer = lstm.num_layers*(2 if lstm.bidirectional else 1)
    n_hidden = lstm.hidden_size

    states = (t.zeros(n_layer, batch_size, n_hidden).to(device),
              t.zeros(n_layer, batch_size, n_hidden).to(device))
    return states


class CustomRnn(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers, bidirectional=True):
        super(CustomRnn, self).__init__()
        self.rnn = t.nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers,
                            bidirectional=bidirectional, dropout=dropout)

    def forward(self, inputs, lenths=None, init_states=None):
        batch_size = inputs.size(0)
        if lenths is not None:
            sort_ind = sorted(range(len(lenths)),
                              key=lambda i: lenths[i], reverse=True)
            seq_lens = [lenths[i] for i in sort_ind]
            emb_sequence = reorder_sequence(inputs, sort_ind, True)
        if init_states is None:
            device = inputs.device
            init_states = init_lstm_states(self.rnn, batch_size, device)
        else:
            init_states = (init_states[0].contiguous(),
                           init_states[1].contiguous())
        if lenths is not None:
            packed_seq = t.nn.utils.rnn.pack_padded_sequence(emb_sequence, seq_lens, batch_first=True)

            packed_out, final_states = self.rnn(packed_seq, init_states)
            lstm_out, _ = t.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            back_map = {ind: i for i, ind in enumerate(sort_ind)}
            reorder_ind = [back_map[i] for i in range(len(seq_lens))]
            lstm_out = reorder_sequence(lstm_out, reorder_ind, True)
            final_states = reorder_lstm_states(final_states, reorder_ind)
        else:
            lstm_out, final_states = self.rnn(inputs, init_states)
        return lstm_out, final_states


if __name__ == '__main__':

    inputs = t.Tensor([[1, 2, 3], [1, 0, 0]])
    print(inputs.shape)
    lenths = t.Tensor([3, 1])
    emb = t.nn.Embedding(5, 12)
    embedded = emb(inputs.long())
    print(embedded.shape)
    rnn = CustomRnn(12, 128, 0.1, 2)
    hs, ls = rnn(embedded, lenths)
    print(hs.shape)