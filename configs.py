import warnings


class DefaultConfig:
    raw_folder = 'Datas/Raw/'
    middle_folder = 'Datas/Middle/'
    processed_folder = 'Datas/Processed/'
    ckpt_root = 'Predictor/checkpoints/'
    gensim_file = 'Datas/glove_model.txt'
    word_vocab = 'Predictor/Tools/Vocabs/word_vocab.pkl'
    char_vocab = 'Predictor/Tools/Vocabs/char_vocab.pkl'

    model_name = 'SNet'

    num_epochs = 20
    passage_max_lenth = 120
    word_max_lenth = 14
    char_embedding_dim = 12
    word_embedding_dim = 300
    batch_size = 32
    hidden_size = 128
    dropout = 0.1
    num_head = 2

    warm_up_step = 4000
    eval_every_step = 2000
    save_every_step = 12000

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print('__', k, getattr(self, k))

