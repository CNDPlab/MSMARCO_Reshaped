class DefaultConfig:
    raw_folder = 'Datas/Raw/'
    middle_folder = 'Datas/Middle/'
    processed_folder = 'Datas/Processed/'
    ckpt_root = 'Predictor/checkpoints/'
    gensim_file = 'Datas/glove_model.txt'
    word_vocab = 'Predictor/Tools/Vocabs/word_vocab.pkl'
    char_vocab = 'Predictor/Tools/Vocabs/char_vocab.pkl'

    passage_max_lenth = 120
    word_max_lenth = 14
    char_embedding_dim = 12
    word_embedding_dim = 300

