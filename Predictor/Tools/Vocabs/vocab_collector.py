

class VocabCollector(object):

    def __init__(self, word_vocab, char_vocab):
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab

    def transfer(self, instance):
        for index, answer in instance['answers'].items():
            if answer['text'] != []:
                instance['answers'][index]['text'] = [self.word_vocab.from_token_id(word) for word in answer['text']]
                instance['answers'][index]['char'] = [[self.char_vocab.from_token_id(char) for char in word] for word in answer['char']]
        for index, passage in instance['passages'].items():
            if passage['text'] != []:
                instance['passages'][index]['text'] = [self.word_vocab.from_token_id(i) for i in passage['text']]
                instance['passages'][index]['char'] = [[self.char_vocab.from_token_id(char) for char in word] for word in passage['char']]

        instance['question']['text'] = [self.word_vocab.from_token_id(i) for i in instance['question']['text']]
        instance['passages']['char'] = [[self.char_vocab.from_token_id(char) for char in word] for word in instance['question']['char']]
        return instance

    def transfer_char(self, instance):
        return instance

    def transfer_pos(self, instance):
        return instance


