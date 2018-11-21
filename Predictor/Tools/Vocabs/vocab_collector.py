

class VocabCollector(object):

    def __init__(self, word_vocab, char_vocab):
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab

    def transfer(self, instance):
        for index, answer in instance['answers'].items():
            if answer['text'] != []:
                instance['answers'][index]['text'] = self.word_vocab.convert(answer['text'], 't2i')
                instance['answers'][index]['char'] = self.char_vocab.convert(answer['text'], 'lt2i')
        for index, passage in instance['passages'].items():
            if passage['text'] != []:
                instance['passages'][index]['text'] = self.word_vocab.convert(passage['text'], 't2i')
                instance['passages'][index]['char'] = self.char_vocab.convert(passage['text'], 'lt2i')

        instance['question']['text'] = self.word_vocab.convert(instance['question']['text'], 't2i')
        instance['question']['char'] = self.char_vocab.convert(instance['question']['text'], 'lt2i')
        return instance

    def transfer_char(self, instance):
        return instance

    def transfer_pos(self, instance):
        return instance


