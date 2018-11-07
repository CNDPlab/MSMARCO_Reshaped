


class Vocab_collector(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def transfer_text(self, instance):
        for index, answer in instance['answers'].items():
            if answer['text'] != []:
                instance['answers'][index]['text'] = [self.vocab.from_token_id(i) for i in answer['text']]
        for index, passage in instance['passages'].items():
            if passage['text'] != []:
                instance['passages'][index]['text'] = [self.vocab.from_token_id(i) for i in passage['text']]
        instance['question']['text'] = [self.vocab.from_token_id(i) for i in instance['question']['text']]
        return instance

    def transfer_char(self, instance):
        return instance

    def transfer_pos(self, instance):
        return instance


