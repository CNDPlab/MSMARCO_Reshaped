from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
import ipdb
tokenizer = TreebankWordTokenizer()



def tokenize(instance):
    instance = sentence_seg(instance)
    instance = word_tokenize(instance)
    return instance

def sentence_seg(instance):
    for index, passage in instance['passages'].items():
        instance['passages'][index]['text'] = sent_tokenize(passage['text'].lower())
    for index, answer in instance['answers'].items():
        try:
            instance['answers'][index]['text'] = sent_tokenize(answer['text'].lower())
        except:
            ipdb.set_trace()
    return instance

def word_tokenize(instance):
    for index, answer in instance['answers'].items():
        if answer['text'] != '':
            instance['answers'][index]['text'] = list(sum([tokenizer.tokenize(sen) for sen in answer['text']], []))
            instance['answers'][index]['char'] = [[char for char in word] for word in instance['answers'][index]['text']]
    for index, passage in instance['passages'].items():
        if passage['text'] != '':
            instance['passages'][index]['text'] = list(sum([tokenizer.tokenize(sen) for sen in passage['text']], []))
            instance['passages'][index]['char'] = [[char for char in word] for word in instance['passages'][index]['text']]
    instance['question']['text'] = tokenizer.tokenize(instance['question']['text'].lower())
    instance['question']['char'] = [[char for char in word] for word in instance['question']['text']]
    return instance