def extract_feature(instance):
    instance = is_in_question(instance)
    return instance


def is_in_question(instance):
    for index, passage in instance['passages'].items():
        in_question = [word in instance['question']['text'] for word in passage['text']]
        instance['passages'][index]['is_in_question'] = [i * 1 for i in in_question]
    return instance


def add_char_feature(instance):




    return instance