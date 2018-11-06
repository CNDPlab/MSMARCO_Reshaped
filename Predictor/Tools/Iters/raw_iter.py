from configs import DefaultConfig
import os
import json


class RawIter(object):
    def __init__(self, set):

        self.file = os.path.join(DefaultConfig.raw_folder, set + '.json')

    def __iter__(self):

        with open(self.file) as reader:
            for line in reader:
                yield line






if __name__ == '__main__':
    it = RawIter('dev')
    for i in it:
        line = json.loads(i)
        input()
