from Predictor.Tools.DataTools import build_data_structure
from Predictor.Tools.DataTools import tokenize
from Predictor.Tools.DataTools import extract_golden_span
import json



DATASETS = ['dev', 'train', 'eval']




def process_instance(raw_line):
    line = json.loads(raw_line)
    instance = build_data_structure(line)
    instance = tokenize(instance)
    instance = extract_golden_span(instance)
    return instance

def convert2id(raw_line):
    line = json.loads(raw_line)
    instance = None

if __name__ == '__main__':
    for dataset in DATASETS:
        pass

