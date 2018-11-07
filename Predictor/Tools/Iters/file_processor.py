import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def process_file(input_file, process_func, output_file):
    """
    :param input_file:  abs file_path
    :param process_func: string_dict to dict  need to json.loads before
    :param output_file: abs file_path
    :return:
    """
    with open(input_file) as reader, open(output_file, 'w') as writer:
        for i in tqdm(reader):
            json.dump(process_func(i), writer)
            writer.write('\n')
    print(f'Done.')


def process_file_mul(input_file, process_fun, output_file, num_thread):
    """
    :param input_file:  abs file_path
    :param process_func: string_dict to dict  need to json.loads before
    :param output_file: abs file_path
    :param num_thread: num thread
    :return:
    """
    with open(input_file) as reader, open(output_file, 'w') as writer:
        datas = reader.readlines()
        with ProcessPoolExecutor(num_thread) as executor:
            for line in tqdm(executor.map(process_fun, datas)):
                json.dump(line, writer)
                writer.write('\n')

    print(f'Done.')
