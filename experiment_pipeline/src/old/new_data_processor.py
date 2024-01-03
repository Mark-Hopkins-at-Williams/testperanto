import os 
import glob 
from abc import ABC, abstractmethod
from new_config import *
from helper import format_number


class DataProcessor:
    """
    class for processing generated testperanto data
    """
    def __init__(self, config: AbstractConfig):
        self.exp_name    = config.exp_name
        self.output_path = config.OUT_PATH
        self.train_path  = config.TRAIN_PATH

        self.max_len     = config.max_len
        self.datasets    = config.datasets

        self.train_size  = config.train_size
        self.test_size   = config.test_size 

    def clean(self):
        ### take generated data and clen it 
        pattern = f"{self.output_path}/{self.exp_name}*" # anything w/ this pattern
        for file_path in glob.glob(pattern):
            # read the file
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # clean the data
            cleaned_lines = [line.split('#')[0].strip() for line in lines]
            with open(file_path, 'w') as file:
                file.write('\n'.join(cleaned_lines))

    def train_test_split(self):
        for dataset in self.datasets:
            folder_path = f"{self.train_path}/{dataset.name}"
            os.makedirs(folder_path, exist_ok=True)
            for lang in ['src', 'tgt']:
                # data = {"SVO" : [sent1, sent2, ...], 'SOV' : [sent1, sent2, ...]}
                data = dataset.get_data(lang)
                train_data = []
                test_data  = []
                dev_data   = []

                for name, info in data.items():
                    length = len(info)
                    train_len = int(self.train_size * length)
                    test_len  = int(self.test_size * length)

                    train  = info[: train_len]
                    test   = info[train_len : train_len + test_len]
                    dev    = info[train_len + test_len:]

                    train_data += train
                    test_data  += test 
                    dev_data   += dev 

                with open(f"{folder_path}/train.{lang}", "w") as f:
                    f.write("".join(train_data))

                with open(f"{folder_path}/test.{lang}", "w") as f:
                    f.write("".join(test_data))

                with open(f"{folder_path}/dev.{lang}", "w") as f:
                    f.write("".join(dev_data))

    def process(self):
        ### process data- clean it and then split it into datasets
        self.clean()
        self.train_test_split()

if __name__ == "__main__":
    config = SVOConfig()
    processor = DataProcessor(config)
    processor.process()