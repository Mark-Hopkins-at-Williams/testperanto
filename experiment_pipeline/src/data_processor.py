import itertools 
import os 
import glob 

from config import *
from helper import format_number

class DataProcessor:
    """
    class for processing generated testperanto data
    """
    def __init__(self, config: AbstractConfig):
        self.corp_lens = config.corp_lens
        self.output_path = config.OUT_PATH
        self.exp_name = config.exp_name
        self.per_tree = config.peranto_tree
        self.train_path = config.TRAIN_PATH
        self.combos = config.combos
        self.num_trans = config.num_trans
        self.train_size = config.train_size
        self.test_size = config.test_size 
        self.dev_size = config.dev_size

    def clean(self):
        pattern = f"{self.output_path}/{self.exp_name}*"
        for file_path in glob.glob(pattern):
            # read the file
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # clean the data
            cleaned_lines = [line.split('#')[0].strip() for line in lines]
            with open(file_path, 'w') as file:
                file.write('\n'.join(cleaned_lines))

    def train_test_split(self):
        for corp_len in self.corp_lens:
            form_len  = format_number(corp_len)
            train_len = int(corp_len * self.train_size)
            test_len  = int(corp_len * self.test_size)
            dev_len   = corp_len - train_len - test_len 

            for combo in self.combos:
                folder_name = f"{'_'.join(combo)}_{form_len}"
                folder_path = f"{self.train_path}/{folder_name}"
                os.makedirs(folder_path, exist_ok=True)

                file_paths = [f'{self.output_path}/{self.exp_name}{form_len}.{name}' for name in combo]
                for i, file_path in enumerate(file_paths):

                    language = combo[i].lower() #svo
                    
                    # read the cleaned data 
                    with open(file_path, 'r') as file:
                        lines = file.readlines()

                    assert train_len + test_len + dev_len <= len(lines), f"Sum of lengths exceeds number of lines for {file_path}"

                    # Split the data
                    train_data = lines[:train_len]
                    test_data  = lines[train_len:train_len+test_len]
                    dev_data   = lines[train_len+test_len:]

                    with open(f"{folder_path}/train.{language}", "w") as f:
                        f.write("".join(train_data))
                    with open(f"{folder_path}/test.{language}", "w") as f:
                        f.write("".join(test_data))
                    with open(f"{folder_path}/dev.{language}", "w") as f:
                        f.write("".join(dev_data))

    def process(self):
        self.clean()
        self.train_test_split()

if __name__ == "__main__":
    config = SVOConfig()
    data_proc = DataProcessor(config)
    data_proc.process()