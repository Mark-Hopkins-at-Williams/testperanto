import itertools 
import os 
from config import Config

class DataProcessor:
    """
    class for processing generated testperanto data
    """
    def __init__(self, config: Config):
        self.corp_lens = config.corp_lens
        self.output_path = config.OUT_PATH
        self.exp_name = config.exp_name
        self.per_tree = config.peranto_tree
        self.num_trans = config.num_trans
        self.train_size = config.train_size
        self.test_size = config.test_size 
        self.dev_size = config.dev_size
        self.train_path = config.TRAIN_PATH

    def format_number(self, num):
        if num >= 1000000:
            return f"{num/1000000:.1f}m"
        elif num >= 1000:
            return f"{num/1000:.1f}k"
        else:
            return str(num)

    def clean(self):
        for corp_len in self.corp_lens:
            for i in range(len(self.per_tree.names)):
                file_path = f"{self.output_path}/{self.exp_name}{self.format_number(corp_len)}.{i}"

                # read the file
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                
                # clean the data
                cleaned_lines = [line.split('#')[0].strip() for line in lines]
                with open(file_path, 'w') as file:
                    file.write('\n'.join(cleaned_lines))

    def train_test_split(self):
        for corp_len in self.corp_lens:
            form_len  = self.format_number(corp_len)
            train_len = int(corp_len * self.train_size)
            test_len  = int(corp_len * self.test_size)
            dev_len   = corp_len - train_len - test_len

            # for each combo, ex: (1, 4)
            for idxs in itertools.combinations(range(len(self.per_tree.names)), self.num_trans):
                folder_name = f"{'_'.join([self.per_tree.names[i] for i in idxs])}_{form_len}"  # SVO_OSV_100
                folder_path = f"{self.train_path}/{folder_name}"
                os.makedirs(folder_path, exist_ok=True)

                for i in idxs:
                    file_path = f"{self.output_path}/{self.exp_name}{form_len}.{i}"
                    language = self.per_tree.languages[i]
                    
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
    config = Config()
    data_proc = DataProcessor(config)
    data_proc.process()