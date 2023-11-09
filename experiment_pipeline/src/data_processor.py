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
        self.train_path = config.TRAIN_PATH
        
        self.num_trans = config.num_trans
        self.train_size = config.train_size
        self.test_size = config.test_size 
        self.dev_size = config.dev_size


    def format_number(self, num):
        if num >= 1000000:
            return f"{num/1000000:.1f}m"
        elif num >= 1000:
            return f"{num/1000:.1f}k"
        else:
            return str(num)

    def clean(self): # TODO: check - modified to allow for duplicate generations 
        for corp_len in self.corp_lens:
            # TODO - account for extra paths
            for i in range(2 * len(self.per_tree.names)): # now cleaning all duplicate translations as well
                ### A here is for the new experiment
                file_path = f"{self.output_path}/{self.exp_name}A{self.format_number(corp_len)}.{i}"

                # read the file
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                
                # clean the data
                cleaned_lines = [line.split('#')[0].strip() for line in lines]
                with open(file_path, 'w') as file:
                    file.write('\n'.join(cleaned_lines))

    def identity_tt_split(self):
        ### for new experiment training svo-svo translation
        for corp_len in self.corp_lens:
            form_len  = self.format_number(corp_len)
            train_len = int(corp_len * self.train_size)
            test_len  = int(corp_len * self.test_size)
            dev_len   = corp_len - train_len - test_len 

            for i, name in enumerate(self.per_tree.names): # TODO: find a way to handle duplicates
                folder_name = f"{name}_{name}_{form_len}"
                folder_path = f"{self.train_path}/{folder_name}"
                os.makedirs(folder_path, exist_ok=True)

                #file_path  = f'{self.output_path}/{self.exp_name}{form_len}.{i}'  # original
                file_path_origional = f'{self.output_path}/{self.exp_name}A{form_len}.{2*i}'
                #print(file_path_origional)
                file_path_duplicate = f'{self.output_path}/{self.exp_name}A{form_len}.{2*i + 1}'
                #print(file_path_duplicate)

                for fpath in [file_path_origional, file_path_duplicate]:
                    language = self.per_tree.languages[i] if fpath == file_path_origional else 'da' 
                    with open(fpath, 'r') as file:
                        lines = file.readlines()

                    assert train_len + test_len + dev_len <= len(lines), f"Sum of lengths exceeds number of lines for {fpath}"

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
        self.identity_tt_split()
        #self.train_test_split()

if __name__ == "__main__":
    config = Config()
    data_proc = DataProcessor(config)
    #data_proc.clean()
    data_proc.identity_tt_split()
