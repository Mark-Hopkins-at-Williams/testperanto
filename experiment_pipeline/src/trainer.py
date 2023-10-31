import itertools 
import os 

from config import Config
from data_processor import DataProcessor
from data_generator import DataGenerator

class Trainer:

    def __init__(self, config: Config):
        self.per_tree = config.peranto_tree
        self.num_trans = config.num_trans
        self.corp_lens = config.corp_lens
        self.train_path = config.TRAIN_PATH
        self.results_path = config.RESULTS_PATH
        self.appa_path    = config.APPA_PATH
        self.num_epochs = config.num_epochs

    def format_number(self, num):
        if num >= 1000000:
            return f"{num/1000000:.1f}m"
        elif num >= 1000:
            return f"{num/1000:.1f}k"
        else:
            return str(num)

    def train(self, n=2):
        cnt = 0
        for corp_len in self.corp_lens:
            for idxs in itertools.combinations(range(len(self.per_tree.names)), self.num_trans):
                folder_name = f"{'_'.join([self.per_tree.names[i] for i in idxs])}_{self.format_number(corp_len)}" 
                data_dir = f"{self.train_path}/{folder_name}"
                work_dir = f"{self.results_path}/{folder_name}"
                src      = self.per_tree.languages[idxs[0]] # [en, ...]  idxs (3, 5)
                tgt      = self.per_tree.languages[idxs[1]] # assumes only 2 for now

                if not os.path.exists(work_dir):
                    cnt += 1
                    print('=' * 15)
                    call = f"sbatch {self.appa_path}/train.sh {work_dir} {data_dir} {src} {tgt} {self.num_epochs}"
                    print(call)
                    print('=' * 15)
                    if cnt == n:
                        return None


if __name__ == '__main__':
    config = Config()
    trainer = Trainer(config)
    trainer.train()