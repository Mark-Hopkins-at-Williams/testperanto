from globals import * 
from helper import format_number

from abc import ABC, abstractmethod 

import os
import json
import itertools

class Dataset:
    """
    A Dataset takes generated data and assembles a dataset. To allow for 
    multilingual training, a Dataset is configured by a list of data names
    (for example OSV.svo_perm) for both src/tgt, as well as an associated 
    corp_lens lst. 

    Example:

    src = ['OSV.svo_perm', 'SOV.svo_perm']
    tgt = ['SVO.svo_perm', 'SVO.svo_perm']
    corp_lens = [32000, 16000]

    this will create a many to one model, where the corpus is comprised of 
    32k sentences mapping OSV -> SVO, and 16k mapping SOV-SVO. 

    See self.create() to see conventions that avoid data contamination
    See Splitter/fetch_data() to see quick ways to create Datasets
    """
    def __init__(self, name, src, tgt, corp_lens):
        self.name = name
        self.src = src
        self.tgt = tgt
        self.corp_lens = corp_lens

    def create(self):
        """
        This creates train, test, dev sets for src/tgt. To see why we can't naively take 
        the first corp_lens[i] sentences of src[i]/tgt[i], note that in the example in the 
        constructor that would repeat SOV sentences (and since all the sentences are from 
        the same generator technically it would have multiple sentences that are all translations
        of each other, which would contaminate the data)
        """
        train_src, test_src, dev_src = [], [], []
        train_tgt, test_tgt, dev_tgt = [], [], []

        dataset_folder = os.path.join(DATASET_PATH, self.name)
        os.makedirs(dataset_folder, exist_ok=True)

        start_line = 0  # to keep track of where to start reading for each language pair
        for i in range(len(self.src)):
            # calculate the number of lines for each subset
            train_end = start_line + int(self.corp_lens[i] * TRAIN_SIZE)
            test_end = train_end + int(self.corp_lens[i] * TEST_SIZE)

            # append the lines to the respective lists
            self.read_to_list(self.src[i], start_line, train_end, train_src)
            self.read_to_list(self.src[i], train_end, test_end, test_src)
            self.read_to_list(self.src[i], test_end, start_line + self.corp_lens[i], dev_src)

            self.read_to_list(self.tgt[i], start_line, train_end, train_tgt)
            self.read_to_list(self.tgt[i], train_end, test_end, test_tgt)
            self.read_to_list(self.tgt[i], test_end, start_line + self.corp_lens[i], dev_tgt)

            start_line += self.corp_lens[i]  # update the start line for the next file

        # write the lists to files
        self.write_list_to_file(train_src, os.path.join(dataset_folder, 'train.src'))
        self.write_list_to_file(test_src, os.path.join(dataset_folder, 'test.src'))
        self.write_list_to_file(dev_src, os.path.join(dataset_folder, 'dev.src'))
        self.write_list_to_file(train_tgt, os.path.join(dataset_folder, 'train.tgt'))
        self.write_list_to_file(test_tgt, os.path.join(dataset_folder, 'test.tgt'))
        self.write_list_to_file(dev_tgt, os.path.join(dataset_folder, 'dev.tgt'))

    def read_to_list(self, file_name, start_line, end_line, target_list):
        fpath = f"{TP_DATA_PATH}/{file_name}"
        with open(fpath, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i >= end_line:
                    break
                if i >= start_line:
                    target_list.append(line.strip())

    def write_list_to_file(self, data_list, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            for line in data_list:
                file.write(f"{line}\n")

def fetch_data(gen_names=None, names=None):
    """
    fetch data names by specifying a generator or a list of names
    
    fetch_data(['svo_perm']) gets all svo_perm data names
    fetch_data(['svo_perm'], ['SVO','SOV']) gets all svo_perm that is subject first 
    """
    def clean():
        """cleans generated data"""
        for file in os.listdir(TP_DATA_PATH):
            file_path = f'{TP_DATA_PATH}/{file}'
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            cleaned_lines = [line.split('#')[0].strip() for line in lines]
            with open(file_path, 'w') as file:
                file.write('\n'.join(cleaned_lines))
    
    clean() # first clean the data
    result    = []
    data_path = f"{DATA_PATH}/data_info.json"

    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    if gen_names is not None:
        for gen_name in gen_names:
            result.extend([f"{name}.{gen_name}" for name in list(data[gen_name].keys())])

    if names is not None:
        for gen_name, name_dict in data.items():
            extras = [f"{name}.{gen_name}" for name in list(name_dict.keys()) if name in names]
            result.extend(extras)

    return list(set(result))

class Splitter(ABC):
    """
    Main class for splitting raw data (i.e. data created via data_generator.py) into Datasets
    ready for training. Splitter is an abstract class that specifies a way to take a list of 
    data names (e.g. SVO.svo_perm) and split it into various Datasets. 
    """
    def __init__(self, splitter_name, names):
        self.splitter_name = splitter_name
        self.names = names

    @abstractmethod
    def create_datasets(self):
        ### splits self.names into a list of Datasets 
        raise NotImplementedError   

    def update_metadata(self):
        ### updates dataset_info.json with new info 
        json_file = f"{DATA_PATH}/dataset_info.json"
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)

        except:
            data = {}

        datasets = self.create_datasets()
        data[self.splitter_name] = []

        for dataset in datasets:
            dataset_info = {
                dataset.name: {
                    'src':       dataset.src,
                    'tgt':       dataset.tgt,
                    'corp_lens': dataset.corp_lens
                }
            }
            data[self.splitter_name].append(dataset_info)

        with open(json_file, 'w') as file:
            json.dump(data, file, indent=4)

class MultiSplitter(Splitter):
    """
    MultiSplitter is used for multilingual training. This is a dumb class rn
    """
    def __init__(self, splitter_name, names, corp_lens, src, tgt):
        super().__init__(splitter_name, names)    
        self.corp_lens = corp_lens
        self.src = src
        self.tgt = tgt
        self.others = [n for n in names if n != src and n != tgt]

    def create_datasets(self):
        get_name = lambda c, k : f"{self.src.split('.')[0]}_{self.tgt.split('.')[0]}_{k}_{format_number(c)}"
        datasets = []
        for c in self.corp_lens:
            for k in range(len(self.others)):
                corp_lens = [int(.5 * c)] + [int(.5*c/(k+1))] * (k)
                corp_lens = corp_lens + [c - sum(corp_lens)]
                dataset = Dataset(
                    name = get_name(c, k+1),
                    src = [self.src] * (k+2),
                    tgt = [self.tgt] + self.others[:k+1], # multilingual
                    corp_lens = corp_lens
                )
                datasets.append(dataset)

        for dataset in datasets:
            dataset.create()

        return datasets 
     
class PairwiseSplitter(Splitter):
    """
    One splitting example is taking pairwise within the names. For example
    this is used in the svo permutation experiment
    """

    def __init__(self, splitter_name, names, corp_lens):
        super().__init__(splitter_name, names)   
        self.corp_lens = corp_lens 
        self.triples   = self.get_triples()      

    def get_triples(self):
        # split names into those with and without "_id"
        all_names = [name for name in self.names if "_id" not in name]
        id_names  = [name for name in self.names if "_id" in name]

        # create all possible pairs (n1, n2) where n1 != n2 from names_without_id
        name_pairs = list(itertools.permutations(all_names, 2))

        triples = [(n1, n2, c) for (n1, n2) in name_pairs for c in self.corp_lens]

        # add id_maps 
        for name in id_names:
            base_name = name.replace("_id", "")
            for c in self.corp_lens:
                if base_name in all_names:
                    triples.append((base_name, name, c))
                    triples.append((name, base_name, c))

        return triples

    def create_datasets(self):
        get_name = lambda n1, n2, c : f"{n1.split('.')[0]}_{n2.split('.')[0]}_{format_number(c)}"
        datasets = [
            Dataset(
                name      = get_name(n1, n2, c),
                src       = [n1],
                tgt       = [n2],
                corp_lens = [c]
                )
        for (n1, n2, c) in self.triples
        ]
        for dataset in datasets:
            dataset.create()

        return datasets 

if __name__ == "__main__":
    names     = fetch_data(['svo_perm'])
    splitter  = MultiSplitter('num_lang', names, [32000], 'SVO.svo_perm', 'OSV.svo_perm')
    splitter.update_metadata()
