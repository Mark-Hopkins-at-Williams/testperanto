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

class BasicMultiSplitter(Splitter):
    """
    BasicMultiSplitter is used for multilingual training. Right now this
    works for the basic multi experiment
    """
    def __init__(self, splitter_name, names, corp_lens, main_src, main_tgt, other):
        super().__init__(splitter_name, names)    
        self.corp_lens = corp_lens
        self.main_src = main_src
        self.main_tgt = main_tgt
        self.other = other

    def create_datasets(self):
        src_data = [n for n in self.names if self.main_src in n][0]
        tgt_data = [n for n in self.names if self.main_tgt in n][0]
        other_data = [n for n in self.names if self.other in n][0]

        datasets = []
        for c in self.corp_lens:
            reg_dataset = Dataset(
                name = f"{self.main_src}_{self.main_tgt}_{format_number(c)}",
                src  = [src_data],
                tgt  = [tgt_data],
                corp_lens = [c]
                )
            
            main_dataset = Dataset(
                name = f"{self.other}_{self.main_tgt}_{format_number(c)}",
                src = [other_data],
                tgt = [tgt_data],
                corp_lens = [c]
            )
        
            mixed_dataset = Dataset(
                name = f"{self.main_src}_{self.other}_{self.main_tgt}_{format_number(c)}",
                src = [src_data, other_data],
                tgt = [tgt_data, tgt_data],
                corp_lens = [c//2] * 2 #16000, 16000
                )
            
            datasets.extend([reg_dataset, main_dataset, mixed_dataset])
         
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
    names     = fetch_data(['basic_multi']) # ['engl1.basic_multi', ... 'japn.basic_multi]
    names = [n for n in names if '2' not in n]
    splitter = BasicMultiSplitter('basic_multi', names, [1000 * (2 ** i) for i in range(1,6)], 
    main_src='en_svo', main_tgt ='en_sov', other='fr')
    splitter.update_metadata()
