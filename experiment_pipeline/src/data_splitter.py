from globals import * 
from helper import format_number

from abc import ABC, abstractmethod 

import os
import json
import itertools

class Dataset:
    """
    A Dataset defines a dataset to train various models on. The data comes from splitting 
    raw testperanto data (generated in data_generator.py).

    Args:
        name (str): dataset name
        src (list[str]): src languages
        tgt (list[str]): tgt languages
        corp_lens (list[int]): the dataset contains corp_lens[i] sentences from src[i] -> tgt[i]
        test_lp (tuple[str], optional): if not None then test/dev only contains test_lp[0] -> test_lp[1] sentences.
        
    Methods:
        create(): Creates train, test, dev sets for src/tgt.
        read_to_list(): Reads lines from a file and adds them to a list.
        write_list_to_file(): Writes lines from a list to a file.

    Important Info:
        - test_lp[0] must be in src, test_lp[1] must be in tgt 
        - This version uses a constant TEST/DEV_SIZE, rather than a proportion. For everything before new_switches
        experiment, we used a train proportion of .8, a test proportion of .1, and a dev proportion of .1. If you 
        need that code just look at earlier version of GitHub. 

    Example:
        >>> dataset = Dataset(
            name      = 'hey_mark',
            src       = ['SVO', 'SVO'],
            tgt       = ['SOV', 'VSO'],
            corp_lens = [2000, 2000],
            test_lp   = ('SVO', 'SOV)
            )
        creates a one to many Dataset called hey_mark comprised of 4000 sentences, 2000 that 
        are SVO -> SOV and 2000 that are SVO -> VSO. If this was called then you could go to
        the datasets folder, find hey_mark, and inside see 
            train.src: contains 2000 + 2000 = 4000 SVO sentences
            test.src : contains TEST_SIZE SVO sentences
            dev.src  : contains DEV_SIZE SVO sentences
            train.tgt: contains 2000 SOV sentences followed by 2000 VSO sentences
            test.tgt : contains TEST_SIZE SOV sentences (note: if test_lp is None this would be half SOV half VSO)
            dev.tgt  : contains DEV_SIZE SOV sentences  (note: if test_lp is None this would be half SOV half VSO)

    Notice here there's a potential data contamination issue. Let's use the Dataset example above. We have 64k raw SVO
    sentences. To create our src data we take the first 2000 sentences (for the SVO -> SOV part). To avoid reusing data,
    when we are creating the SVO -> VSO part of our dataset we take the next 2000 sentences. 
    """

    def __init__(self, name, src, tgt, corp_lens, test_lp=None):
        self.name      = name
        self.src       = src
        self.tgt       = tgt
        self.corp_lens = corp_lens
        self.test_lp   = test_lp

    def create(self):
        train_src, test_src, dev_src = [], [], []
        train_tgt, test_tgt, dev_tgt = [], [], []
        dataset_folder               = os.path.join(DATASET_PATH, self.name)
        os.makedirs(dataset_folder, exist_ok=True)
        start_line = 0

        dev_interval  = DEV_SIZE // len(self.src)
        test_interval = TEST_SIZE // len(self.src)

        for i, (src_name, tgt_name) in enumerate(zip(self.src, self.tgt)):
            train_end = start_line + self.corp_lens[i]
            test_end  = train_end + test_interval
            dev_end   = test_end + dev_interval

            self.read_to_list(src_name, start_line, train_end, train_src)
            self.read_to_list(tgt_name, start_line, train_end, train_tgt)

            test_src_name = src_name if not self.test_lp else self.test_lp[0]
            test_tgt_name = tgt_name if not self.test_lp else self.test_lp[1]
            dev_src_name  = src_name if not self.test_lp else self.test_lp[0]
            dev_tgt_name  = tgt_name if not self.test_lp else self.test_lp[1]


            self.read_to_list(test_src_name, train_end, test_end, test_src)
            self.read_to_list(test_tgt_name, train_end, test_end, test_tgt)
            self.read_to_list(dev_src_name, test_end, dev_end, dev_src)
            self.read_to_list(dev_tgt_name, test_end, dev_end, dev_tgt)

            amt = self.corp_lens[i] + test_interval + dev_interval
            start_line += amt
        
        self.write_to_files(dataset_folder, train_src, test_src, dev_src, train_tgt, test_tgt, dev_tgt)
            
    def read_to_list(self, file_name, start_line, end_line, target_list):
        fpath = f"{TP_DATA_PATH}/{file_name}"
        with open(fpath, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i >= end_line:
                    break
                if i >= start_line:
                    target_list.append(line.strip())

    def write_to_files(self, dataset_folder, train_src, test_src, dev_src, train_tgt, test_tgt, dev_tgt):
        file_pairs = {
            'train.src': train_src, 'test.src': test_src, 'dev.src': dev_src,
            'train.tgt': train_tgt, 'test.tgt': test_tgt, 'dev.tgt': dev_tgt
        }
        for filename, data_list in file_pairs.items():
            self.write_list_to_file(data_list, os.path.join(dataset_folder, filename))

    def write_list_to_file(self, data_list, file_path):
        """
        Writes each line from a list into a file.

        Args:
            data_list (list of str): The list of lines to write.
            file_path (str): The path to the file to write to.
        """
        with open(file_path, 'w', encoding='utf-8') as file:
            for line in data_list:
                file.write(f"{line}\n")

class Splitter(ABC):
    """
    Splitter is an abstract class for splitting raw data (i.e. data created via data_generator.py) into 
    a list of Datasets ready for training. To create a subclass you need to specify a way to take a list
    of data names (e.g. SVO.svo_perm) and split it into various Datasets.

    Args:
        splitter_name (str): name of splitter
        names (list[str]): list of names to split
            often subclasses will have more params 
    
    Methods:
        create_datasets (abstract): splits self.names into list[Datasets]
        update_metadata           : updates dataset_info.json with all info

    Important Info:
        - Because I'm a crap coder sometimes the create_datasets() should end with a loop over the Datasets,
        calling .create() on each (this is an easy but irrelevant fix)

    See below for examples of Splitters
    """
    def __init__(self, splitter_name, names):
        self.splitter_name = splitter_name
        self.names = names

    @abstractmethod
    def create_datasets(self):
        raise NotImplementedError   

    def update_metadata(self):
        json_file = f"{DATA_PATH}/dataset_info.json"
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
        except:
            data = {}

        datasets                 = self.create_datasets()
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

class PairwiseSplitter(Splitter):
    """
    PairwiseSplitter takes a bunch of names and creates pairwise datasets for each corpus size.
    This is used in the svo_perm experiment.

    Additional Args:
        - corp_lens (list[int]): list of different dataset sizes

    Formally, let N be the set of names and C be the set of corp lens. The datasets created are
    
    >>> Dataset(
        src       = [n1],
        tgt       = [n2],
        corp_lens = c
        )
    for each c in C for each distinct n1, n2 in names.

    Important Info:
        - Sometimes you may want to train identity maps, which differ only in vocabulary. To do
        this you need to generate two instances of the same (amr, mm, lang) triple, with one named
        the same as the other + "_id" 
    """

    def __init__(self, splitter_name, names, corp_lens):
        super().__init__(splitter_name, names)   
        self.corp_lens = corp_lens 
        self.triples   = self.get_triples()      

    def get_triples(self):
        ### split names into those with and without "_id"
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

class BasicMultiSplitter(Splitter):
    """
    BasicMultiSplitter is used for multilingual training. Right now this
    works for the basic multi experiment.

    Additional Args:
        - corp_lens (list[int]): list of corpus lengths
        - main_src (str)       : subset of data name that will be src (see below for better description)
        - main_tgt (str)       : equivalent for tgt
        - other (str)          : equivalent for other 

    Here we create 3 datasets for each c in corp_lens:
        1) src         -> tgt (control)
        2) other       -> tgt 
        3) src + other -> tgt

    main_src is just the naming convention used, but it must be a subset of the data name to allow the 
    code to find the data corresponding to main_src
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
     
class SwitchesSplitter(Splitter):
    """
    SwitchesSplitter is used for the switches multilingual experiment. 

    Additional Args:
        - corp_lens (list[int]): list of corpus sizes
        - src (str)            : src language
        - src_name             : (str) name for src 
        - tgt (str)            : tgt language
        - tgt_name (str)       : name for tgt

    Specifically, we have the 2^6 = 64 tp raw data. We specify a src 
    (we used s011101) and a name for it (en), as well for tgt (s000000 and japn). 
    Then for each c in corp_lens we train
        src + aux -> tgt, where aux in {switches} - tgt

    Really you should be using NewSwitches.
    """
    def __init__(self, splitter_name, names, corp_lens, src, src_name, tgt, tgt_name):
        super().__init__(splitter_name, names)    
        self.corp_lens = corp_lens
        self.src = src
        self.tgt = tgt
        self.src_name = src_name 
        self.tgt_name = tgt_name
        self.auxilaries = [n for n in names if n != src and n != tgt]

    def create_datasets(self):
        datasets = []

        for c in self.corp_lens:
            main = Dataset(
                name = f"{self.src_name}_{self.tgt_name}_{format_number(c)}",
                src = [self.src],
                tgt = [self.tgt],
                corp_lens = [c]
                )
            
            datasets.append(main)
            for aux in self.auxilaries:
                dataset = Dataset(
                    name = f"{aux.split('.')[0]}_{self.src_name}_{self.tgt_name}_{format_number(c)}",
                    src = [self.src, aux],
                    tgt = [self.tgt, self.tgt],
                    corp_lens = [c // 2] * 2,
                    test_lp = (src, tgt)
                    )

                datasets.append(dataset)
         
        for dataset in datasets:
            dataset.create()

        return datasets 

class NewSwitchesSplitter(Splitter):
    """
    SwitchesSplitter is used for the new_switches multilingual experiment. 

    Additional Args:
        - corp_lens (list[int]): list of corpus sizes
        - src (str)            : src language
        - src_name             : (str) name for src 
        - tgt (str)            : tgt language
        - tgt_name (str)       : name for tgt

    This is the same as switches except we have an extra generation of src (call it src2).
    So for each c in corp_lens we train
        src + aux -> tgt, where aux in {switches} + src2 - src - tgt
    """
    def __init__(self, splitter_name, names, corp_lens, src, src_name, tgt, tgt_name):
        super().__init__(splitter_name, names)    
        self.corp_lens = corp_lens
        self.src = src
        self.tgt = tgt
        self.src_name = src_name 
        self.tgt_name = tgt_name
        self.auxilaries = [n for n in names if n != src and n != tgt]

    def create_datasets(self):
        datasets = []

        for c in self.corp_lens:
            for aux in self.auxilaries:
                dataset = Dataset(
                    name = f"{aux.split('.')[0]}_{self.src_name}_{self.tgt_name}_{format_number(c)}_new",
                    src = [self.src, aux],
                    tgt = [self.tgt, self.tgt],
                    corp_lens = [c // 2] * 2,
                    test_lp = (self.src, self.tgt)
                    )

                datasets.append(dataset)
         
        for dataset in datasets:
            dataset.create()

        return datasets 

def fetch_data(gen_names=None, names=None):
    """
    fetch raw data from data_info.json by specifying a generator/list of names
    
    Args:
        gen_names (list[str]): generator names to fetch
        names (list[str])    : specific names to fetch

    Example:
        >>> fetch_data(['svo_perm']) gets all svo_perm data names
        >>> fetch_data(['svo_perm'], ['SVO','SOV']) gets all svo_perm that are subject first 
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
    
    clean() 
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

if __name__ == "__main__":
    names = fetch_data(['new_switches']) 
    corp_lens = [1000 * (2 ** i) for i in range(1,5)]
    src = 's011101.new_switches'
    tgt = 's000000.new_switches'
    splitter = NewSwitchesSplitter('new_switches', names, corp_lens, src, 'en', tgt, 'japn')
    splitter.update_metadata()
