import itertools
import os
from abc import ABC, abstractmethod
from helper import format_number

"""
Functionality To Do:
    Track different metrics (fairseq?)
    Track metrics across different languages/easier access (no idea)
    More model flexibility (easy)
    Different splitters (should be easy)
    Global data collection
    Separate Data Generation and Then processing for each experiment
"""

class PerantoTree:
    def __init__(self,
            json_path,
            amr_files,
            middleman_files,
            language_files,
            names
            ):
        
        self.amr_paths    = [f'{json_path}/amr_files/{amr_file}.json' for amr_file in amr_files]
        self.middle_paths = [f'{json_path}/middleman_files/{mid_file}.json' for mid_file in middleman_files]
        self.lang_paths   = [f'{json_path}/language_files/{lang_file}.json' for lang_file in language_files]

        self.names        = names 

        self.file_order   = {i : name for i, name in enumerate(self.names)}

        self.data         = [self.amr_paths, self.middle_paths, self.lang_paths]
        self.num_paths  = len(self.amr_paths) * len(self.middle_paths) * len(self.lang_paths)

class Dataset:
    
    def __init__(self,
            src,
            tgt,
            corp_lens,
            name,
            data_path
            ):
        if not isinstance(src, list):
            src       = [src]
            tgt       = [tgt]
            corp_lens = [corp_lens]
        
        if not len(src) == len(tgt) == len(corp_lens):
            raise ValueError(f"src, tgt, corp_lens must all be same sizes.")

        src_counts = dict(zip(src, corp_lens))
        tgt_counts = dict(zip(tgt, corp_lens))
        src_paths = {name : f'{data_path}.{name}' for name in src}
        tgt_paths = {name : f'{data_path}.{name}' for name in tgt}

        self.src_info  = {n : (src_counts[n], src_paths[n]) for n in src}
        self.tgt_info  = {n : (tgt_counts[n], tgt_paths[n]) for n in tgt}
        self.name = name 

    def get_data(self, lang='src'):
        if lang == 'src':
            paths = self.src_info 
        else:
            paths = self.tgt_info 
        
        data = {}
        for name, (cnt, fpath) in paths.items():
            with open(fpath, 'r') as f:
                lines = f.readlines()
                data[name] = lines[:cnt] # so always taking first part 

        return data 

class Model:

    def __init__(self, **kwargs):
        
        self.num_epochs = 1000
        self.patience   = 75
        self.size       = 'XS'
    
        size_map = {
            "XS" : 'transformer_iwslt_de_en',
            "S"  : 'transformer',
            'M'  : "transformer_vaswani_wmt_en_de_big"
            }

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model = size_map[self.size] # which model to use 

class AbstractConfig(ABC):
    """
    AbstractConfig is an abstract experiment configuration. Subclasses need an exp_name, 
    and they can override any default params defined here.
    
    The params here help streamline stuff down the line. For example all the paths are 
    defined here, as well as things like whether or not to train identity maps etc.
    """
    def __init__(self, **kwargs):
        super().__init__()

        ### path config
        self.SRC_PATH     = os.getcwd()                                     # path to src code 
        self.MAIN_PATH    = os.path.dirname(self.SRC_PATH)                  # main experiment_pipeline path
        self.PERANTO_PATH = os.path.dirname(self.MAIN_PATH)                 # testperanto main path
        self.DATA_PATH    = f"{self.MAIN_PATH}/experiment_data"             # data path
        self.EXP_PATH     = f"{self.DATA_PATH}/experiments/{self.exp_name}" # path for this experiment
        self.TRAIN_PATH   = f"{self.EXP_PATH}/data"                         # for train/test/dev sets
        self.RESULTS_PATH = f'{self.EXP_PATH}/results'                      # for model results
        self.APPA_PATH    = f"{self.PERANTO_PATH}/appa-mt/fairseq"          # for appa fairseq training
        self.JSON_PATH    = f"{self.DATA_PATH}/peranto_files"               # contains 3 folders w/ tp config 
        self.OUT_PATH     = f"{self.EXP_PATH}/output"                       # path for generated data 
        self.YAML_FPATH   = f"{self.EXP_PATH}/{self.exp_name}.yaml"         
        self.SH_FPATH     = f"{self.EXP_PATH}/{self.exp_name}.sh"           

        ### data generator config
        self.amr_files       = ['amr']
        self.middleman_files = ['middleman']                                # json_path/middleman_files/middleman.json
        self.language_files  = [f"english{''.join(p)}"                      # json_path/language_files/englishSVO.json ...
                                for p in itertools.permutations("SVO")]     
        self.names           = [''.join(p) for p in itertools.permutations("SVO")]
        self.max_len         = 32000

        ### data processor config
        self.train_size      = .8                                                     # train proprortion 
        self.test_size       = .1                                                     # test proportion
        self.dev_size        = .1                                                     # dev proportion
        self.split_method    = 'pairwise_splitter'                                    # how to split generated data into datasets 
        self.splitter_params = {"corp_lens" : [1000 * (2 ** i) for i in range(6)]}    # splitting configs

        ### trainer config
        self.num_gpus      = 2
        self.num_models    = 1                                                        # models per dataset
        self.model_params  = [{}]                                                     # params to change for models

        for key, value in kwargs.items():
            setattr(self, key, value)

        ### higher level data structures 
        self.per_tree = None                                                # PerantoTree configures a data generation
        self.datasets = None                                                # List of Datasets to create from (PerantoTree, C) pair
        self.models   = None                                                # List of Models to train for each dataset in self.datasets

        self.initialize()

    @property
    @abstractmethod
    def exp_name(self):
        """Need to specify experiment name."""
        pass

    def initialize(self):
        paths = [
            self.SRC_PATH,
            self.MAIN_PATH,
            self.PERANTO_PATH,
            self.DATA_PATH,
            self.EXP_PATH,
            self.JSON_PATH,
            self.OUT_PATH,
            self.TRAIN_PATH,
            self.RESULTS_PATH
            ]
        
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

        ### check json_path has correct subfolders 
        self.per_tree = PerantoTree(                                   
                json_path       = self.JSON_PATH,                              
                amr_files       = self.amr_files,                                 
                middleman_files = self.middleman_files,                                
                language_files  = self.language_files,
                names           = self.names
                )
        
        self.datasets = self.split(self.split_method)
        self.models   = [Model(**params) for params in self.model_params]

    def split(self, split_method='pairwise_splitter'):
        if split_method == 'pairwise_splitter':
            id_map    = self.splitter_params.get('id_map', None)
            corp_lens = self.splitter_params['corp_lens']
            get_name  = lambda n1, n2, c : f"{n1}_{n2}_{format_number(c)}"
            names     = self.per_tree.names

            if id_map is None:
                triples = [(n1, n2, c) for n1 in names for n2 in names for c in corp_lens if n1 != n2]
            else:
                pairwise_names = [name for name in names if name not in id_map and name not in id_map.values()]
                triples  = [(n1, n2, c) for n1 in pairwise_names for n2 in pairwise_names for c in corp_lens if n1 != n2]
                triples += [(n1, n2, c) for (n1, n2) in list(id_map.items()) for c in corp_lens]
                triples += [(n2, n1, c) for (n1, n2) in list(id_map.items()) for c in corp_lens]

            datasets = [
                Dataset(
                    src = n1,
                    tgt = n2,
                    corp_lens = c,
                    name = get_name(n1, n2, c),
                    data_path = f'{self.OUT_PATH}/{self.exp_name}{format_number(self.max_len)}'
                    )
                for (n1, n2, c) in triples
                ]
            
            return datasets

class SVOConfig(AbstractConfig):

    @property
    def exp_name(self):
        return 'svo_perm_test'

    def __init__(self, **kwargs):
        kwargs['max_len'] = 1000
        kwargs['splitter_params'] = {'corp_lens' : [500, 1000]}
        super().__init__(**kwargs)

class SizeConfig(AbstractConfig):

    @property
    def exp_name(self):
        return 'model_size'

    def __init__(self, **kwargs):
        kwargs['language_files'] = ['englishSVO', 'englishOVS']
        kwargs['names'] = ['SVO', "OVS"]
        kwargs['max_len'] = 16000
        kwargs['splitter_params'] = {'corp_lens' : [1000, 4000, 16000]}
        kwargs['num_models'] = 3
        kwargs['model_params'] = [{'size' : 'XS'}, {'size' : 'S'}, {'size' : 'M'}]
        super().__init__(**kwargs)

if __name__ == "__main__":
    config = SVOConfig()