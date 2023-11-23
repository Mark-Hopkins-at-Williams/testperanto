import itertools
import os
from abc import ABC, abstractmethod

__all__ = ['AbstractConfig', 'VarConfig', 'SVOConfig']

class PerantoTree:
    """
    PerantoTree specfies a tree of json files. This tree configures the data generation, 
    which is performed by testperanto.

    The tree is a 3 level tree, with top level amr files, middle level middleman files, and
    bottom level language files. 

    json_path: path to a folder with subfolders amr_files, language_files, middleman_files w/ json files
    amr_files: name of amr_file(s)
    middleman_files: name of middleman file(s)
    language_files: name of language file(s)
    names: name to associate each path in the tree 
    identity: true iff training identity maps (e.g. src to src_id)
    """

    def __init__(self, 
            json_path,
            amr_files,
            middleman_files,
            language_files,
            names,
            identity=False
            ):

        self.amr_paths    = [f'{json_path}/amr_files/{amr_file}.json' for amr_file in amr_files]
        self.middle_paths = [f'{json_path}/middleman_files/{mid_file}.json' for mid_file in middleman_files]
        self.lang_paths   = [f'{json_path}/language_files/{lang_file}.json' for lang_file in language_files]
        self.identity     = identity
        self.names        = names #[SVO, OVS, ...] 

        if self.identity:
            self.lang_paths += self.lang_paths # duplicate to allow for identity mapping
            self.dup_names  = [f'{name}_id' for name in names] # duplicate for identity mapping
            self.file_order = {i : name for i, name in enumerate(self.names + self.dup_names)}
        else:
            self.file_order = {i : name for i, name in enumerate(self.names)}


        self.data         = [self.amr_paths, self.middle_paths, self.lang_paths]
        self.num_paths  = len(self.amr_paths) * len(self.middle_paths) * len(self.lang_paths)

    def get(self):
        return self.data

    def get_names(self):
        return self.names
 

class AbstractConfig(ABC):
    """
    AbstractConfig is an abstract experiment configuration. Subclasses need an exp_name, 
    and they can override any default params defined here.
    
    The params here help streamline stuff down the line. For example all the paths are 
    defined here, as well as things like whether or not to train identity maps etc.
    """
    def __init__(self, **kwargs):
        super().__init__()

        ### path configurations
        self.SRC_PATH     = os.getcwd()                                     # src code
        self.MAIN_PATH    = os.path.dirname(self.SRC_PATH)                  # main experiment_pipeline path
        self.PERANTO_PATH = os.path.dirname(self.MAIN_PATH)                 # testperanto main path
        self.DATA_PATH    = f"{self.MAIN_PATH}/experiment_data"             # data path
        self.EXP_PATH     = f"{self.DATA_PATH}/experiments/{self.exp_name}" # path for this experiment
        self.TRAIN_PATH   = f"{self.EXP_PATH}/data"                         # for train/test/dev sets
        self.RESULTS_PATH = f'{self.EXP_PATH}/results'                      # for model results
        self.APPA_PATH    = f"{self.PERANTO_PATH}/appa-mt/fairseq"          # for appa fairseq training
        self.JSON_PATH    = f"{self.DATA_PATH}/peranto_files"               # contains 3 folders w/ tp config 
        self.OUT_PATH     = f"{self.EXP_PATH}/output"                       # path for generated data 
        self.YAML_FPATH   = f"{self.EXP_PATH}/{self.exp_name}.yaml"         # yaml filepath (filepath so don't os.makedirs)
        self.SH_FPATH     = f"{self.EXP_PATH}/{self.exp_name}.sh"           # sh filepath 

        ### data generator configs
        self.identity        = True                                         # true if training src to src_id
        self.amr_files       = ['amr']
        self.middleman_files = ['middleman']                                # json_path/middleman_files/middleman.json
        self.language_files  = [f"english{''.join(p)}"                      # json_path/language_files/englishSVO.json ...
                                for p in itertools.permutations("SVO")]     
        self.names           = ['SVO', 'SOV', 'VSO', 'VOS', 'OSV', 'OVS']

        self.corp_lens    = [1000 * (2 ** i) for i in range(10)]            # 1000, 2000, ..., 512000
        self.num_cores    = 1                                               # this is probably not used anymore but ok

       
        ### data processor configs 
        self.num_trans    = 2                                               # if 2 takes pairs of languages, 3 triples, ...
        self.train_size   = .8                                              # train proprortion 
        self.test_size    = .1                                              # test proportion
        self.dev_size     = .1                                              # dev proportion

        ### trainer configs- we can add more here if needed 
        self.num_epochs    = 1000
        self.num_gpus      = 2
        self.patience      = 75                                             # num epochs of no bleu improvement before early stopping


        for key, value in kwargs.items():
            setattr(self, key, value)

        self.peranto_tree = PerantoTree(                                   
            json_path       = self.JSON_PATH,                              
            amr_files       = self.amr_files,                                 
            middleman_files = self.middleman_files,                                
            language_files  = self.language_files,                      
            names           = self.names,
            identity        = self.identity
            )

        self.combos = list((itertools.combinations(self.peranto_tree.names, self.num_trans)))

        if self.peranto_tree.identity: # needs identiy map
            self.combos += [(name, f"{name}_id") for name in self.peranto_tree.names]

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

class SVOConfig(AbstractConfig):
    """
    SVOConfig trains pairwise between permutations of svo
    
    """

    @property
    def exp_name(self):
        return 'svo_perm'

    def __init__(self, **kwargs):
        kwargs['identity'] = True
        kwargs['corp_lens'] = [1000 * (2 ** i) for i in range(1, 6)]
        super().__init__(**kwargs)


class VarConfig(AbstractConfig):
    """
    VarConfig tests the variability of model results by generating
    5 datasets under the same configuration, training pairwise, and 
    checking the variability of model results. 
    
    """
    @property
    def exp_name(self):
        return 'data_variability'

    def __init__(self, **kwargs):
        kwargs['corp_lens']      = [2000]
        kwargs['language_files'] = ['englishSVO'] * 5
        kwargs['identity']       = False 
        kwargs['names']          = [f'SVO{i}' for i in range(1,6)]
        super().__init__(**kwargs) 

if __name__ == '__main__':
    pass 

