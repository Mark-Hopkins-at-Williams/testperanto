import os 
from dataclass import dataclass
import numpy as np

from treebank import TripleStore

SRC_PATH  = os.getcwd()
MAIN_PATH = os.path.dirname(SRC_PATH)
PER_PATH  = os.path.dirname(MAIN_PATH)
DATA_PATH = f"{MAIN_PATH}/data"
PARAM_PATH = f"{DATA_PATH}/parameters"

@dataclass
class Config:
    distribution: str = None
    input_space: dict = {
        "Strength" : np.linspace(20, 40, 11),
        "Discount" : np.linspace(0, 1, 21)[:-1]
        }
    only_pronouns: bool = False
    
class Experiment:
    distributions = [
            'vb', 'nn', 
            'nn.arg0', 'nn.arg1', 
            'nn.arg0.$y0', 'nn.arg1.$y0'
            ]

    def __init__(self, config: Config):
        self.initalize_params(config)
        self.dist          = config.distribution
        self.input_space   = config.input_space 
        self.only_pronouns = config.only_pronouns # change to self.pron_filter
        self.run           = self.get_run()
        self.param_path    = f"{PARAM_PATH}/..."
        self.json_path     = f"{DATA_PATH}/..."
        self.sh_path       = f"..."
        self.output_path   = f"..."
        self.bank_store    = TripleStore()

    def initalize(self, config):
        if not config.distribution in self.distributions:
            raise Exception("config.distribution must be either 'vb', 'nn', 'nn.arg0', 'nn.arg1', 'nn.arg0.$y0', or 'nn.arg1.$y0'")

        if not list(config.input_space.keys()) == ['Strength', 'Discount']:
            raise Exception("config.input_space but have keys Strength/Discount")
        
        for val in config.input_space.values():
            if not isinstance(val, list):
                raise Exception("config.input_space values must be lists")

        ### make sure you run distributions sequentially (use self.distributions)

    def get_run_num(self):
        ### returns run num (check which paths already exist)
        raise NotImplementedError

    def create_param_space(self):
        ### writes parameters- very similar to write_input_space in experiment.py 
        raise NotImplementedError

    def create_json_configs(self):
        ### similar to create_json_files()- there's a chance this need to be an 
        ### abstract class and this an abstract method, but I think we can avoid that 
        ### also this might need to be broken into 2 functions (pronoun adjusted here)
        ### since pronouns adjusted need a TripleStore.get(dist, pronoun_filter) to fetch data 
        raise NotImplementedError

    def create_sh_script(self):
        ### creates a .sh script that, when ran, will create testperanto generated output 
        ### using the json files created above 
        raise NotImplementedError 
    
    def clean_peranto_data(self):
        ### only call after experiment.run()
        ### peranto data outputs he eat food #... so clean to match desired output 
        ### might need some TripleStore like functionality here (if so make abstract)
        raise NotImplementedError 

    def run(self):
        self.create_param_space()
        self.create_json_configs()
        self.create_sh_script()
        print(f'Experiment setup completed. Please run shell script on appa.')


    

"""
Each experiment is an experiment to generate data that can then be used
to visualize/tune the hyperparameters 

You specify a distribution (the distribution to run the experiment on)
and whether or not it's only pronouns or not, as well as an input space 

The experiment class has .run(), which
    will create a {dist}_pron_{if_pronouns}_params{num_runs}.txt file with parameters in parameters folder
    will create a amr_s{strength}_d{discount}.json file in {dist}_pron{if_pronouns}/run_{num_runs} folder 
    for each strength/discount pair (also adjusts pronoun so needs to learn distribution from TripleStore)
    also creates shell script?? 

After this one would use appa and run the experiment by just calling the created shell script 
Then you could call the experiment.clean_data(), which will take the .txt file created and edit it to look
exactly like associated treebank data
"""

"""
class Experiment:

    distribution = 'vb'
    

global_distributions = [vb, nn, nn.arg0, nn.arg1, nn.arg0.$y0, nn.arg1.$y0]
    define input space
    generate parameters.txt

(2) define global distribution sequence
    - global verb -> global noun -> ...
(3) for distribution in global_distributions
    - create input space (maybe fancy way of finding)
    - define function that edits json 
    - create json config files
    - define function that edits bash
    - create bash script
    - find optimal parameters and set this from now on!

"""