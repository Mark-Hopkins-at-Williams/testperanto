import os 
import numpy as np

from treebank import TripleStore

SRC_PATH  = os.getcwd()
MAIN_PATH = os.path.dirname(SRC_PATH)
PER_PATH  = os.path.dirname(MAIN_PATH)
DATA_PATH = f"{MAIN_PATH}/data"
PARAM_PATH = f"{DATA_PATH}/parameters"
"""
parameters:
    {dist}_params.txt   (nn_params1.txt)

json_configs:
    {dist}:
        {dist}_s{strength}_d{discount}.json

sh_scripts:
    {dist}.sh

peranto_output:
    {dist}:
        {dist}_s{strength}_d{discount}_peranto.txt
"""
class Config:

    def __init__(self, distribution: str=None, input_space=None):
        self.distribution = distribution

        if input_space is None:
            input_space: dict = {
            "Strength" : np.linspace(20, 40, 11),
            "Discount" : np.linspace(0, 1, 21)[:-1]
            }
        self.input_space = input_space
    
class Experiment:
    distributions = [
            'vb', 'nn', 
            'nn.arg0', 'nn.arg1', 
            'nn.arg0.$y0', 'nn.arg1.$y0'
            ]
    bank_store = TripleStore()
    pron_count = (0.6981516025097507, 0.2518229608275394) # subj prop, obj prop

    def __init__(self, config: Config):
        self.initalize(config)
        self.dist          = config.distribution
        self.input_space   = config.input_space 
        self.num_pron      = self.get_num_pron()
        self.param_path    = f"{PARAM_PATH}/{self.dist}_params.txt"
        self.json_path     = f"{DATA_PATH}/{self.dist}"
        self.sh_path       = f"{DATA_PATH}/sh_scripts"
        self.output_path   = f"{DATA_PATH}/{self.dist}"

    def initalize(self, config):
        if not config.distribution in self.distributions:
            raise Exception("config.distribution must be either 'vb', 'nn', 'nn.arg0', 'nn.arg1', 'nn.arg0.$y0', or 'nn.arg1.$y0'")

        if not list(config.input_space.keys()) == ['Strength', 'Discount']:
            raise Exception("config.input_space but have keys Strength/Discount")
        
        for val in config.input_space.values():
            if not isinstance(val, list):
                raise Exception("config.input_space values must be lists")

        ### make sure you run distributions sequentially (use self.distributions)

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
        ### only call after experiment.setup()
        ### peranto data outputs he eat food #... so clean to match desired output 
        ### might need some TripleStore like functionality here (if so make abstract)
        raise NotImplementedError 

    """
    Also visualization.ipynb-like functions to read all data,
    compute the mse and write results to a file (strength discount -> mse)
    then create and save plot 

    this should all be wrapped in a .run() function
    """
    
    def setup(self):
        #self.create_param_space()
        #self.create_json_configs() 
        #self.create_sh_script()
        print(f'Experiment setup completed. Please run shell script on appa.')
        pass 
    
if __name__ == "__main__":
    config = Config('nn')
    exp = Experiment(config)

    exp.get_num_pron()
    
    
"""




Each experiment is an experiment to generate data that can then be used
to visualize/tune the hyperparameters 

You specify a distribution (the distribution to run the experiment on)
and whether or not it's only pronouns or not, as well as an input space 

The experiment class has .steup(), which
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