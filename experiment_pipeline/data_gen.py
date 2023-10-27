# class for generating 
import yaml
import itertools
import os


class DataGen:
    """
    Class for generating testperanto data for translations. 
    """

    def __init__(self, lang_list, corp_sizes, num_translations, experiment_name):
        """
        lang_list is a list of paths to json files used in yaml file generation. 
        Please note that these file paths MUST BE FROM THE ROOT DIRECTORY. for example: experiment_pipeline/tesperonto_files/englishOSV.json

        corp_sizes is a list of ints that denotes the desired sizes for corpi

        num_translations is the number of languages for which translations will be produced at a given time.
        for example, if num translations is 2, then we will generate translation data for every pair of languages in the lang_list.
        if it were 3, we would generate translations for every tripple, and so on. 

        Please note that there a lot of path variables right now. If anyone has a better way to 1) keep track of root paths in a way that 
        preserves the funcitonality of the _generate_filemap function and 2) still allows things to be run from root while also allowing for checking 
        the existance of directories, that would be fantastic. 
        """
        self.lang_list = lang_list
        self.corp_sizes = corp_sizes
        self.num_translations = num_translations
        self.root_to_current = "experiment_pipeline"
        self.experiment_name = experiment_name
        self.yaml_path = f"yaml_files/{self.experiment_name}"
        self.external_yaml_path = f"experiment_pipeline/{self.yaml_path}"
        self.testperanto_path = f"experiment_pipeline/tesperonto_files" # path to access all tesperonto files that are used. 
        self.output_path = f"experiment_pipeline/tesperonto_output/{self.experiment_name}"
        self.internal_output_path = f"tesperonto_output/{self.experiment_name}"
        self.sh_path = f"sh_scripts/{self.experiment_name}"
        self.file_map = self._generate_filemap()
        # initialize testperonto_files directory
        #os.makedirs(self.testperanto_path, exist_ok = True)
    
    def _generate_filemap(self):
        """
        naming conventions will be important in this class. Having a data structure that maps source file groups from the lang list to file names 
        will be useful. This function creates a dictionary of {tuple(string): string}, where tuple(string) denotes the combination of source files for a file name, 
        and the string that the tuple maps to is the file name.
        """
        maping = dict()
        combos = list(itertools.combinations(self.lang_list, self.num_translations)) # list of tuples of file names that go together for translation
        for combo in combos:
            list_of_paths = list(combo)
            file_name = ""
            for i, path in enumerate(list_of_paths):
                striped_name = path[len(self.testperanto_path) + 1:-5]
                if i == len(list_of_paths) - 1:
                    file_name = file_name + striped_name
                else:
                    file_name = file_name + striped_name + "_"
            maping[combo] = file_name
        return maping

    def get_file_map(self):
        return self.file_map
            
            
    def set_lang_list(self, lang_list):
        self.lang_list = lang_list
        # add code to reset state to where it should be if this occures 

    def set_corp_sizes(self, corp_sizes):
        self.lang_list = lang_list
        # add code to reset state to where it should be if this occures
    
    def generate_yaml(self): # working 
        """
        Generates yaml files for all translation combinations. 
        """
        # ensure path exists 
        os.makedirs(self.yaml_path, exist_ok = True)
        
        for key in self.file_map:
            paths_in_yaml = list(key)
            data = [
                "examples/svo/amr.json", # potentially add flexibility to modify. TODO: change to internal path 
                "examples/svo/middleman.json", # potentially add flexibility to modify TODO: change to internal path
                {
                    "branch": {idx + 1: [path] for idx, path in enumerate(paths_in_yaml)}
                }
            ]
            yaml_name = self.file_map[key] + ".yaml"
            
            with open(f"{self.yaml_path}/{yaml_name}", "w") as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)

    def create_sh_script(self, num_cores = 32): # working
    """
    Creates a .sh script, that when run from the TESPERONTO ROOT DIRECTORY, will generate text 
    in paralell (default is 32 cores) for all translation + size combinations. 
    """

        # make sure output path exists 
        os.makedirs(self.internal_output_path, exist_ok = True) # we might want to put this in init... not sure
        os.makedirs(self.sh_path, exist_ok = True)

        with open(f"{self.sh_path}/{self.experiment_name}.sh", 'w') as f:

            # Write the initial part of the script
            f.write("""#!/bin/sh\n""")

            # initialize the slerm stuff 
            f.write(f"""
#SBATCH -c {num_cores} # Request {num_cores} CPU cores
#SBATCH -t 0-02:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o output.out # File to which STDOUT will be written
#SBATCH -e error.err # File to which STDERR will be written
#SBATCH --gres=gpu:0 # Request 0 GPUs\n""")

            # Begin the parallel section
            f.write(f"parallel --jobs {num_cores} <<EOT\n")

            # Loop through each combination and append to the script
            for key in self.file_map:
                file_name = self.file_map[key]
                yaml_file_name = f"{self.external_yaml_path}/{file_name}.yaml"
                for num_sentences in self.corp_sizes:
                    #os.makedirs(f"{self.internal_output_path}/{num_sentences}")
                    # Construct the Python call for each job and add it to the commands to be run by parallel
                    python_call = f"python scripts/parallel_gen.py -c {yaml_file_name} -n {num_sentences} -o {self.output_path}/{num_sentences}\n"
                    f.write(python_call)

            # End the parallel section
            f.write("EOT\n")
    
    def run_sh_script(self):
        pass
    
    def train_test_split(self):
        pass


if __name__ == "__main__":
    root_to_current = "experiment_pipeline"
    tesperonto_path = "tesperonto_files"
    lang_list = [f"{root_to_current}/{tesperonto_path}/englishOSV.json", f"{root_to_current}/{tesperonto_path}/englishOVS.json"]
    corp_sizes = [100, 200, 300]
    gen = DataGen(lang_list, corp_sizes, 2, "test_1")
    gen.generate_yaml()
    gen.create_sh_script()




