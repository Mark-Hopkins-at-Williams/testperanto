import yaml
import itertools 
import json

from globals import *

class PerantoTree:
    """
    PerantoTree specifies a 3 level tree for generating testperanto data in parallel. This
    tree resembles the structure of the yaml file to be created/passed to parallel_gen.py.

    amr_files, middleman_files, language_files are lists of json filenames
    names is a list of names to assign to each (amr, mm, lang) triple, where amr in amr_files,
    mm in middleman_files, and lang in language_files.
    """
    def __init__(self,
            amr_files,
            middleman_files,
            language_files,
            names
            ):
        self.amr_paths    = [f'{JSON_PATH}/amr_files/{amr_file}.json' for amr_file in amr_files]
        self.middle_paths = [f'{JSON_PATH}/middleman_files/{mid_file}.json' for mid_file in middleman_files]
        self.lang_paths   = [f'{JSON_PATH}/language_files/{lang_file}.json' for lang_file in language_files]

        self.names        = names 
        self.file_order   = {i : name for i, name in enumerate(self.names)}
        self.data         = [self.amr_paths, self.middle_paths, self.lang_paths]
        self.num_paths    = len(self.amr_paths) * len(self.middle_paths) * len(self.lang_paths)

        file_info     = [[amr, mm, lang] for amr in amr_files for mm in middleman_files for lang in language_files]
        self.name_map = {name : info for (name, info) in zip(names, file_info)}

class Generator:
    """
    A Generator generates data. Generation involves a tuple (P,c), where 
    P is a PerantoTree and c is a corpus size. This class creates a .sh script 
    that generates the data, as well as updates the metadata data_info.json, 
    which has info on data created. 
    """

    def __init__(self, name: str, per_tree: PerantoTree, max_len: int):
        self.name     = name 
        self.per_tree = per_tree
        self.max_len  = max_len

        self.yaml_fpath = f"{RUN_PATH}/{name}.yaml"
        self.sh_fpath   = f"{RUN_PATH}/{name}_gen.sh"
        self.data_fpath = f"{DATA_PATH}/data_info.json" 

    def generate_yaml(self): 
        """Generates yaml file"""
        def format(paths):
            ### how yaml config must look 
            if len(paths) == 1:  
                return paths[0]
            
            return {
                "branch" : {idx + 1 : [path] for idx, path in enumerate(paths)}
                }
        
        per_tree = self.per_tree.data # [[amr_path1, ...], [middle_path1, ...], [lang_path1, ...]]

        data = [format(paths) for paths in per_tree]

        with open(self.yaml_fpath, "w") as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

    def create_sh_script(self):
        """Creates sh script to run generate data assoc. w/ above yaml file"""
        with open(self.sh_fpath, 'w') as f:
            f.write("#!/bin/sh\n")

            f.write(f"""
#SBATCH -c 1 # Request 1 CPU cores
#SBATCH -t 0-10:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o {RUN_PATH}/{self.name}_gen.out # File to which STDOUT will be written
#SBATCH -e {RUN_PATH}/{self.name}_gen.err # File to which STDERR will be written
#SBATCH --gres=gpu:0 # Request 0 GPUs\n""")

            ### call parallel gen and generate data 
            max_len = self.max_len
            f.write(f"python {PERANTO_PATH}/scripts/parallel_gen.py -c {self.yaml_fpath} -n {max_len} -o {TP_DATA_PATH}\n") 

            ### mv data to TP_DATA_PATH with correct naming convention
            for j in range(self.per_tree.num_paths):
                name = self.per_tree.file_order[j]
                f.write(f"mv {TP_DATA_PATH}.{j} {TP_DATA_PATH}/{name}.{self.name}\n")

    def update_metadata(self):
        ### updates data_info.json, which collects info on what the data is (SVO is [amr.json -> mm.json -> englishSVO.json])
        with open(self.data_fpath, 'r') as file:
                data = json.load(file)

        data[self.name] = self.per_tree.name_map

        with open(self.data_fpath, 'w') as file:
            json.dump(data, file, indent=4)

    def generate(self):
        self.generate_yaml()
        self.create_sh_script()
        self.update_metadata()

def get_per_tree(type='svo_perm'):
    """fetch PerantoTree for different types of data generation schemes"""
    if type == 'svo_perm':
        amr_files        = ['amr']
        middleman_files  = ['middleman']                            
        language_files   = [f"english{''.join(p)}"                     
                         for p in itertools.permutations("SVO")]     
        names            = ['SVO', 'SOV', 'VSO', 'VOS', 'OSV', 'OVS']

        per_tree = PerantoTree(
            amr_files = amr_files,
            middleman_files = middleman_files,
            language_files = language_files,
            names = names
            )
    elif type == 'basic_multi':
        """
        Generate 2 fancy english SVO, 1 french, 1 fancy english SOV

        To Train:

        en_svo1 -> en_sov
        en_svo1 + en_svo2 -> en_sov
        en_svo1 + fr -> en_sov
        """
        amr_files = ['amr_fancy']
        middleman_files = ['middleman_fancy']
        language_files = ['english', 'english', 'french', 'english_sov']
        names = ['en_svo1', 'en_svo2', 'fr', 'en_sov']

        per_tree = PerantoTree(
            amr_files = amr_files,
            middleman_files = middleman_files,
            language_files = language_files,
            names = names
            )
    elif type == 'switches':
        amr_files = ['amr_fancy']
        middleman_files = ['middleman_fancy']

        binary_strings = [''.join(bits) for bits in itertools.product('01', repeat=6)]
        language_files = [
            f"lang_{b}" for b in binary_strings
        ]
        names = [
            f's{b}' for b in binary_strings
        ]
        per_tree = PerantoTree(
            amr_files = amr_files,
            middleman_files = middleman_files,
            language_files = language_files,
            names = names
            )

    else:
        raise ValueError("wrong type bro")

    return per_tree 

if __name__ == "__main__":
    per_tree = get_per_tree(type='switches')
    max_len = 32000
    generator = Generator("switches", per_tree, max_len)
    generator.generate()




