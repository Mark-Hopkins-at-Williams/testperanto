import yaml
import itertools 
import json

from globals import *

class PerantoTree:
    """
    PerantoTree specifies a 3 level tree for generating testperanto data in parallel. This
    tree resembles the structure of the yaml file to be created/passed to parallel_gen.py. 
    PerantoTree is used to configure a Generator (see below), which generates the data.

    Args:
        amr_files       (list[str]): list of amr.json filenames       (ex: amr, amr_fancy)
        middleman_files (list[str]): list of middleman.json filenames (ex: middleman, middleman_fancy)
        language_files  (list[str]): list of language json filenames  (ex: englishSVO, lang_000000)
        names           (list[str]): list of names to assign to each leaf in tree 

    Important Info:
        - Each amr file should be in experiments/peranto_configs/amr_files/ (and analogous for mm/lang) 
        - names[i] corresponds to file_info[i], where file_info = [(amr, mm, lang) 
                                                                for amr in amr_files 
                                                                for mm in mm_files
                                                                for lang in l_files
                                                                ]
        - Usually you specify a single amr/mm file, with different language generations 

    Example:
    
    >>> amr_files        = ['amr']
    >>> middleman_files  = ['middleman']                            
    >>> language_files   = [f"english{''.join(p)}"                     
                            for p in itertools.permutations("SVO")]     
    >>> names            = ['SVO', 'SOV', 'VSO', 'VOS', 'OSV', 'OVS']
    >>> per_tree = PerantoTree(
            amr_files = amr_files,
            middleman_files = middleman_files,
            language_files = language_files,
            names = names
            )
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
    Generator generates raw testperanto data, and it's configured by a tuple (P,c),
    where P is a PerantoTree and c is a corpus size to generate. The main method 
    of the Generator is .generate(), which does a number of things (see below)

    Args:
        name (str):             Generator name (used as ext in data files, i.e. en.{gen_name})
        per_tree (PerantoTree): PerantoTree to specify what data to generate
        max_len (int):          How much data to generate (default is prob 32/64k)

    Methods:
        generate_yaml   : converts per_tree.data to yaml file that configures parallel_gen.py 
        create_sh_script: creates .sh script {gen_name}_gen.sh (in experiments/run_outputs) to call parallel_gen.py
        update_metadata : updates experiments/data_info.json, which includes data about the generated data 
        generate        : calls the above 3 method sequentially 

    Important Info:
        - metadata allows you to remember what each generation scheme consists of, and it takes the following form:
            {
            "svo_perm": {
                "SVO": [
                    "amr",
                    "middleman",
                    "englishSVO"
                ],
                ...
            this tells us there's a Generator called svo_perm, and that SVO is the name associated with the generation
            amr -> middleman -> englishSVO (amr -> mm -> lang file)
        - You should ever only train on data created in the same generation scheme (else the sentences don't match). In
        other words, you should only train between two data files (say en.{gen_name1} and sp.{gen_name2}) if they have the 
        same extension ({gen_name1} == {gen_name2})
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
        ### updates data_info.json
        with open(self.data_fpath, 'r') as file:
                data = json.load(file)

        data[self.name] = self.per_tree.name_map

        with open(self.data_fpath, 'w') as file:
            json.dump(data, file, indent=4)

    def generate(self):
        ### main method to call
        self.generate_yaml()
        self.create_sh_script()
        self.update_metadata()

def get_per_tree(type='svo_perm'):
    """
    Fetches a different type of PerantoTree (just so you don't need to keep rewriting).
    When you want to add a new PerantoTree just add another elif clause 
    """
    if type == 'svo_perm':
        """
        Generates basic 3 word language with all 3! = 6 word order permutations
        
        We want to train pairwise here to see basic impact on word order
        """
        amr_files        = ['amr']
        middleman_files  = ['middleman']                            
        language_files   = [f"english{''.join(p)}"                     
                         for p in itertools.permutations("SVO")]     
        names            = ['SVO', 'SOV', 'VSO', 'VOS', 'OSV', 'OVS']

    elif type == 'basic_multi':
        """
        Generates 2 enSVO, 1 fr, 1 enSOV (all fancy)

        We want to train 
        en_svo1           -> en_sov
        en_svo1 + en_svo2 -> en_sov
        en_svo1 + fr      -> en_sov

        to see if there's synergy 
        """
        amr_files       = ['amr_fancy']
        middleman_files = ['middleman_fancy']
        language_files  = ['english', 'english', 'french', 'english_sov']
        names           = ['en_svo1', 'en_svo2', 'fr', 'en_sov']

    elif type == 'switches':
        """
        Generates 2^6 = 64 fancy languages, each configured by 6 binary switches (see White/Cotterell 2021)

        Let src = s011101 (switches for en), tgt = s000000 (switches for japn). We want to train

        src + aux -> tgt, where aux is anything except for tgt.

        This is actually a mistake on my part; we really should generate two instances of s011101 to change the vocab.
        """
        amr_files       = ['amr_fancy']
        middleman_files = ['middleman_fancy']
        binary_strings  = [''.join(bits) for bits in itertools.product('01', repeat=6)]
        language_files  = [f"lang_{b}" for b in binary_strings]
        names           = [f's{b}' for b in binary_strings]

    elif type == 'new_switches':
        """
        Does the same as switches except generates two instances of s011101. This generation also was used to 
        completely change the vocab (don't use "regular" words like preset pronouns/prepositions). To do this 
        we use map_to_burmese (see helper.py)
        """
        amr_files       = ['amr_fancy']
        middleman_files = ['middleman_fancy']
        binary_strings  = [''.join(bits) for bits in itertools.product('01', repeat=6)]
        language_files  = [f"lang_{b}" for b in binary_strings] + ['lang_011101'] 
        names           = [f's{b}' for b in binary_strings] + [f's011101_2']

    else:
        raise ValueError("wrong name bro")

    per_tree = PerantoTree(
        amr_files       = amr_files,
        middleman_files = middleman_files,
        language_files  = language_files,
        names           = names
        )

    return per_tree 

if __name__ == "__main__":
    per_tree  = get_per_tree(type='new_switches')
    max_len   = 32000
    generator = Generator("new_switches", per_tree, max_len)
    generator.generate()




