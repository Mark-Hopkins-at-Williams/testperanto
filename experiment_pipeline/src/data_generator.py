import yaml
import subprocess 
import time
from config import Config
from helper import format_number

class DataGenerator:
    """
    Class for generating testperanto data for translations. 
    """

    def __init__(self, config: Config):
        self.per_tree = config.peranto_tree 
        self.corp_lens = config.corp_lens
        self.yaml_fpath = config.YAML_FPATH
        self.sh_fpath = config.SH_FPATH
        self.num_cores = config.num_cores
        self.output_path = config.OUT_PATH
        self.tp_path = config.PERANTO_PATH
        self.exp_name = config.exp_name
        self.file_order = self.per_tree.file_order
        self.num_paths  = self.per_tree.num_paths

    def generate_yaml(self): 
        """
        Generates yaml file
        """
        def format(paths):
            if len(paths) == 1:
                return paths[0]
            
            return {
                "branch" : {idx + 1 : [path] for idx, path in enumerate(paths)}
                }
        
        per_tree = self.per_tree.data 

        data = [format(paths) for paths in per_tree]

        with open(self.yaml_fpath, "w") as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

    def create_sh_script(self):
        """
        Modifies the script to generate only the largest corpus length in parallel and
        then splits it into smaller lengths as specified in self.corp_lens. It handles
        multiple output files from parallel_gen.py and dynamically determines the range
        of j based on self.num_paths.
        """

        with open(self.sh_fpath, 'w') as f:
            # Write the initial part of the script
            f.write("#!/bin/sh\n")

            # Initialize the slurm stuff 
            f.write(f"""
#SBATCH -c {self.num_cores} # Request {self.num_cores} CPU cores
#SBATCH -t 0-10:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o {self.output_path}/output.out # File to which STDOUT will be written
#SBATCH -e {self.output_path}/error.err # File to which STDERR will be written
#SBATCH --gres=gpu:0 # Request 0 GPUs\n""")

            # Calculate the chunk size for parallel generation
            max_corp_len = self.corp_lens[-1]  # largest one 
            chunk_size = max_corp_len // self.num_cores

            # Begin the parallel section
            f.write(f"parallel --jobs {self.num_cores} <<EOT\n")
            
            for i in range(self.num_cores):
                # Call parallel_gen.py to generate a chunk of the largest corpus size
                temp_file_base = f"{self.output_path}/temporary/temp_{i}"
                python_call = f"python {self.tp_path}/scripts/parallel_gen.py -c {self.yaml_fpath} -n {chunk_size} -o {temp_file_base}\n"
                f.write(python_call)

            # End the parallel section
            f.write("EOT\n")

            # generate the script for aggregating and splitting files- group by j (not by which core)
            for j in range(self.num_paths):
                f.write(f"cat {self.output_path}/temporary/temp_*.{j} > {self.output_path}/{self.exp_name}{format_number(max_corp_len)}.{self.file_order[j]}\n")
                for corp_len in self.corp_lens[:-1]:
                    # pipe first corp_len to new file 
                    f.write(f"head -n {corp_len} {self.output_path}/{self.exp_name}{format_number(max_corp_len)}.{self.file_order[j]} > {self.output_path}/{self.exp_name}{format_number(corp_len)}.{self.file_order[j]}\n")

            # Delete the temporary files
            f.write(f"rm -f {self.output_path}/temporary/temp_*.*\n")

    def generate(self):
        self.generate_yaml()
        self.create_sh_script()

if __name__ == "__main__":
    config = Config()
    data_gen = DataGenerator(config)
    data_gen.generate()







