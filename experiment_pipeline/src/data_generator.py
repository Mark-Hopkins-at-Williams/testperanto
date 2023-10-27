import yaml
import subprocess 
import time
from config import Config

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
            
        data = [format(paths) for paths in self.per_tree.data]

        with open(self.yaml_fpath, "w") as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

    def create_sh_script(self): 
        """
        Creates a .sh script, that when run from the TESPERONTO ROOT DIRECTORY, will generate text 
        in paralell (default is 32 cores) for all translation + size combinations. 
        """
        def format_number(num):
            if num >= 1000000:
                return f"{num/1000000:.1f}m"
            elif num >= 1000:
                return f"{num/1000:.1f}k"
            else:
                return str(num)

        with open(self.sh_fpath, 'w') as f:

            # Write the initial part of the script
            f.write("""#!/bin/sh\n""")

            # initialize the slurm stuff 
            f.write(f"""
#SBATCH -c {self.num_cores} # Request {self.num_cores} CPU cores
#SBATCH -t 0-02:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o {self.output_path}/output.out # File to which STDOUT will be written
#SBATCH -e {self.output_path}/error.err # File to which STDERR will be written
#SBATCH --gres=gpu:0 # Request 0 GPUs\n""")

            # Begin the parallel section
            f.write(f"parallel --jobs {self.num_cores} <<EOT\n")

            for num_sent in self.corp_lens:
                formatted_num = format_number(num_sent)
                # Construct the Python call for each job and add it to the commands to be run by parallel
                python_call = f"python {self.tp_path}/scripts/parallel_gen.py -c {self.yaml_fpath} -n {num_sent} -o {self.output_path}/{self.exp_name}{formatted_num}\n"
                f.write(python_call)

            # End the parallel section
            f.write("EOT\n")
    
    def run_sh_script(self):
        # submit the job via slurm
        result = subprocess.run(['sbatch', self.sh_fpath], capture_output=True, text=True)
        job_id = result.stdout.split()[-1]  # get the job ID from the output
        return job_id

    def monitor_job(self, job_id):
        # assuming 35it/s, time in sec ~ sum(corp_lens)/35
        est_time = sum(self.corp_lens) // 35

        while True:
            # check the job status using squeue
            squeue_result = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
            if job_id not in squeue_result.stdout:
                # job has completed
                break
            else:
                # wait for a while before checking again
                time.sleep(max(est_time // 10, 1))

    def generate(self):
        self.generate_yaml()
        self.create_sh_script()
        job_id = self.run_sh_script() 
        self.monitor_job(job_id)
        self.clean_data()

if __name__ == "__main__":
    config = Config()
    data_gen = DataGenerator(config)
    data_gen.generate()








