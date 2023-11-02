import itertools 
import os 

from config import Config
from data_processor import DataProcessor
from data_generator import DataGenerator

class Trainer:

    def __init__(self, config: Config):
        self.per_tree = config.peranto_tree
        self.num_trans = config.num_trans
        self.corp_lens = config.corp_lens
        self.train_path = config.TRAIN_PATH
        self.results_path = config.RESULTS_PATH
        self.exp_path = config.EXP_PATH
        self.appa_path    = config.APPA_PATH
        self.num_epochs = config.num_epochs

    def format_number(self, num):
        if num >= 1000000:
            return f"{num/1000000:.1f}m"
        elif num >= 1000:
            return f"{num/1000:.1f}k"
        else:
            return str(num)

    def create_train_script(self):
        shell_script = f'{self.exp_path}/train.sh'

        with open(shell_script, 'w') as f: 
            f.write("""#!/bin/sh\n""")
            f.write(f"""
#SBATCH -c 8 # Request 8 CPU cores
#SBATCH -t 2-00:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o {self.results_path}/output.out # File to which STDOUT will be written
#SBATCH -e {self.results_path}/error.err # File to which STDERR will be written
#SBATCH --gres=gpu:2 # Request 2 GPUs\n""")
            
            calls = self.get_calls()

            # Initialize an empty list to store job IDs
            job_ids = []

            for call in calls:
                f.write("OUTPUT=$( " + call.strip() + " )\n")  # Get the output of sbatch command
                f.write("JOB_ID=$(echo $OUTPUT | awk '{print $NF}')\n")  # Extract the job ID
                job_ids.append("$JOB_ID")

                # If there are more than 1 job IDs stored, wait for the earliest job to complete
                if len(job_ids) > 1:
                    f.write(f"""srun -n 1 -c 1 --mem=1M sh -c "while squeue -j {job_ids.pop(0)} | grep -q ' R\| PD'; do sleep 60; done"\n""")
                        # Ensure all jobs complete at the end
            for job_id in job_ids:
                f.write(f"""srun -n 1 -c 1 --mem=1M sh -c "while squeue -j {job_id} | grep -q ' R\| PD'; do sleep 60; done"\n""")
            
            f.write("EOT\n")

    def get_calls(self):
        calls = []
        for corp_len in self.corp_lens:
            for idxs in itertools.combinations(range(len(self.per_tree.names)), self.num_trans):
                folder_name = f"{'_'.join([self.per_tree.names[i] for i in idxs])}_{self.format_number(corp_len)}" 
                data_dir = f"{self.train_path}/{folder_name}"
                work_dir = f"{self.results_path}/{folder_name}"

                ### calls are only things to still be trained so you can call this again if it screws up midway
                if not os.path.exists(work_dir):
                    src = self.per_tree.languages[idxs[0]] 
                    tgt = self.per_tree.languages[idxs[1]] 
                    
                    call = f"sbatch {self.appa_path}/train.sh {work_dir} {data_dir} {src} {tgt} {self.num_epochs}\n"
                    calls.append(call)
        return calls

if __name__ == '__main__':
    config = Config()
    trainer = Trainer(config)
    trainer.create_train_script()