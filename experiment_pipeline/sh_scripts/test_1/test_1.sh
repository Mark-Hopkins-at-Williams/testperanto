#!/bin/sh

sbatch <<EOT
#!/bin/sh
#SBATCH -c 1 # Request 1 CPU cores
#SBATCH -t 0-02:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o output.out # File to which STDOUT will be written
#SBATCH -e error.err # File to which STDERR will be written
#SBATCH --gres=gpu:0 # Request 0 GPUs

python ../../../scripts/parallel_gen.py -c experiment_pipeline/yaml_files/test_1/englishOSV_englishOVS.yaml -n 100 -o experiment_pipeline/tesperonto_output/test_1
EOT
                    
sbatch <<EOT
#!/bin/sh
#SBATCH -c 1 # Request 1 CPU cores
#SBATCH -t 0-02:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o output.out # File to which STDOUT will be written
#SBATCH -e error.err # File to which STDERR will be written
#SBATCH --gres=gpu:0 # Request 0 GPUs

python ../../../scripts/parallel_gen.py -c experiment_pipeline/yaml_files/test_1/englishOSV_englishOVS.yaml -n 200 -o experiment_pipeline/tesperonto_output/test_1
EOT
                    
sbatch <<EOT
#!/bin/sh
#SBATCH -c 1 # Request 1 CPU cores
#SBATCH -t 0-02:00 # Runtime in D-HH:MM
#SBATCH -p dl # Partition to submit to
#SBATCH --mem=2G # Request 2G of memory
#SBATCH -o output.out # File to which STDOUT will be written
#SBATCH -e error.err # File to which STDERR will be written
#SBATCH --gres=gpu:0 # Request 0 GPUs

python ../../../scripts/parallel_gen.py -c experiment_pipeline/yaml_files/test_1/englishOSV_englishOVS.yaml -n 300 -o experiment_pipeline/tesperonto_output/test_1
EOT
                    