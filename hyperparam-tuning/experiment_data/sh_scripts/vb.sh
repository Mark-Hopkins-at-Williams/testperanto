#!/bin/sh
        #SBATCH -c 1                # Request 1 CPU core
        #SBATCH -t 0-02:00          # Runtime in D-HH:MM, minimum of 10 mins
        #SBATCH -p dl               # Partition to submit to 
        #SBATCH --mem=10G           # Request 10G of memory
        #SBATCH -o output.out       # File to which STDOUT will be written
        #SBATCH -e error.err        # File to which STDERR will be written
        #SBATCH --gres=gpu:0        # Request 0 GPUs

        JSON_PATH="/mnt/storage/jcheigh/testperanto/hyperparam-tuning/experiment_data/json_data"
        DATA_PATH="/mnt/storage/jcheigh/testperanto/hyperparam-tuning/experiment_data/peranto_output"
        PERANTO_PATH="/mnt/storage/jcheigh/testperanto"

        for json_file in $JSON_PATH/vb_amr_*.json; do
            strength=$(echo $json_file | grep -o -E 's[0-9]+' | sed 's/s//')
            discount=$(echo $json_file | grep -o -E 'd[0-9]+' | sed 's/d//')
            
            python $PERANTO_PATH/scripts/generate.py -c $json_file $PERANTO_PATH/examples/svo/middleman1.json $PERANTO_PATH/examples/svo/english1.json --sents -n 5897 > $DATA_PATH/peranto_vb_s${strength}_d${discount}.txt
        done
        