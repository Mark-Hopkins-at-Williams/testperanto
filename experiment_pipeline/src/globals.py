import os

""" PATH CONFIGS """
SRC_PATH     = os.getcwd()                        # src folder
MAIN_PATH    = os.path.dirname(SRC_PATH)          # main experiment_pipeline path
PERANTO_PATH = os.path.dirname(MAIN_PATH)         # testperanto main path
DATA_PATH    = f"{MAIN_PATH}/experiments"         # all data 
RUN_PATH     = f"{DATA_PATH}/run_outputs"         # scripts to run (.sh/.err/.out/.yaml)
TP_DATA_PATH = f"{DATA_PATH}/tp_data"             # generated testperanto data
RESULTS_PATH = f"{DATA_PATH}/results"             # model results
PLOT_PATH    = f"{DATA_PATH}/plots"               # model plots
DATASET_PATH = f"{DATA_PATH}/datasets"            # processed datasets
APPA_PATH    = f"{PERANTO_PATH}/appa-mt/fairseq"  # appa fairseq training
JSON_PATH    = f"{DATA_PATH}/peranto_configs"     # testperanto config files

""" DATA CONFIGS """
TEST_SIZE    = 1000                               # test sentences per dataset
DEV_SIZE     = 1000                               # dev sentences per dataset 

""" MODEL CONFIGS """
NUM_GPUS     = 2                                  # num gpus
NUM_EPOCHS   = 300                                # num epochs   (can be overwritten)
PATIENCE     = 30                                 # num patience (can be overwritten)
MODEL_SIZE   = "XS"                               # model size   (can be overwritten)