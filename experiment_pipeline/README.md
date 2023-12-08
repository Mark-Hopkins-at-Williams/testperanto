## Training Pipeline 

**Repo Description:** this repository enables one to run various *experiments*. An *experiment* typically consists of training multiple neural machine translation models on testperanto generated artificial language. For a high level workflow look below, and for more granular instructions go straight to the bottom.

**High Level Workflow:**
- Configure a new experiment by adding a subclass to config.py
- Call data_generator.py, which creates a .sh script to generate testperanto data
- Run the .sh script and clean/split the data using data_processor.py
- Call trainer.py to create multiple .sh training scripts
- After training, call results.py to create a pd.DataFrame with results
- Analyze the data in analysis.ipynb 

**Repo Layout:**
experiment_pipeline/                                          
├── experiment_data/                                      contains all data
│   ├── experiments/                                      
│   │   └── svo_perm/                                     ex: svo permutation experiment
│   │       ├── results/                                  model results/data
│   │       │   └── SVO_SOV_2.0k/                         ex: 2k corp len model trained from SVO to SOV
│   │       │       ├── scores/                           scores of model (bleu, etc.)
│   │       │       ├── translations/                     translations (src, hyp, ref)
│   │       ├── example_outputs.txt                       example translations 
│   │       └── results.csv                               csv of model results
│   └── peranto_files/                                      
│       └── amr_files/                                    amr json files for tp
├── src/
│   ├── analysis.ipynb                                    model results analysis
│   ├── config.py                          
│   ├── results.py                                        creates results.csv
│   ├── data_generator.py                             
│   ├── data_processor.py       
│   └── trainer.py  
└── README.md

**Specific Instructions:**

Important Notes:
- for each file, make sure config is set to whatever config you are using (config = HiMark())
- to run the data_gen .sh script, you need testperanto (conda activate jhc5 works)
- to run the train .sh script(s), you need fairseq (conda activate fairseq.v2 works)

1. Create config subclass
    - go to config.py and create a new subclass: say class HiMark(AbstractConfig):
    - add a function def exp_name(self) which returns the name of the experiment (here name is "mark")
    - in the init override any default params you want (look at other subclasses for examples)
    - add this new class to the __all__ list
2. Call data_generator.py
    - call data_generator.py 
    - this file will create a .sh script that calls tp/parallel_gen.py in order to create the data
    - this will create a mark.sh script in experiment_data/experiments/mark/
    - sbatch mark.sh (with default params prob takes <1hr, but you can't see progress until it finishes)
3. Call data_processor.py
    - make sure that the data has been fully generated (you can check by looking at output folder)
    - call data_processor.py, which should (immediately) clean/split the generated data
4. Call trainer.py
    - call trainer.py, which will create config.num_gpus training scripts (one per gpu)
    - these will be called train0.sh, train1.sh (located in experiment_data/experiments/mark)
    - sbatch both (conda activate fairseq.v2 works)
5. Call results.py
    - one the models are trained (or you can do it during), call results.py
    - this will create a results.csv with model results 
6. Analyze data in analysis.ipynb
    - Go to analysis.ipynb and add an elif to get_data(): elif exp_name == 'mark': config = HiMark()
    - Use built in data visualizion functionality to analyze (look at notebook for usage)
