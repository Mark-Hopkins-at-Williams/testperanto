## Training Pipeline

**Repo Description:** this repository enables one to run various *experiments*. An *experiment* typically consists of training multiple neural machine translation models on testperanto generated artificial language. At a high level there are 4 (distinct) steps: data generation, data splitting, training, and analysis of results. Below is a high level workflow, under that there are more specific instructions for each step, and at the bottom are extra setup instructions.

**High Level Workflow:**
The data_generator.py file enables you to generate raw testperanto data. Then, data_splitter.py enables you to take generated data and compile them into datasets ready to be trained on. You can instantiate any number of models to train on various datasets in trainer.py, and finally you can analyze the results in analysis.ipynb. I'll now go through each of these:

**Data Generation:**

Goal: generate a .sh script that generates testperanto data in parallel via parallel_gen.py

Description: Generation begins with a PerantoTree: a 3 lvl tree structure that resembles a configuration for testperanto (i.e. a tree of amr -> middleman -> language json config files). A Generator is defined by a (P,c) tuple, where P is a PerantoTree and c is a corpus length to generate. 

Usage: 
1. Add a new type of PerantoTree in get_per_tree() to specify another preset generation configuration
2. Define a new Generator G with a clear name (e.g. 'svo_perm'), a PerantoTree, and a corpus length (should be 32/64k)
3. Call G.generate() and run the .sh script (found in experiments/run_outputs/) using a conda venv like jhc5 

If necessary, add new peranto config files in experiments/peranto_configs/{file type}_files. Then add a brief description in
peranto_info.json

Output: running the .sh script mentioned above will add new data to experiments/tp_data with the naming convention {data.name}{generator.name} (e.g. SVO.svo_perm). There will also be "metadata" added to experiments/data_info.json, which looks like this:

{
    "svo_perm": {
        "SVO": [
            "amr",
            "middleman",
            "englishSVO"
        ],
        ...
}

**Data Splitting:**

Goal: take raw testperanto data generated above and split it into datasets that are able to be trained 

Description: Data splitting takes the data above and turns them into Datasets. A Dataset includes train/test/dev sets for src/tgt, and src/tgt can potentially include multiple "languages" (i.e. multilingual support). However, it's annoying to define a ton of Datasets, so there's an abstract Splitter object that takes data and defines ways to split the data into Datasets. 

Usage:
1. Fetch data generated in the Generator step using the fetch_data function
2. Specify a type of Splitter S (or define a new one) to split the fetched data into Datasets
3. Call S.update_metadata()

Output: following the above steps will create folders in experiments/datasets that are in the form fairseq requires to be trained. There will also be "metadata" added to experiments/dataset_info.json, which looks like this:

{
    "svo_pairwise": [
        {
            "SOV_OSV_1.0k": {
                "src": [
                    "SOV.svo_perm"
                ],
                "tgt": [
                    "OSV.svo_perm"
                ],
                "corp_lens": [
                    1000
                ]
                }
            },


**Training:**

Goal: take datasets and train them 

Description: training takes the datasets above and trains them using Models. Models are transformers with slightly varying hyperparameters/architecture size. Everything is done through a Trainer object. 

Usage:
1. Fetch datasets created above using fetch_data()
2. Specify a list of Model objects (each model will be trained on each dataset)
3. Define a Trainer object and call trainer.create_train_script()
4. Run the .sh scripts created (found in run_outputs) using a conda venv w/ fairseq (fairseq.v2 works)

Output: following the above steps will create folders in experiments/results with model results. It will also update results.csv, which looks like this:

exp_name,work_dir,dataset_name,model_name,bleu,chrF2,num_steps,src,tgt,corp_lens,model_arch,num_epochs,patience
svo_perm,XS_OSV_OVS_1.0k,OSV_OVS_1.0k,XS,9.2,44.7,2460.0,['OSV.svo_perm'],['OVS.svo_perm'],[1000],XS,1000,75

As the models are training, you can call python results.py, which will update the csv with the scores (replacing BLEU = None to the actual BLEU score for example).

Finally, you can analyze the data using analysis.ipynb. 

**Setup Instructions:**

Here's the repo layout I'm using (though you can change these easily by just adjusting globals.py):

```
├── src/ 
│ ├── globals.py # global variables
│ ├── data_generator.py # generate raw tp data
│ ├── data_splitter.py # split data into datasets
│ ├── trainer.py # train datasets
│ ├── results.py # update experiment results
│ └── analysis.ipynb # analyze model results
│
└── experiments/ # experimental data
├── run_outputs/ # .sh, .out, .err, .yaml files
├── peranto_configs/ # tp config .json files
│ ├── amr_files/ # amr related files
│ ├── middleman_files/ # middleman related files
│ └── language_files/ # language related files
├── tp_data/ # contains raw tp data
├── datasets/ # contains datasets
├── results/ # model results
├── plots/ # analysis plots
├── results.csv # all model results data
├── peranto_info.json # metadata about peranto_configs
├── data_info.json # metadata about tp_data
├── dataset_info.json # metadata about datasets
└── plot_info.json # metadata about plots
```

Additionally, you need to add a directory appa-mt in testperanto, which just is exactly [https://github.com/Mark-Hopkins-at-Williams/appa-mt/tree/main/fairseq](this) directory, except train.sh/prepare_data.sh are slightly modified (these updated files are located in experiments). 

