## hyperparam-tuning 

The code in this folder tunes the two hyperparameters of **testperanto**, *strength* and *discount*. 

### Repo Structure: 

├── src                                    
│   ├── treebank.py                                                   <- scrape/read treebank data
│   ├── experiment.py                                                 <- experiment class, used to tune ind. dist.
│   └── peranto_triples.py                                            <- storage class for peranto data 
├── experiment_data                                  
│   ├── parameters                                       
│   │   └── {dist}_params.txt                                         <- parameters tried for {dist}
│   ├── json_data                                                     
│   │   ├── {dist}                                                    
│   │   │   ├── {dist}_amr_s{strength}_d{discount}.json               <- json config file for {dist},s={strength},d={discount}
│   │   │── baseline.json                                             <- baseline json config file 
│   ├── sh_scripts                                
│   │   └── {dist}.sh                                                 <- shell script for {dist}
│   ├── peranto_output                                                
│   │   ├── {dist}
│   │   │   └── peranto_{dist}_s{strength}_d{discount}.txt            <- peranto output for {dist}, s={strength}, d={discount}
│   │   ├── {dist}_modified
│   │   │   └── peranto_{dist}_s{strength}_d{discount}_modified.txt   <- cleaned output for {dist},s={strength},d={discount}
│   ├── plots                              
│   │   └── {dist}_curves.jpg                                         <- singleton prop curve for {dist}
│   ├── mse_results                        
│   │   └── {dist}_mse_results.txt                                    <- mse results for {dist}
│   ├── scraped_treebank.json                                         <- scraped treebank data
│   └── treebank.conllu                                               <- original treebank conllu data 
├── README.md                                                         <- Project description/relevant links

### Results:

          || Results ||
==================================
|| nn         : s = 22, d = .2  ||
|| vb         : s = 22, d = .5  ||
|| nn.arg0    : s = 40, d = .4  ||
|| nn.arg1    : s = 26, d = .6  ||
|| nn.arg0.$y0: s = 34, d = .8  ||
|| nn.arg1.$y1: s = 34, d = .45 ||
==================================

### Description of Treebank Scraping:

We start with experiment_data/treebank.conllu, and in src/treebank.py we first scrape:
- scrape (Subject, Verb, Object) triples and lemmatize, lowercase
- allow (S,V,O) and (S',V,O) (i.e. allow if multiple triples with same verb)
- keep track of stuff that are/aren't pronouns

The way this is done is in data/scraped_treebank.json, which is a json file that 
contains subject, verb, object, subj_pron, obj_pron tuples (subj_pron is True iff subj is pronoun).

There's also a TripleStore class here. This class basically allows you to store and retrieve SVO triples.
The functionality is abstracted away and all you need to do is the following:
    
    store = TripleStore()
    store.get(distribution, pronoun_filter),

Here:
    distribution in [vb, nn, nn.arg0, nn.arg1, nn.arg0.$y0, nn.arg1.$y0] is the distribution
    you want the data for (so for example nn.arg0 will give you a list of subjects)

    pronoun_filter in [both, either, subject, object, None] is what pronouns you need

### Description of Experiments

We want to tune distributions in an hierarchical manner: we only want to tune a distribution if 
its base distribution has been tuned. So there's a global lists, and you can only run experiments
on a distribution if you've tuned ones you should have already tuned.

Each experiment allows you to grid search through a set of strength/discount pairs, for a specific 
distribution, with a specific pronoun filter. 

Once you specify these params (instance variables) you should 
    experiment.setup()

Experiment setup will do a few things:
    (1) create a file in the parameters folder specifying the hyperparameters being searched through
    (2) create a bunch of json config files (for testperanto to call) that specify specific strength/discount
    (3) create a .sh script that will generate testperanto output for each of these json config files

At this point you should run this .sh script on appa- it should create a bunch of testperanto output data 

You should now call experiment.run(), which does the following:
    (1) clean the data- changes from testperanto output data to data similar to the analogous treebank data 
    (2) read in all the now generated data, as well as the treebank data 
    (3) compute mse between treebank and each generated data, save this all to a file 
    (4) create and save an interactive singleton proportion curve 
        (a) this requires computing singleton proportions

Basically, the treebank scraping is probably done, so we just need to fill out the Experiment class
and then start running these experiments on appa. 

Note:
in order to run these function properly, one must create a data directory within the hyperparam-tuning directory with the following sub-directories: json_data, parameters, peranto_output, and sh_scripts. 

