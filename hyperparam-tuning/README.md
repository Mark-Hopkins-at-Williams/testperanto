## hyperparam-tuning 

This folder tunes the two hyperparameters of **testperanto**, *strength* and *discount*. 


change paths
don't pay attention to SVO order
singleton verb distribution then noun then next 
ignore arg2 (indirect object)

transformer training script 

class Experiment:

    distribution = 'vb'
    

global_distributions = [vb, nn, nn.arg0, nn.arg1, nn.arg0.$y0, nn.arg1.$y0]
    define input space
    generate parameters.txt

{Strength :[ ....]}

class Experiment:

    dis



(2) define global distribution sequence
    - global verb -> global noun -> ...
(3) for distribution in global_distributions
    - create input space (maybe fancy way of finding)
    - define function that edits json 
    - create json config files
    - define function that edits bash
    - create bash script
    - find optimal parameters and set this from now on!
(4) visualization/tuning
    - for distribution in distributions
    - run visualization ... 

Do once with only pronouns and pronouns param = 1.0
otherwise with everything with subject pronoun/object pronoun based on initial statistics 

Wants:

(1) scraped data
    - easily access all triples, all SV, all OV, all S, all V, all O, all N, all N-pronoun etc. 
(2) class 
