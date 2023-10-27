## Training Pipeline 

This subdirectory aims to be a flexible end to end training pipeline, which will allow us to conduct various experiments on tranformer based models. Here's the high level pipeline:
1. Generate data:
    - data generation is done by running the testperanto/scripts/parallel_gen.py file, which allows one to generate artificial language in parallel
2. Data processing:
    - data processing is basically preparing the generated data to be trained on. This can be anything from taking raw testperanto output and cleaning it, train/test/val splits, etc.
3. Model/training configs:
    - this is just specifying configurations for the model/how its trained.
4. Processing/analyzing performance:
    - once the model is trained we can evaluate/analyze the performance of the model

    






arallel_gen.py:

- takes a YAML file that specifies how to generate multiple languages in parallel, which allows it to be the same meaning
- Thus, the y/z variables are the same, it’s just the voicebox mapping tow ords

Take amr

use parallel gen to generate 6 branches (english.json to svo, sov, …)

- generate a lot
- train, test, validation (dev)
- train transformer and generate curve with train_size (n) on x axis and test bleu/chrf on y axis for each SVO → SOV pair (generate same number of test/dev for each)
- train such that epochs on x axis and validation loss has elbow (doesn’t overfit)
- fairseq for training
- 

model

parallel generation