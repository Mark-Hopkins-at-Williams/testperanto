import time 

from config import Config
from data_generator import DataGenerator
from data_processor import DataProcessor
from trainer import Trainer

def main():
    start = time.time()
    config = Config()
    data_gen = DataGenerator(config)
    data_gen.generate()
    mid1= time.time()
    print(f"Generation took {round(mid1 -start, 3)} seconds...")
    data_proc = DataProcessor(config)
    data_proc.process()
    mid2 = time.time()
    print(f"Processing took {round(mid2-  mid1, 3)} seconds...")
    #trainer = Trainer(config)
    #trainer.train()
    #print(f"Entire process took {round(time.time() - start, 3)} seconds...")


if __name__ == '__main__':
    main()