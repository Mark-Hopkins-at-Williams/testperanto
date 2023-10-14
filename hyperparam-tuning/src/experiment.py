import os 
import numpy as np
import json
import matplotlib.pyplot as plt 

from treebank import TripleStore
from peranto_triples import PerantoTripleStore

SRC_PATH  = os.getcwd()
MAIN_PATH = os.path.dirname(SRC_PATH)
PER_PATH  = os.path.dirname(MAIN_PATH)
DATA_PATH = f"{MAIN_PATH}/data"
JSON_PATH = f"{DATA_PATH}/json_data"
PARAM_PATH = f"{DATA_PATH}/parameters"

class Config:

    def __init__(self, distribution: str=None, input_space=None):
        self.distribution = distribution

        if input_space is None:
            input_space: dict = {
            "Strength" : list(np.linspace(20, 40, 11)),
            "Discount" : list(np.linspace(0, 1, 21)[:-1])
            }
        self.input_space = input_space
    
class Experiment:
    distributions = [
            'nn', 'vb',
            'nn.arg0', 'nn.arg1', 
            'nn.arg0.$y0', 'nn.arg1.$y0'
            ]

    def __init__(self, config: Config):
        self.initalize(config)
        self.dist          = config.distribution
        self.input_space   = config.input_space 
        self.num_pron      = (0.6981516025097507, 0.2518229608275394) # subj prop, obj prop
        self.store         = TripleStore()
        self.param_path    = f"{PARAM_PATH}/{self.dist}_params.txt"
        self.json_dict     = f"{JSON_PATH}"
        self.json_path     = f"{JSON_PATH}/{self.dist}"
        self.sh_path       = f"{DATA_PATH}/sh_scripts"
        self.output_path   = f"{DATA_PATH}/peranto_output"
        self.mse_path      = f"{DATA_PATH}/mse_results"
        self.plot_path     = f"{DATA_PATH}/plots"

    def initalize(self, config):
        # check config
        if not config.distribution in self.distributions:
            raise Exception("config.distribution must be either 'vb', 'nn', 'nn.arg0', 'nn.arg1', 'nn.arg0.$y0', or 'nn.arg1.$y0'")

        if not list(config.input_space.keys()) == ['Strength', 'Discount']:
            raise Exception("config.input_space but have keys Strength/Discount")
        
        for val in config.input_space.values():
            if not isinstance(val, list):
                print(val)
                print(type(val))
                raise Exception("config.input_space values must be lists")

        ### make sure you run distributions sequentially (use self.distributions)

    def create_param_space(self):
        """
        Given input space {"Strength" : [strengths], "Discount" : [discounts]}
        this function creates a parameters.txt file that contains all (S, D) pair
        """
        # generate pairs
        inputs = self.input_space
        pairs = [(strength, discount) for strength in inputs["Strength"] for discount in inputs["Discount"]]

        # Create parameters.txt file and write in paramaters
        with open(self.param_path, "w") as f:
            for pair in pairs:
                f.write(f"{pair[0]}, {pair[1]}\n")

    def create_json_configs(self, base_file = "amr1.json"):
        """
        Uses output from create_param_space() input space (create_input_space()) and 
        for each (S,D) pair copies the amr1.json file and changes the 
        strength/discount to (S,D). All of this is saved in a folder of json files
        """

        json_file_to_read = f"{self.json_dict}/{base_file}"

        with open(self.param_path, "r") as f:
            lines = f.readlines()
            parameters = [(float(line.split(",")[0].strip()), float(line.split(",")[1].strip())) for line in lines]
        
        # Read the original JSON content
        with open(json_file_to_read, "r") as f:
            json_content = json.load(f)

        for strength, discount in parameters:
            # modify pyor dists as appropriate
            for dist in json_content["distributions"]:
            # modify appropriate paramaters, according to the distribution
                if dist["name"] == self.dist:
                    dist['strength'] = strength
                    dist['discount'] = discount 
            else: # for else hits iff above if statement never hits
                print(f"Couldn't find distribution: {self.dist}")
                """
                I think the above works fine 

                if self.dist == "vb" and dist["name"] == "vb":
                    dist["strength"] = strength
                    dist["discount"] = discount
                elif self.dist == "nn" and dist["name"] == "nn":
                    dist["strength"] = strength
                    dist["discount"] = discount
                elif self.dist == "nn.arg0" and dist["name"] == "nn.arg0":
                    dist["strength"] = strength
                    dist["discount"] = discount
                elif self.dist == "nn.arg1" and dist["name"] == "nn.arg1":
                    dist["strength"] = strength
                    dist["discount"] = discount
                elif self.dist == "nn.arg0.$y0" and dist["name"] == "nn.arg0.$y0":
                    dist["strength"] = strength
                    dist["discount"] = discount
                elif self.dist == "nn.arg1.$y0" and dist["name"] == "nn.arg1.$y0":
                    dist["strength"] = strength
                    dist["discount"] = discount
                else: 
                    pass # cases do not encompase everything 
                """

            # set proportion of pronouns as appropriate  
            for rule in json_content["rules"]:
                if rule["rule"] == "$qnn.arg0.$y1 -> (inst nn.$y1)":
                    rule["base_weight"] = float(1 - self.num_pron[0])
                if rule["rule"] == "$qnn.arg0.$y1 -> (inst pron.$z1)":
                    rule["base_weight"] = float(self.num_pron[0])
                if rule["rule"] == "$qnn.arg1.$y1 -> (inst nn.$y1)":
                    rule["base_weight"] = float(1-self.num_pron[1])
                if rule["rule"] == "$qnn.arg1.$y1 -> (inst pron.$z1)":
                    rule["base_weight"] = float(self.num_pron[1])
            
            # Save the modified JSON content to a new file
            new_file_name = f"{self.json_dict}/{self.dist}_amr_s{int(strength)}_d{int(discount*100)}.json"
            with open(new_file_name, "w") as f:
                json.dump(json_content, f, indent=4)

    def create_sh_script(self, json_path=None, data_path=None, peranto_path=None):
        """
        runs the .sh script created by create_sh_script()
        """
        if json_path == None:
            json_path=self.json_dict
        if data_path == None:
            data_path=self.output_path
        if peranto_path == None:
            peranto_path=PER_PATH # this prevents errors when moving users

        script_name = f"{self.sh_path}/{self.dist}.sh"

        shell_script_content = f"""#!/bin/sh
        #SBATCH -c 1                # Request 1 CPU core
        #SBATCH -t 0-02:00          # Runtime in D-HH:MM, minimum of 10 mins
        #SBATCH -p dl               # Partition to submit to 
        #SBATCH --mem=10G           # Request 10G of memory
        #SBATCH -o output.out       # File to which STDOUT will be written
        #SBATCH -e error.err        # File to which STDERR will be written
        #SBATCH --gres=gpu:0        # Request 0 GPUs

        JSON_PATH="{json_path}"
        DATA_PATH="{data_path}"
        PERANTO_PATH="{peranto_path}"

        for json_file in $JSON_PATH/{self.dist}_amr_*.json; do
            strength=$(echo $json_file | grep -o -E 's[0-9]+' | sed 's/s//')
            discount=$(echo $json_file | grep -o -E 'd[0-9]+' | sed 's/d//')
            
            python $PERANTO_PATH/scripts/generate.py -c $json_file $PERANTO_PATH/examples/svo/middleman1.json $PERANTO_PATH/examples/svo/english1.json --sents -n 5897 > $DATA_PATH/peranto_{self.dist}_s${{strength}}_d${{discount}}.txt
        done
        """

        with open(script_name, 'w') as script_file:
            script_file.write(shell_script_content)
    
    def setup(self, base_file="amr1.json"):
        """
        One-stop shop for creating all necessary files and sh scripts for an expirament. 
        Note that the base file argument determins which previous previous paramater settings are 
        used to generate json files for subsequent iterations of tuning.
        """
        self.create_param_space()
        self.create_json_configs(base_file)
        self.create_sh_script()
        print(f'Experiment setup completed. Please run shell script on appa.')

    def get_peranto_data(self):
        """
        Overwrites Testperanto generated output files to only contain relevant information
        based on the distribution that is being tuned.
        """
        peranto_data = {}
        # iterate through all files 
        with open(self.param_path, "r") as f:
            lines = f.readlines()
            parameters = [(float(line.split(",")[0].strip()), float(line.split(",")[1].strip())) for line in lines]

        for strength, discount in parameters:
            file_path = f"{self.output_path}/peranto_{self.dist}_s{strength}_d{discount}.txt"
            store = PerantoTripleStore() 
            try:
                with open(file_path, 'r') as file:
                    for line in file.readlines():
                        subject = line[0]
                        verb = line[1]
                        obj= line[2]
                        store.add_triple(subject, verb, obj)      
            except FileNotFoundError:
                print(f"The file at path {file_path} was not found.")
                return None
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                return None
            stuff_to_write = store.get(self.dist)  
            peranto_data[(strength, discount)] = stuff_to_write.copy()
            # write filtered content to files
            try:
                with open(file_path, 'w') as file:
                    # Iterate through each tuple in the list
                    for tup in stuff_to_write:
                        # Iterate through each item in the tuple
                        for item in tup:
                            # Write each item on a line separated by space 
                            file.write(str(item) + " ")
                        # Add an newline for separation between tuples
                        file.write("\n")
            except Exception as e:
                print(f"An error occurred: {str(e)}")

        return peranto_data

    def get_singleton_prop(self):
        data = self.get_peranto_data() # {(strength, discount) : [(he, eat), (my, name), ...]} 

        def get_prop(lst):
            singleton_count = 0
            seen = set()
            singletons = []
            total = []

            if len(lst[0]) == 1 or isinstance(str, lst[0]): # self.dist in [nn, vb, nn.arg0, nn.arg1]
                # the isinstance checks in case data is (strength, discount) : ['he', 'my', ...]
                lst = [(x, x) for x in lst] # this just makes things convenient

            for idx, (a, b) in enumerate(lst):
                if (a, b) not in seen: # singleton
                    singleton_count += 1
                    seen.add((a, b))
                singletons.append(singleton_count)
                total.append(idx + 1)

            singleton_prop = np.array(singletons) / np.array(total)
            return singleton_prop

        peranto_prop = {(str, dis) : get_prop(lst) for (str, dis), lst in data.items()}
        treebank_prop = get_prop(self.store.get(self.dist))
        
        return peranto_prop, treebank_prop

    def get_top_k(self, singleton_prop, treebank_prop, k=10):
        def mse(x, y):
            return sum((a - b)**2 for a,b in zip(x, y))

        mse_results = {(str, dis) : mse(treebank_prop, sing_prop)
                        for (str, dis), sing_prop in singleton_prop.items()}

        mse_results = sorted(mse_results.items(), key = lambda x : x[1]) # sort by mse

        try:
            file_path = f"{self.mse_path}/{self.dist}_mse_results.txt"
            
            with open(file_path, 'w') as file:
                for (strength, discount), mse in mse_results.items():
                    file.write(f"S ={str(strength)}, D={str(discount)} MSE: {mse}")
                    file.write("\n")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
        return [(str, dis) for (str, dis), _ in mse_results][:k]

    def create_plot(self, singleton_prop, treebank_prop, best_params):
        plt.figure(figsize=(10, 6))
        
        data = {f"S={strength}, D={discount}" : curve for (strength, discount), curve in singleton_prop.items() 
                if (strength, discount) in best_params}
        data["Treebank"] = treebank_prop

        for name, curve in data.items():
            plt.plot(list(range(1, len(lst) + 1)), curve, label=name)
            
        dist_map = {
            "nn"          : "Nouns",
            "vb"          : "Verbs",
            'nn.arg0'     : "Subjects",
            "nn.arg1"     : "Objects",
            "nn.arg0.$y0" : "Subject Verb Pairs",
            "nn.arg1.$y0" : "Verb Object Pairs"
        }
        plt.title(f"Singleton Proportion for {dist_map[self.dist]}")
        plt.xlabel("Total Number of Entries")
        plt.ylabel("Singleton Proportion")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)

        path = f"{self.plot_path}/{self.dist}_curves.jpg"
        plt.savefig(path)
        plt.show()
    
    def run(self, k=10):
        singleton_prop, treebank_prop = self.get_singleton_prop()
        best_params = self.get_top_k(singleton_prop, treebank_prop, k)
        self.create_plot(singleton_prop, treebank_prop, best_params)

if __name__ == "__main__":
    config = Config('nn')
    exp = Experiment(config)
    exp.setup()
    
    