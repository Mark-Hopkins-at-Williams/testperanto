import os
import json 

import numpy as np

SRC_PATH      = os.getcwd()
MAIN_PATH     = os.path.dirname(SRC_PATH)
DATA_PATH     = f"{MAIN_PATH}/data"
PER_DATA_PATH = f"{DATA_PATH}/peranto_data"
PER_JSON_PATH = f"{DATA_PATH}/peranto_json"
PERANTO_PATH  = f"{os.path.dirname(MAIN_PATH)}/testperanto"
JSON_PATH     = f"{PERANTO_PATH}/examples/svo"

def write_input_space(inputs, path=DATA_PATH):
    """
    Given a dict {"Strength" : [strengths], "Discount" : [discounts]}
    this function creates a parameters.txt file that contains all (S, D) pair
    """
    # generate pairs
    pairs = [(strength, discount) for strength in inputs["Strength"] for discount in inputs["Discount"]]

    # Create parameters.txt file
    with open(os.path.join(path, "parameters.txt"), "w") as f:
        for pair in pairs:
            f.write(f"{pair[0]}, {pair[1]}\n")

def create_json_files():
    """
    Uses the above input space (create_input_space()) and 
    for each (S,D) pair copies the amr1.json file and changes the 
    strength/discount to (S,D). All of this is saved in a folder of json files
    """
    PARAM_PATH = f"{DATA_PATH}/parameters.txt"
    JSON_FILE  = f"{JSON_PATH}/amr1.json"

    with open(PARAM_PATH, "r") as f:
        lines = f.readlines()
        parameters = [(float(line.split(",")[0].strip()), float(line.split(",")[1].strip())) for line in lines]
    
    # Read the original JSON content
    with open(JSON_FILE, "r") as f:
        json_content = json.load(f)
    
    # For each parameter set, create a new JSON file
    for strength, discount in parameters:
        # Modify the strength and discount in the JSON content
        for dist in json_content["distributions"]:
            if "strength" in dist:
                dist["strength"] = strength
            if "discount" in dist:
                dist["discount"] = discount
        
        # Save the modified JSON content to a new file
        new_file_name = f"amr_s{int(strength)}_d{int(discount*100)}.json"
        with open(os.path.join(PER_JSON_PATH, new_file_name), "w") as f:
            json.dump(json_content, f, indent=4)

def main():
    def create_input_space():
        inputs = {
            "Strength" : np.linspace(20, 40, 11),
            "Discount" : np.linspace(0, 1, 21)[:-1]
        }
        write_input_space(inputs)
    create_input_space()
    create_json_files()

if __name__ == "__main__":
    pass #main()