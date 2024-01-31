from globals import TP_DATA_PATH

def format_number(num):
    """formats large integers (5000 -> 5k, 8000000 -> 8m)"""
    if num >= 1000000:
        return f"{num/1000000:.1f}m"
    elif num >= 1000:
        return f"{num/1000:.1f}k"
    else:
        return str(num)

def map_to_burmese(text):
    """takes str text and adds 4031 to utf-8 encoding"""
    translated = []
    for char in text:
        if char != ' ' and char != '\n': # preserve spaces/new lines
            for byte in char.encode('utf-8'):
                translated.append(chr(byte + 4031))
        else:
            translated.append(char)  
    return ''.join(translated)

def process_files(file_paths):
    """takes list of paths, opens the txt and changes it to map_to_burmese(txt)"""
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.readlines()

        transformed_content = [map_to_burmese(line) for line in content]

        with open(path, 'w', encoding='utf-8') as file:
            file.writelines(transformed_content)

if __name__ == "__main__":
    names = ['s110000.new_switches', 's010011.new_switches', 
    's011101_2.new_switches', 's110101.new_switches', 's000100.new_switches', 
    's111111.new_switches', 's001110.new_switches', 's100011.new_switches', 
    's011111.new_switches', 's010000.new_switches', 's011010.new_switches', 
    's100010.new_switches', 's110011.new_switches', 's111010.new_switches', 
    's111001.new_switches', 's111100.new_switches', 's010111.new_switches', 
    's000011.new_switches', 's011011.new_switches', 's010100.new_switches', 
    's111101.new_switches', 's111000.new_switches', 's100101.new_switches', 
    's110110.new_switches', 's001010.new_switches', 's001101.new_switches', 
    's110010.new_switches', 's011100.new_switches', 's101010.new_switches', 
    's010001.new_switches', 's000010.new_switches', 's110111.new_switches', 
    's001011.new_switches', 's100001.new_switches', 's010110.new_switches', 
    's010101.new_switches', 's100000.new_switches', 's011110.new_switches', 
    's000000.new_switches', 's000110.new_switches', 's001001.new_switches', 
    's001111.new_switches', 's000101.new_switches', 's010010.new_switches', 
    's011000.new_switches', 's001100.new_switches', 's101110.new_switches', 
    's111110.new_switches', 's000111.new_switches', 's101101.new_switches', 
    's001000.new_switches', 's110001.new_switches', 
    's100100.new_switches', 's100110.new_switches', 's101111.new_switches', 
    's110100.new_switches', 's111011.new_switches', 's101001.new_switches', 
    's101000.new_switches', 's100111.new_switches', 's000001.new_switches', 
    's011001.new_switches', 's101011.new_switches']

    paths = [f'{TP_DATA_PATH}/{switch}' for switch in names]
    process_files(paths) # process all but s011101