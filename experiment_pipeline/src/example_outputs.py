from config import SVOConfig

def extract_translation_info(lines, index):
    src_line = lines[index].split('\t')[1]
    tgt_line = lines[index + 1].split('\t')[1]
    model_line_parts = lines[index + 2].split('\t')
    model_translation = model_line_parts[2]
    return src_line, tgt_line, model_translation

def write_translations(output_path, results_path, names, lengths):
    with open(f"{output_path}/translations.txt", "w") as file:
        for name in names:
            for length in lengths:
                file_path = f"{results_path}/{'_'.join(name)}_{length}/translations"
                # Assuming that the file exists and is readable
                with open(file_path, "r") as f:
                    lines = f.readlines()[:10]

                src_name, tgt_name = name
                file.write(f"SRC: {src_name}, TGT: {tgt_name}, LEN: {length}\n")
                file.write(f"{'=' * 40}\n")

                for i in range(0, 10, 5):  # Process two translations (5 lines each)
                    src_line, tgt_line, model_translation = extract_translation_info(lines, i)
                    file.write(f"SRC Translation {i//5 + 1}: {src_line}")
                    file.write(f"TGT Translation {i//5 + 1}: {tgt_line}")
                    file.write(f"Model Translation {i//5 + 1}: {model_translation}")
                
                file.write("\n")  # New line after each name/length pair


config = SVOConfig()

output_path = config.EXP_PATH
results_path = config.RESULTS_PATH
names = [("OSV", "VOS"), ("OSV", "SVO"), ("SVO", "SOV")]
lengths = ["2.0k", "4.0k", "8.0k", "16.0k", "32.0k"]

write_translations(output_path, results_path, names, lengths)
