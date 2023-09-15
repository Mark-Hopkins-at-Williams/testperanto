from random import shuffle

def split_lines(filename, dev_pct=0.1, test_pct=0.1):
    verb_appearances = dict()    
    with open(filename) as reader:
        for i, line in enumerate(reader):
            _, verb = line.split("#")
            verb = verb.strip()
            if verb not in verb_appearances:
                verb_appearances[verb] = []
            verb_appearances[verb].append(i)
    train, dev, test, unassigned = [], [], [], []
    for verb in verb_appearances:
        line_nums = verb_appearances[verb]
        if len(line_nums) < 3:
            train.extend(line_nums)
        else:
            train.append(line_nums[0])
            dev.append(line_nums[1])
            test.append(line_nums[2])
            unassigned.extend(line_nums[3:])
    num_lines = len(train) + len(dev) + len(test) + len(unassigned)
    num_dev_needed = int(dev_pct * num_lines) - len(dev)
    num_test_needed = int(test_pct * num_lines) - len(test)
    shuffle(unassigned)
    dev.extend(unassigned[:num_dev_needed])
    test.extend(unassigned[num_dev_needed:num_dev_needed+num_test_needed])
    train.extend(unassigned[num_dev_needed+num_test_needed:])
    train = set(train)
    dev = set(dev)
    test = set(test)
    dev_and_test_strings = set()
    with open(filename) as reader:
        for i, line in enumerate(reader):
            if i in dev or i in test:
                dev_and_test_strings.add(line.strip())
    revised_train = set()
    with open(filename) as reader:
        for i, line in enumerate(reader):
            if i in train and line.strip() not in dev_and_test_strings:
                revised_train.add(i)
    return revised_train, dev, test    


def filter_file(orig_file, filtered_file, line_nums):
    with open(orig_file) as reader:
        with open(filtered_file, 'w') as writer:
            for i, line in enumerate(reader):
                if i in line_nums:
                    fields = line.split("#")
                    writer.write(fields[0].strip() + "\n")


def get_all_lines(filename):
    lines = []
    with open(filename) as reader:
        for line in reader:
            lines.append(line.strip())
    return lines


def assess_overlap(file1, file2):
    lines1 = get_all_lines(file1)
    lines2 = get_all_lines(file2)
    print(f"Num of lines in file1: {len(lines1)}")
    print(f"Num of lines in file2: {len(lines2)}")
    lines1 = set(lines1)
    lines2 = set(lines2)
    print(f"Unique lines in file1: {len(lines1)}")
    print(f"Unique lines in file2: {len(lines2)}")
    print(f"Num overlapping lines: {len(lines1 & lines2)}")


def organize_parallel_data(lang1file, lang2file, lang1ext, lang2ext):
    train, dev, test = split_lines(lang1file)
    filter_file(lang1file, f'train.{lang1ext}', train)
    filter_file(lang2file, f'train.{lang2ext}', train)
    filter_file(lang1file, f'dev.{lang1ext}', dev)
    filter_file(lang2file, f'dev.{lang2ext}', dev)
    filter_file(lang1file, f'test.{lang1ext}', test)
    filter_file(lang2file, f'test.{lang2ext}', test)


#organize_parallel_data('eng-nah.0', 'eng-nah.1', 'eng', 'nah')
assess_overlap('train.nah', 'dev.nah')