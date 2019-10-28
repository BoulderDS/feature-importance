import json
import sys
import os
import gzip 
import pickle
import numpy as np
import csv

def load_json_list(filename):
    data = []
    with gzip.open(filename, "rt") as fin:
        for line in fin:
            data.append(json.loads(line))
    return data

def get_data(dataset_name, save_dir):
    file_name = dataset_name+'.jsonlist.gz'
    path = os.path.join(save_dir, file_name)
    data = load_json_list(path)
    tokens = [d["tokens"] for d in data]
    labels = [d["label"] for d in data]
    return tokens, labels

def main(dataset_name):
    sets = ["train", "dev", "test"]
    data_saving_path = "data/{}/".format(dataset_name)
    for set_type in sets:
        sents, labels = get_data("{}_".format(dataset_name)+set_type, data_saving_path)
        # convert -1 1 labels to 0, 1 for fitting into sst-2 data processing format
        np_labels = np.clip(1+np.asarray(labels), a_min=0, a_max=1)
        new_file_path = os.path.join(data_saving_path, set_type+".tsv")
        with open(new_file_path, "w") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t", quotechar='"')
            # write schema
            writer.writerow(["sent", "label"])
            for i in range(len(sents)):
                writer.writerow([sents[i], np_labels[i]])
        
if __name__ == "__main__":
    main(sys.argv[1])