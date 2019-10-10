from . import InputExample
import csv
import gzip
import os

class ParaBankReader:
    """
    Reads in the ParaBank dataset. Each line contains three index numbers, two sentences (s1_col_idx, s2_col_idx) and one label (score_col_idx)
    """
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def get_examples(self, filename, max_examples=0):
        """
        filename specified which data split to use (train.csv, dev.csv, test.csv).
        """
        data = csv.reader(open(os.path.join(self.dataset_folder, filename), encoding="utf-8"))
        examples = []
        for row in data:
            id, _,_, s1, s2, score = row
            examples.append(InputExample(guid=filename+str(id), texts=[s1, s2], label=score))

            if max_examples > 0 and len(examples) >= max_examples:
                break

        return examples
