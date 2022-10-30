import pandas as pd


def create_df_from_text_file(path: str):
    file = open(path, 'r')
    data = {"tokens": [], "label": []}
    tokens_sequence = []
    labels_sequence = []
    for line in file:
        if line == '\n':
            if len(labels_sequence) > 0:
                data["tokens"].append(tokens_sequence)
                data["label"].append(labels_sequence)
                tokens_sequence, labels_sequence = [], []
            continue
        try:
            token, label = line.split()
            tokens_sequence.append(token)
            labels_sequence.append(label)
        except ValueError:
            print("Couldn't split line to token and label: {}".format(line))
            pass
    file.close()
    return pd.DataFrame.from_dict(data)
