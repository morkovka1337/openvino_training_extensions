import argparse
import os.path
import pickle as pkl
import json
from os.path import join

# import torch

START_TOKEN = 0
PAD_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3
NUMBER_SIGNS = set("0123456789.")
# buid sign2id


def split_number(sign):
    if set(sign) <= NUMBER_SIGNS:
        return list(sign)
    else:
        return [sign]


class Vocab(object):
    def __init__(self, loaded_sign2id=None, loaded_id2sign=None):
        if loaded_id2sign is None and loaded_sign2id is None:
            self.sign2id = {"<s>": START_TOKEN, "</s>": END_TOKEN,
                            "<pad>": PAD_TOKEN, "<unk>": UNK_TOKEN}
            self.id2sign = dict((idx, token)
                                for token, idx in self.sign2id.items())
            self.length = 4
        else:
            assert isinstance(loaded_id2sign, dict) and isinstance(loaded_sign2id, dict)
            assert len(loaded_id2sign) == len(loaded_sign2id)
            self.sign2id = loaded_sign2id
            self.id2sign = loaded_id2sign
            self.length = len(loaded_id2sign)

    def add_sign(self, sign):
        if sign not in self.sign2id:
            self.sign2id[sign] = self.length
            self.id2sign[self.length] = sign
            self.length += 1

    def add_formula(self, formula):
        for sign in formula:
            self.add_sign(sign)

    def __len__(self):
        return self.length

    def construct_phrase(self, indices, max_len=None):
        phrase_converted = []
        if max_len is not None:
            indices_to_convert = indices[:max_len]
        else:
            indices_to_convert = indices

        for token in indices_to_convert:
            val = token.item()
            if val == END_TOKEN:
                break
            phrase_converted.append(
                self.id2sign.get(val, "?"))

        return " ".join(phrase_converted)

    def construct_indices(self, phrase):
        """
        given a phrase, returns indices which
        this phrase corresponds to
        """
        indices = []

        for token in phrase.split():
            indices.append(self.sign2id.get(token))
        return torch.Tensor(indices)


def write_vocab(data_dir, json=True):
    """
    traverse training formulas to make vocab
    and store the vocab in the file
    """
    vocab = Vocab()
    formulas_file = join(data_dir, 'formulas.norm.lst')
    with open(formulas_file, 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]

    with open(join(data_dir, 'train_filter.lst'), 'r') as f:

        for line in f:
            _, idx = line.strip('\n').split()
            idx = int(idx)
            formula = formulas[idx].split()
            formula_splitted_numbers = []
            for sign in formula:
                formula_splitted_numbers += split_number(sign)

            vocab.add_formula(formula_splitted_numbers)

    vocab_file = join(data_dir, 'vocab.pkl')
    print("Writing Vocab File in ", vocab_file)
    with open(vocab_file, 'wb') as w:
        dict_to_store = {
            "id2sign": vocab.id2sign,
            "sign2id": vocab.sign2id,
        }
        if json:
            json.dump(dict_to_store,
                      w, indent=4, sort_keys=True)
        else:
            pkl.dump(dict_to_store, w)


def read_vocab(vocab_path):
    if '.pkl' in vocab_path:
        with open(vocab_path, "rb") as f:
            vocab_dict = pkl.load(f)
    elif 'json' in vocab_path:
        with open(vocab_path, "r") as f:
            vocab_dict = json.load(f)
            for k, v in vocab_dict['id2sign'].items():
                del vocab_dict['id2sign'][k]
                vocab_dict['id2sign'][int(k)] = v
    else:
        raise ValueError("Wrong extension of the vocab file")
    vocab = Vocab(loaded_id2sign=vocab_dict["id2sign"], loaded_sign2id=vocab_dict["sign2id"])
    return vocab


def pkl_to_json(vocab_path):
    with open(vocab_path, "rb") as f:
        vocab_dict = pkl.load(f)
        dict_to_store = {
            "id2sign": vocab_dict["id2sign"],
            "sign2id": vocab_dict["sign2id"],
        }
        json_path = vocab_path.replace(".pkl", ".json")
        with open(json_path, 'w') as dest:
            json.dump(dict_to_store, dest, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Im2Latex Training Program")
    parser.add_argument('--data_path', help='path to the formulas')
    args = parser.parse_args()
    write_vocab(args.data_path)