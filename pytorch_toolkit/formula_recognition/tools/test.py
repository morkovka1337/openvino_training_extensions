import argparse
import yaml
import os.path
from functools import partial

from tqdm import tqdm

import torch
from im2latex.data.utils import collate_fn, create_list_of_transforms, get_timestamp
from im2latex.data.vocab import read_vocab
from im2latex.datasets.im2latex_dataset import (BatchRandomSampler,
                                                Im2LatexDataset)
from im2latex.models.im2latex_model import Im2latexModel
from torch.utils.data import DataLoader
from tools.evaluation_tools import Im2latexRenderBasedMetric


class Evaluator():
    def __init__(self, work_dir, config):
        self.config = config
        self.model_path = config.get('model_path')
        self.val_path = config.get('val_path')
        self.vocab = read_vocab(config.get('vocab_path'))
        self.val_transforms_list = config.get('val_transforms_list')
        self.work_dir = work_dir
        self.val_results_path = os.path.join(self.work_dir, "val_results")
        self.print_freq = config.get('print_freq', 16)
        self.create_dirs()
        self.load_dataset()
        self.model = Im2latexModel(config.get('backbone_type'), config.get(
            'backbone_config'), len(self.vocab), config.get('head'))
        if self.model_path is not None:
            self.model.load_weights(self.model_path)

        self.device = config.get('device', 'cpu')
        self.model = self.model.to(self.device)
        self.time = get_timestamp()

    def create_dirs(self):
        if not os.path.exists(self.val_results_path):
            os.makedirs(self.val_results_path)

    def load_dataset(self):

        val_dataset = Im2LatexDataset(self.val_path, 'validate',
                                      transform=None
                                      )
        val_sampler = BatchRandomSampler(dataset=val_dataset, batch_size=1)
        batch_transform_val = create_list_of_transforms(self.val_transforms_list)
        self.val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=partial(collate_fn, self.vocab.sign2id,
                               batch_transform=batch_transform_val),
            num_workers=os.cpu_count())

    def validate(self):
        self.model.eval()
        print("Validation started")
        annotations = []
        predictions = []
        metric = Im2latexRenderBasedMetric()
        with torch.no_grad():

            for img_name, imgs, tgt4training, tgt4cal_loss in tqdm(self.val_loader):
                print(imgs.shape)
                imgs = imgs.to(self.device)
                tgt4training = tgt4training.to(self.device)
                tgt4cal_loss = tgt4cal_loss.to(self.device)
                _, pred = self.model(imgs)
                for j, phrase in enumerate(pred):
                    gold_phrase_str = self.vocab.construct_phrase(tgt4cal_loss[j])
                    pred_phrase_str = self.vocab.construct_phrase(phrase,
                                                                  max_len=1 +
                                                                  len(gold_phrase_str.split()))

                    annotations.append((gold_phrase_str, img_name[j]))
                    predictions.append((pred_phrase_str, img_name[j]))

        res = metric.evaluate(annotations, predictions)
        return res


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config')
    args.add_argument('--work_dir')
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SaveLoadr)
    validator = Evaluator(args.work_dir, config)
    print(validator.validate())