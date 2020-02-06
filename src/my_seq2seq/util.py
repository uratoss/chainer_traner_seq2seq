#!/usr/bin/env python3
import chainer

import chainer.functions as F

from chainer import reporter

import numpy as xp

from chainer.datasets import TupleDataset
from chainer.datasets import split_dataset_random
from chainer.dataset.convert import to_device

# {{{ def load_data(id_dict, x_path, t_path, train_per = 0.7, valid_per = 0.1)
# no necessary to explain


def load_dataset(id_dict, x_path, t_path, train_per=0.7, valid_per=0.1):
    xs = []
    for line in open(x_path).readlines():
        words = line.replace("\n", "").strip().split()
        words_id = [id_dict[word] for word in words]
        xs.append(xp.array(words_id, dtype=xp.int32))
    ts = []
    for line in open(t_path).readlines():
        words = line.replace("\n", "").strip().split()
        words_id = [id_dict[word] for word in words]
        ts.append(xp.array(words_id, dtype=xp.int32))
    dataset = TupleDataset(xs, ts)

    n_train = int(len(dataset) * train_per)
    n_valid = int(len(dataset) * valid_per)
    train, valid_test = split_dataset_random(dataset, n_train, seed=0)
    valid, test = split_dataset_random(valid_test, n_valid, seed=0)

    return (train, valid, test)
# }}}

# {{{ def converter(batch,device)
# convert from tupledataset to


def converter(batch, device=-1):
    xs = []
    ts = []
    for b in batch:
        x = b[0]
        t = b[1]
        xs.append(to_device(device, x))
        ts.append(to_device(device, t))
    return (xs, ts)
# }}}

# {{{ MyRegressor(chainer.Chain)
# This class wrap predictor.
# If you input xs and ts, predict ys by giving  xs for predictor, and calc loss.


class MyRegressor(chainer.Chain):
    def __init__(self, predictor):
        super(MyRegressor, self).__init__(predictor=predictor)

    def __call__(self, xs, ts):
        self.loss = None
        ys = self.predictor(xs=xs, ts=ts)
        # 損失を計算する
        for y, t in zip(ys, ts):
            loss = F.softmax_cross_entropy(y, t)
            self.loss = loss if self.loss is None else self.loss + loss
        self.loss = self.loss / len(xs)

        reporter.report({'loss': self.loss}, self)
        return self.loss
# }}}

# {{{ Generator
# no necessary to explain


class Generator:
    def __init__(self, predictor, device=-1, converter=converter, max_size=30):
        self.predictor = predictor
        self.device = device
        self.converter = converter
        self.max_size = max_size

    def __call__(self, dataset):
        xs, ts = self.converter(dataset, self.device)
        ys = self.predictor(xs=xs, max_size=self.max_size)
        ys = self.molder(ys)
        return ys

    def molder(self, ys):
        ys = [xp.reshape(xp.concatenate(y), (1, len(y))) for y in ys]
        ys = xp.concatenate(ys)
        ys = xp.hsplit(ys, ys.shape[1])
        ys = [xp.concatenate(y) for y in ys]
        ys = to_device(self.device, ys)
        return ys
# }}}
