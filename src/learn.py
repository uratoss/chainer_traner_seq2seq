#!/usr/bin/env python3
import chainer

from chainer import optimizers
from chainer import iterators

from chainer import training
from chainer.training import extensions
from chainer.training.triggers import EarlyStoppingTrigger

from chainer.serializers import save_npz

#import cupy as xp
import numpy as xp

import pickle
import argparse
import os

#from my_seq2seq import w2v
from my_seq2seq import seq2seq
from my_seq2seq import util

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name',
                        default='model/seq2seq/predictor.npz')
    parser.add_argument('-e', '--epoch', default='100')
    parser.add_argument('-b', '--batch_size', default='30')
    parser.add_argument('-d', '--device_number', default='-1')
    args = parser.parse_args()

    with open("data/vocab.dump", "rb") as f:
        vocab = pickle.load(f)
    with open("data/rvocab.dump", "rb") as f:
        rvocab = pickle.load(f)
    x_file = "./data/x.txt"
    t_file = "./data/t.txt"
    train, valid, test = util.load_dataset(vocab, x_file, t_file)

    batch_size = int(args.batch_size)
    n_epoch = int(args.epoch)
    device = int(args.device_number)
    out_directory = os.path.dirname(args.model_name)

    train_iter = iterators.SerialIterator(train, batch_size)
    valid_iter = iterators.SerialIterator(
        valid, batch_size, shuffle=False, repeat=False)

    #predictor = seq2seq.Seq2seq(n_vocab=len(vocab))
    predictor = seq2seq.GAtt(n_vocab=len(vocab),n_lay=2,n_unit=200)

    if device >= 0:
        predictor.to_gpu(device)

    model = util.MyRegressor(predictor)
    optimizer = optimizers.Adam().setup(model)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=util.converter, device=device)

    trigger = EarlyStoppingTrigger(monitor='val/main/loss', check_trigger=(1, 'epoch'),
                                   patients=5, max_trigger=(n_epoch, 'epoch'))
    trainer = training.Trainer(updater, trigger, out=out_directory)

    trainer.extend(extensions.LogReport(trigger=(1, 'epoch'), log_name='log'))
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.Evaluator(valid_iter, model,
                                        converter=util.converter, device=device), name='val')
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'val/main/loss', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.ParameterStatistics(
        model.predictor.W, {'mean': xp.mean}, report_grads=True))

    trainer.run()
    gen_model = util.Generator(predictor=predictor, device=device, max_size=30)
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        ys_list = gen_model(test)
        for ys in ys_list:
            for y in ys:
                y = int(y)
                if y is vocab['<eos>']:
                    print('\n')
                    break
                print(rvocab[y], end='')

    if device >= 0:
        predictor.to_cpu()
    save_npz(args.model_name, predictor)
