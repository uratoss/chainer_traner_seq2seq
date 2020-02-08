#!/usr/bin/env python3
import chainer

from chainer.serializers import load_npz

import numpy as xp

import pickle
import MeCab
import sys
import argparse

import random

#from my_seq2seq import w2v
from my_seq2seq import seq2seq
from my_seq2seq import util

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name',
                        default='model/seq2seq/predictor.npz')
    args = parser.parse_args()

    # 単語辞書の取得
    with open("data/vocab.dump", "rb") as f:
        vocab = pickle.load(f)
    with open("data/rvocab.dump", "rb") as f:
        rvocab = pickle.load(f)

    # モデルの読み込み
    predictor = seq2seq.Seq2seq(n_vocab=len(vocab))
    load_npz(args.model_name, predictor)

    # デバイスidの設定
    device = -1
    # モデルをデバイスに送る
    if device >= 0:
        predictor.to_gpu(device)

    model = util.Generator(predictor=predictor, device=device, max_size=10)

    # MeCabの設定
    m = MeCab.Tagger('-Owakati -d /usr/lib64/mecab/dic/mecab-ipadic-neologd')

    # ユーザの入力を処理
    user_input = sys.stdin.readline()
    s = m.parse(user_input.replace(' ', '').strip()
                ).replace('\n', '').strip().split()
    xs = []
    for x in s:
        try:
            xs.append(vocab[x])
        except(KeyError):
            xs.append(random.uniform(0, len(vocab)-1))
    xs.append(vocab['<eos>'])
    xs = xp.array(xs).astype(xp.int32)
    test = [(xs, xp.zeros(1).astype(xp.int32))]

    with chainer.using_config("train", False), chainer.using_config(
        "enable_backprop", False
    ):
        ys_list = model(test)
        for ys in ys_list:
            for y in ys:
                y = int(y)
                if y is vocab["<eos>"]:
                    print("\n")
                    break
                print(rvocab[y], end="")
