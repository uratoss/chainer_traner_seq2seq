#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import chainer

from chainer.serializers import load_npz

import numpy as xp

import pickle
import MeCab
import sys
import argparse

import random

from my_seq2seq import seq2seq
from my_seq2seq import util

import flask

import neologdn

model = None
vocab = None
rvocab = None
m = None
app = flask.Flask(__name__)


def load_models(model_name):
    global model
    global vocab
    global rvocab
    global m

    # 単語辞書の取得
    with open("data/vocab.dump", "rb") as f:
        vocab = pickle.load(f)
    with open("data/rvocab.dump", "rb") as f:
        rvocab = pickle.load(f)

    # モデルの読み込み
    #predictor = seq2seq.Seq2seq(n_vocab=len(vocab))
    predictor = seq2seq.GAtt(n_vocab=len(vocab))
    load_npz(model_name, predictor)

    model = util.Generator(predictor=predictor)
    m = MeCab.Tagger('-Owakati')


@app.route("/predict", methods=["POST"])
def predict():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }
    if flask.request.method == "POST":
        if flask.request.get_json().get("xs"):
            user_input = flask.request.get_json().get("xs")
            normalized = neologdn.normalize(user_input)
            s = m.parse(normalized).replace('\n', '').strip().split()
            print('xs is ',s)
            xs = []
            for x in s:
                try:
                    xs.append(vocab[x])
                except(KeyError):
                    xs.append(random.uniform(0, len(vocab)-1))
            xs.append(vocab['<eos>'])
            xs = xp.array(xs).astype(xp.int32)
            dummy = [(xs, xp.zeros(1).astype(xp.int32))]

            with chainer.using_config("train", False), chainer.using_config(
                "enable_backprop", False
            ):
                ys_list = model(dummy)[0]
                ys = []
                for y in ys_list:
                    if int(y) is vocab["<eos>"]:
                        break
                    ys.append(rvocab[int(y)])

            # classify the input feature
            response["ys"] = ''.join(ys)
            print('ys is ',response["ys"])

            # indicate that the request was a success
            response["success"] = True
    # return the data dictionary as a JSON response
    return flask.jsonify(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name',
                        default='model/seq2seq/predictor.npz')
    args = parser.parse_args()

    load_models(args.model_name)
    print('start server')
    app.run()
