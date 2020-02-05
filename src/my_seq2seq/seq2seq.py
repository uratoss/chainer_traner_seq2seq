#!/usr/bin/env python3
import chainer

import chainer.links as L
import chainer.functions as F

import numpy as xp

# {{{ Seq2seq(chainer.Chain)
# sequence to sequence class
# This class output reply sequences if you input utterance data

class Seq2seq(chainer.Chain):
    def __init__(self, embed, n_lay=1, n_unit=100, dropout=0.5):
        super().__init__()
        with self.init_scope():
            self.embed = embed
            print(self.embed.getVectorSize())
            print(len(self.embed.getVocab()))
            self.encoder = L.NStepLSTM(n_lay, self.embed.getVectorSize(), n_unit, dropout)
            self.decoder = L.NStepLSTM(n_lay, self.embed.getVectorSize(), n_unit, dropout)
            self.W = L.Linear(n_unit, len(self.embed.getVocab()))

    def __call__(self, xs, ts=None, hx=None, cx=None, max_size=30):
        global vocab
        # エンコーダ側の処理
        xs_embeded = [self.embed(x) for x in xs]
        hx, cx, _ = self.encoder(hx, cx, xs_embeded)
        # デコーダ側の処理
        eos = xp.array([self.embed.getID("<eos>")], dtype=xp.int32)
        if ts is None:
            ys = [eos] * len(xs)
            ys_list = []
            for i in range(max_size):
                ys_embeded = [self.embed(y) for y in ys]
                hx, cx, ys_embeded = self.decoder(hx, cx, ys_embeded)
                ys = [xp.reshape(xp.argmax(F.softmax(self.W(y_embeded)).data), (1))
                      for y_embeded in ys_embeded]
                ys_list.append(ys)
            ys_list.append([eos] * len(xs))
            return ys_list
        else:
            ts = [F.concat((eos, t[:-1]), axis=0) for t in ts]
            ts_embeded = [self.embed(t) for t in ts]
            _, _, ys_embeded = self.decoder(hx, cx, ts_embeded)
            ys = [self.W(y) for y in ys_embeded]
            return ys
# }}}
