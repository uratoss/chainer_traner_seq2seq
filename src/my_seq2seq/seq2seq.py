#!/usr/bin/env python3
import chainer

import chainer.links as L
import chainer.functions as F

import numpy as xp

# {{{ Seq2seq(chainer.Chain)
# sequence to sequence class
# This class output reply sequences if you input utterance data


class Seq2seq(chainer.Chain):
    def __init__(self, n_vocab, n_lay=1, n_unit=100, dropout=0.5):
        super().__init__()
        with self.init_scope():
            self.embedx = L.EmbedID(n_vocab, n_unit)
            self.embedy = L.EmbedID(n_vocab, n_unit)
            self.encoder = L.NStepLSTM(
                n_lay, n_unit, n_unit, dropout)
            self.decoder = L.NStepLSTM(
                n_lay, n_unit, n_unit, dropout)
            self.W = L.Linear(n_unit, n_vocab)

    def __call__(self, xs, ts=None, hx=None, cx=None, max_size=30):
        # エンコーダ側の処理
        xs_embeded = [self.embedx(x[:-1]) for x in xs]
        hx, cx, _ = self.encoder(hx, cx, xs_embeded)
        # デコーダ側の処理
        if ts is None:
            ys = [xp.array([x[-1]], dtype=xp.int32) for x in xs]
            ys_list = []
            for i in range(max_size):
                ys_embeded = [self.embedy(y) for y in ys]
                hx, cx, ys_embeded = self.decoder(hx, cx, ys_embeded)
                ys = [xp.reshape(xp.argmax(F.softmax(self.W(y_embeded)).data), (1))
                      for y_embeded in ys_embeded]
                ys_list.append(ys)
            ys = [xp.array([x[-1]], dtype=xp.int32) for x in xs]
            ys_list.append(ys)
            return ys_list
        else:
            ts = [F.concat((xp.reshape(t[-1], (1)), t[:-1]), axis=0)
                  for t in ts]
            ts_embeded = [self.embedy(t) for t in ts]
            _, _, ys_embeded = self.decoder(hx, cx, ts_embeded)
            ys = [self.W(y) for y in ys_embeded]
            return ys
# }}}

# 'GAtt(chainer.Chain)'{{{


class GAtt(chainer.Chain):
    def __init__(self, n_vocab, n_lay=1, n_unit=100, dropout=0.5):
        super().__init__()
        with self.init_scope():
            self.embedx = L.EmbedID(n_vocab, n_unit)
            self.embedy = L.EmbedID(n_vocab, n_unit)
            self.encoder = L.NStepBiLSTM(n_lay, n_unit, n_unit, dropout)
            self.decoder = L.NStepLSTM(n_lay, n_unit, n_unit, dropout)
            self.W = L.Linear(n_unit, n_vocab)
            self.W1 = L.Linear(2 * n_unit, n_unit)
            self.W2 = L.Linear(2 * n_unit, n_unit)
            self.W3 = L.Linear(2 * n_unit, n_unit)
            self.Wc1 = L.Linear(n_unit, n_unit)
            self.Wc2 = L.Linear(n_unit, n_unit)

    def scoreDot(self, atts, ys):
        xs = [self.W3(att) for att in atts]
        dot = F.batch_matmul(ys,F.transpose(xs,(0,2,1)))
        aws = F.softmax(dot,2)
        cts = None
        for x,aw in zip(xs,aws): # split batch
            aw = F.expand_dims(aw,1)
            x = F.tile(F.expand_dims(x,0),(aw.shape[0],1,1))
            ct = F.batch_matmul(aw,x)
            cts = ct if cts is None else F.concat((cts,ct),1)
        cts = F.transpose(cts,(1,0,2))
        ds = [F.tanh(self.Wc1(ct) + self.Wc2(y)) for y, ct in zip(ys,cts)] 
        #for y,ct in zip(ys,cts):
        #    d = F.tanh(self.Wc1(ct) + self.Wc2(y))
        #    d = F.expand_dims(d,0)
        #    ds = d if ds is None else F.concat((ds,d),0)
        return ds

    def __call__(self, xs, ts=None, hx=None, cx=None, max_size=30):
        # エンコーダ側の学習
        xs_embeded = [self.embedx(x[:-1]) for x in xs]
        hx, cx, attentions = self.encoder(hx, cx, xs_embeded)

        # 双方向なので、デコーダ用に次元を合わせる
        # hx(lay*2,batch,demb) -> hx(lay,batch,demb)
        # cx(lay*2,batch,demb) -> cx(lay,batch,demb)
        hx = F.concat((hx[0::2], hx[1::2]), axis=2)
        shape = hx.shape
        hx = self.W1(F.reshape(hx, (shape[0]*shape[1], shape[2])))
        hx = F.reshape(hx, (shape[0], shape[1], int(shape[2]/2)))

        cx = F.concat((cx[0::2], cx[1::2]), axis=2)
        shape = cx.shape
        cx = self.W2(F.reshape(cx, (shape[0]*shape[1], shape[2])))
        cx = F.reshape(cx, (shape[0], shape[1], int(shape[2]/2)))

        if ts is None:
            ys = [xp.array([x[-1]], dtype=xp.int32) for x in xs]
            ys_list = []
            for i in range(max_size):
                ys_embeded = [self.embedy(y) for y in ys]
                hx, cx, ys_embeded = self.decoder(hx, cx, ys_embeded)
                ys_embeded = self.scoreDot(attentions, ys_embeded)
                sys.exit()
                ys = [xp.reshape(xp.argmax(F.softmax(self.W(y_embeded)).data), (1))
                      for y_embeded in ys_embeded]
                ys_list.append(ys)
            ys = [xp.array([x[-1]], dtype=xp.int32) for x in xs]
            ys_list.append(ys)
            return ys_list
        else:
            # デコーダ側の学習
            ts = [F.concat((xp.reshape(t[-1], (1)), t[:-1]), axis=0)
                  for t in ts]

            ts_embeded = [self.embedy(t) for t in ts]
            _, _, ys_embeded = self.decoder(hx, cx, ts_embeded)
            # attを計算する
            ys_embeded = self.scoreDot(attentions, ys_embeded)
            ys = [self.W(y) for y in ys_embeded]
            return ys

# }}}
