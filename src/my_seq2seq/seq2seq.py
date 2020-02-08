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
            self.Wa1 = L.Linear(n_unit, n_unit)
            self.Wa2 = L.Linear(n_unit, n_unit)
            self.Wa3 = L.Linear(n_unit, n_unit)
            self.Wc1 = L.Linear(n_unit, n_unit)
            self.Wc2 = L.Linear(n_unit, n_unit)
            self.Wc3 = L.Linear(n_unit, n_unit)

    # 'dotScore' {{{
    def scoreDot(self, attentions, ys_embeded_list):
        ys = []
        for attention, ys_embeded in zip(attentions, ys_embeded_list):
            # attention -> 1文のエンコーダの出力
            # ys_embeded -> 1文のデコーダの出力
            attention = self.W3(attention)
            target = []
            for y_embeded in ys_embeded:
                y_embeded = F.reshape(y_embeded, (1, y_embeded.shape[0]))
                # y_embeded -> デコーダLSTMブロックの1単語の出力
                # ctを計算する
                s = 0.0
                scores = []
                for enc in attention:
                    enc = F.reshape(enc, (1, enc.shape[0]))
                    # ecn -> エンコーダLSTMブロックの1単語の出力
                    scores.append(
                        F.exp(F.matmul(y_embeded, enc, transb=True)).data[0][0])
                    s += scores[-1]
                ct = xp.zeros((1, y_embeded.shape[-1]), dtype=xp.float32)
                for enc, score in zip(attention, scores):
                    # ecn -> エンコーダLSTMブロックの1単語の出力
                    # score -> 各エンコーダの出力に対するdot積の結果
                    alpi = score / s
                    ct += alpi * enc.data[0]
                # ctから出力単語を作る
                ct = chainer.Variable(ct)
                target.append(F.tanh(self.Wc1(ct) + self.Wc2(y_embeded)))
            ys.append(F.concat(target, axis=0))
        return ys

    # }}}

    # 'generalScore' {{{
    def scoreGeneral(self, attentions, ys_embeded_list):
        ys = []
        for attention, ys_embeded in zip(attentions, ys_embeded_list):
            # attention -> 1文のエンコーダの出力
            # ys_embeded -> 1文のデコーダの出力

            # エンコーダの出力を順方向・逆方向に分ける
            attention_f, attention_b = F.split_axis(attention, 2, axis=1)
            target = []
            for y_embeded in ys_embeded:
                y_embeded = F.reshape(y_embeded, (1, y_embeded.shape[0]))
                # y_embeded -> デコーダLSTMブロックの1単語の出力
                # ctを計算する
                s = 0.0
                scores = []
                for enc_f, enc_b in zip(attention_f, attention_b):
                    enc_f = F.reshape(enc_f, (1, enc_f.shape[0]))
                    enc_b = F.reshape(enc_b, (1, enc_b.shape[0]))
                    # ecn_f -> 順方向エンコーダLSTMブロックの1単語の出力
                    # ecn_b -> 逆方向エンコーダLSTMブロックの1単語の出力
                    scores.append(
                        F.exp(self.Wa1(enc_f) + self.Wa2(enc_b) + self.Wa3(y_embeded)).data[
                            0
                        ][0]
                    )
                    s += scores[-1]
                ct_f = xp.zeros((1, y_embeded.shape[-1]), dtype=xp.float32)
                ct_b = xp.zeros((1, y_embeded.shape[-1]), dtype=xp.float32)
                for enc_f, enc_b, score in zip(attention_f, attention_b, scores):
                    enc_f = F.reshape(enc_f, (1, enc_f.shape[0]))
                    enc_b = F.reshape(enc_b, (1, enc_b.shape[0]))
                    # ecn_f -> 順方向エンコーダLSTMブロックの1単語の出力
                    # ecn_b -> 逆方向エンコーダLSTMブロックの1単語の出力
                    alpi = score / s
                    ct_f += alpi * enc_f.data[0]
                    ct_b += alpi * enc_b.data[0]
                # ctから出力単語を作る
                ct_f = chainer.Variable(ct_f)
                ct_b = chainer.Variable(ct_b)
                target.append(
                    F.tanh(self.Wc1(ct_f) + self.Wc2(ct_b) + self.Wc3(y_embeded)))
            ys.append(F.concat(target, axis=0))
        return ys

    # }}}

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
            # ts = self.scoreGeneral(attentions, ys_embeded)
            ys = [self.W(y) for y in ys_embeded]
            return ys

# }}}
