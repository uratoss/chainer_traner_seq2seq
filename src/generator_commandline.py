import chainer

import chainer.links as L
import chainer.functions as F

from chainer import reporter
from chainer import optimizers
from chainer import iterators

from chainer.datasets import TupleDataset
from chainer.datasets import split_dataset_random
from chainer.dataset.convert import to_device

from chainer import training
from chainer.training import extensions
from chainer.training.triggers import EarlyStoppingTrigger

from chainer.serializers import load_npz

import numpy as np

import pickle
import MeCab
import sys
import argparse

def load_data(filename):
  global vocab
  sequences = open(filename).readlines()
  data = []
  for line in sequences:
    line = line.replace("\n","<eos>").strip().split()
    words = [vocab[word] for word in line]
    data.append(np.array(words).astype(np.int32))
  return data

def converter(batch,device):
  xs = []
  ts = []
  for b in batch:
    x = b[0]
    t = b[1]
    xs.append(to_device(device,x))
    ts.append(to_device(device,t))
  return (xs,ts)

class Seq2seq(chainer.Chain):
  def __init__(self, n_vocab, n_lay=1, n_unit=100, dropout=0.5):
    super(Seq2seq, self).__init__()
    with self.init_scope():
      self.embedx=L.EmbedID(n_vocab, n_unit)
      self.embedy=L.EmbedID(n_vocab, n_unit)
      self.encoder=L.NStepLSTM(n_lay, n_unit, n_unit, dropout)
      self.decoder=L.NStepLSTM(n_lay, n_unit, n_unit, dropout)
      self.W=L.Linear(n_unit, n_vocab)


  def __call__(self, xs, ts=None, hx=None, cx=None, max_size=30):
    global vocab
    # エンコーダ側の処理
    xs_embeded = [self.embedx(x) for x in xs]
    hx, cx, _ = self.encoder(hx, cx, xs_embeded)
    # デコーダ側の処理
    eos = np.array([vocab["<eos>"]], dtype=np.int32)
    if ts is None:
      eos = np.array([vocab["<eos>"]], dtype=np.int32)
      ys = [eos] * len(xs)
      ys_list = []
      for i in range(max_size):
        ys_embeded = [self.embedy(y) for y in ys]
        hx,cx,ys_embeded = self.decoder(hx,cx,ys_embeded)
        ys = [np.reshape(np.argmax(F.softmax(self.W(y_embeded)).data),(1)) for y_embeded in ys_embeded]
        ys_list.append(ys)
      ys_list.append([eos] * len(xs))
      return ys_list
    else:
      ts = [F.concat((eos, t), axis=0) for t in ts]
      ts_embeded = [self.embedy(t) for t in ts]
      _, _, ys_embeded = self.decoder(hx, cx, ts_embeded)
      ys = [self.W(y) for y in ys_embeded]
      return ys

class Generator:
  def __init__(self,predictor, device=-1, converter = converter, max_size = 30):
    self.predictor = predictor
    self.device = device
    self.converter = converter
    self.max_size = max_size

  def __call__(self, dataset):
    global vocab
    global rvocab
    xs, ts = self.converter(dataset,self.device)
    ys = self.predictor(xs=xs, max_size=self.max_size)
    ys = self.molder(ys, self.device)
    return ys

  def molder(self, ys, device):
    ys= [np.reshape(np.concatenate(y),(1,len(y))) for y in ys]
    ys = np.concatenate(ys)
    ys = np.hsplit(ys,ys.shape[1])
    ys= [np.concatenate(y) for y in ys]
    ys = to_device(device,ys)
    return ys

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model_name',default='model/predictor.npz')
args = parser.parse_args()

# 単語辞書の取得
with open("data/vocab.dump", "rb") as f:
    vocab = pickle.load(f)
with open("data/rvocab.dump", "rb") as f:
    rvocab = pickle.load(f)

# モデルの読み込み
predictor = Seq2seq(n_vocab=len(vocab))
load_npz(args.model_name, predictor)

# デバイスidの設定
device = -1
# モデルをデバイスに送る
if device >= 0:
    predictor.to_gpu(device)

model = Generator(predictor=predictor, device=device, max_size=30)

# MeCabの設定
m = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

# ユーザの入力を処理
user_input = sys.stdin.readlines()[0]
s = m.parse(user_input.replace(' ','').strip()).replace('\n','').strip().split()
xs = []
for x in s:
    try:
        xs.append(vocab[x])  
    except(KeyError):
        xs.append(vocab['<unk>'])
xs = np.array(xs).astype(np.int32)
test = [(xs,np.zeros(1).astype(np.int32))]

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
