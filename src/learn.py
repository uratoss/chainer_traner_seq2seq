#!/usr/bin/env python3
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

from chainer.serializers import save_npz

import cupy as np

import pickle

def load_data(filename):
  global vocab
  sequences = open(filename).readlines()
  data = []
  for line in sequences:
    line = line.replace("\n","<eos>").strip().split()
    words = [vocab[word] for word in line]
    data.append(np.array(words).astype(np.int32))
  return data

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

class MyRegressor(chainer.Chain):
  def __init__(self, predictor):
    super(MyRegressor,self).__init__(predictor=predictor)

  def __call__(self, xs, ts):
    global vocab
    self.ys = None
    self.loss = None
    self.ys = self.predictor(xs = xs, ts = ts)
    eos = np.array([vocab["<eos>"]], dtype=np.int32)
    ts = [F.concat((t, eos), axis=0) for t in ts]
    # 損失を計算する
    for y, t in zip(self.ys, ts):
      loss = F.softmax_cross_entropy(y, t)
      self.loss = loss if self.loss is None else self.loss + loss
    self.loss = self.loss / len(xs)

    reporter.report({'loss':self.loss}, self)
    return self.loss

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

def converter(batch,device):
  xs = []
  ts = []
  for b in batch:
    x = b[0]
    t = b[1]
    xs.append(to_device(device,x))
    ts.append(to_device(device,t))
  return (xs,ts)

# 単語とidの辞書
with open("data/vocab.dump", "rb") as f:
    vocab = pickle.load(f)
with open("data/rvocab.dump", "rb") as f:
    rvocab = pickle.load(f)

x_file = "data/x.txt"
t_file = "data/t.txt"

xs = load_data(x_file)
ts = load_data(t_file)

dataset = TupleDataset(xs,ts)

n_train = int(len(dataset) * 0.7)
n_valid = int(len(dataset) * 0.1)
train, valid_test = split_dataset_random(dataset, n_train, seed=0)
valid, test = split_dataset_random(valid_test, n_valid, seed=0)

batch_size = 32
n_epoch=500
device = 0

train_iter = iterators.SerialIterator(train, batch_size)
valid_iter = iterators.SerialIterator(valid, batch_size, shuffle=False, repeat=False)

predictor = Seq2seq(n_vocab=len(vocab))

if device >= 0:
    predictor.to_gpu(device)

model = MyRegressor(predictor)
optimizer = optimizers.Adam().setup(model)
updater = training.StandardUpdater(train_iter, optimizer, converter=converter, device=device)

trigger = EarlyStoppingTrigger(monitor='val/main/loss', check_trigger=(1, 'epoch'),
                               patients=10, max_trigger=(n_epoch, 'epoch'))
trainer = training.Trainer(updater, trigger, out='results/seq2seq')

trainer.extend(extensions.LogReport(trigger=(1, 'epoch'), log_name='log'))
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.Evaluator(valid_iter, model, converter=converter, device=device), name='val')
trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'val/main/loss', 'elapsed_time']))
trainer.extend(extensions.PlotReport(['main/loss','val/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.ParameterStatistics(model.predictor.W, {'mean': np.mean}, report_grads=True))

trainer.run()

gen_model = Generator(predictor = predictor, device= device, max_size = 30)
with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
  ys_list = gen_model(test)
  for ys in ys_list:
    for y in ys:
      y = int(y)
      if y is vocab['<eos>']:
        print('\n')
        break
      print(rvocab[y],end='')

if device >= 0:
    predictor.to_cpu()
save_npz('results/seq2seq/predictor.npz', predictor)
