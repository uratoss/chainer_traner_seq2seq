import logging
from gensim.models import word2vec
import sys

import numpy as xp


class W2v():
    __model = None
    __vocab_index = None

    def __init__(self, model_path=None):
        if not (model_path is None):
            try:
                self.__model = word2vec.Word2Vec.load(model_path)
                self.__vocab_index = self.__model.wv.index2word
            except FileNotFoundError as e:
                print(e)

    def train(self, corpus_path, size=100, min_count=1, window=5, iter=20, out_path='./model/word2vec/word2vec.model'):
        logging.basicConfig(
            format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
        try:
            sentences = word2vec.PathLineSentences(corpus_path)
        except Exception as e:
            print(e)
            return

        if self.__model is None:
            self.__model = word2vec.Word2Vec(
                sentences, size=size, min_count=min_count, window=window, iter=iter)
        else:
            self.__model.build___vocab(sentences, update=True)
            self.__model.train(
                sentences, total_examples=self.__model.corpus_count, epochs=self.__model.iter)
        self.__vocab_index = self.__model.wv.index2word
        self.__model.save(out_path)

    def getWv(self, word):
        try:
            return self.__model.wv[word]
        except Exception as e:
            # print(e)
            return xp.random.uniform(-1.0, 1.0, (1, self.__model.vector_size))[0]

    def getWvID(self, word_id):
        return self.getWv(self.getWord(word_id))

    def getVocab(self):
        return self.__vocab_index

    def getWord(self, word_id):
        try:
            return self.__vocab_index[word_id]
        except Exception as e:
            # print(e)
            return '<unk>'

    def getID(self, word):
        try:
            return self.__vocab_index.index(word)
        except Exception as e:
            print(e)
            return -1

    def getVectorSize(self):
        return self.__model.vector_size


class EmbedID(W2v):
    def __init__(self, model_path=None):
        super().__init__(model_path)

    def __call__(self, sequence):
        try:
            embed_sequence = []
            for word in sequence:
                embed_sequence.append(super().getWvID(word))
            return xp.array(embed_sequence, dtype=xp.float32)
        except:
            return


if __name__ == '__main__':
    w2v = W2v()
    w2v.train('./data', size=200, iter=50)
