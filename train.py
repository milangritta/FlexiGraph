# -*- coding: utf-8 -*-
import codecs
import numpy as np
import cPickle
from keras import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine import Model
from keras.layers.merge import concatenate
from keras.layers import Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from preprocessing import BATCH_SIZE, EMBEDDING_DIMENSION, CONTEXT_LENGTH, UNKNOWN
from preprocessing import generate_arrays_from_file, ENCODING_MAP_2x2, ENCODING_MAP_1x1
from subprocess import check_output

print(u"Dimension:", EMBEDDING_DIMENSION)
print(u"Context length:", CONTEXT_LENGTH)
word_to_index = cPickle.load(open(u"data/w2i.pkl"))
print(u"Vocabulary Size:", len(word_to_index))

vectors = {UNKNOWN: np.ones(EMBEDDING_DIMENSION), u'0': np.ones(EMBEDDING_DIMENSION)}
for line in codecs.open(u"../data/glove.twitter." + str(EMBEDDING_DIMENSION) + u"d.txt", encoding=u"utf-8"):
    if line.strip() == "":
        continue
    t = line.split()
    vectors[t[0]] = [float(x) for x in t[1:]]
print(u'Vectors...', len(vectors))

weights = np.zeros((len(word_to_index), EMBEDDING_DIMENSION))
oov = 0
for w in word_to_index:
    if w in vectors:
        weights[word_to_index[w]] = vectors[w]
    else:
        weights[word_to_index[w]] = np.random.normal(size=(EMBEDDING_DIMENSION,), scale=0.3)
        oov += 1

weights = np.array([weights])
print(u'Done preparing vectors...')
print(u"OOV (no vectors):", oov)
#  --------------------------------------------------------------------------------------------------------------------
print(u'Building model...')
embeddings = Embedding(len(word_to_index), EMBEDDING_DIMENSION, input_length=CONTEXT_LENGTH, weights=weights)

context_words_single = Input(shape=(CONTEXT_LENGTH,))
cws = embeddings(context_words_single)
cws = Conv1D(2500, 1, activation='relu', strides=1)(cws)
cws = GlobalMaxPooling1D()(cws)
cws = Dense(250)(cws)
cws = Dropout(0.5)(cws)

context_words_pair = Input(shape=(CONTEXT_LENGTH,))
cwp = embeddings(context_words_pair)
cwp = Conv1D(1000, 2, activation='relu', strides=1)(cwp)
cwp = GlobalMaxPooling1D()(cwp)
cwp = Dense(250)(cwp)
cwp = Dropout(0.5)(cwp)

context_words_triple = Input(shape=(CONTEXT_LENGTH,))
cwt = embeddings(context_words_triple)
cwt = Conv1D(500, 1, activation='relu', strides=1)(cwt)
cwt = GlobalMaxPooling1D()(cwt)
cwt = Dense(250)(cwt)
cwt = Dropout(0.5)(cwt)

loc2vec = Input(shape=(len(ENCODING_MAP_1x1),))
l2v = Dense(5000, activation='relu', input_dim=len(ENCODING_MAP_1x1))(loc2vec)
l2v = Dense(1000, activation='relu')(l2v)
l2v = Dropout(0.5)(l2v)

inp = concatenate([cwp, cws, cwt, l2v])
inp = Dense(units=len(ENCODING_MAP_2x2), activation=u'softmax')(inp)
model = Model(inputs=[context_words_pair, context_words_single, context_words_triple, loc2vec], outputs=[inp])
model.compile(loss=u'categorical_crossentropy', optimizer=u'rmsprop', metrics=[u'accuracy'])

print(u'Finished building model...')
#  --------------------------------------------------------------------------------------------------------------------
checkpoint = ModelCheckpoint(filepath=u"../data/weights.{epoch:02d}-{acc:.2f}.hdf5", verbose=0)
early_stop = EarlyStopping(monitor=u'acc', patience=5)
file_name = u"../data/WikiGeolocateLongTrain.txt"
model.fit_generator(generate_arrays_from_file(file_name, word_to_index),
                    steps_per_epoch=int(check_output(["wc", file_name]).split()[0]) / BATCH_SIZE,
                    epochs=250, callbacks=[checkpoint, early_stop])
