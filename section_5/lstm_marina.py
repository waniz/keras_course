import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, TimeDistributed, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop


filename = "data/war_peace.txt"
raw_text = open(filename, encoding="utf-8").read()
raw_text = raw_text.lower()

raw_text_ru = re.sub("[^а-я ,.!–?]", "", raw_text)

chars = sorted(list(set(raw_text_ru)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: %s" % n_chars)
print("Total Vocab: %s" % n_vocab)

maxlen = 50
step = 1
sentences = []
next_chars = []
for i in range(0, len(raw_text_ru) - maxlen, step):
    sentences.append(raw_text_ru[i: i + maxlen])
    next_chars.append(raw_text_ru[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
sentences_train = sentences[:int(len(sentences) * 0.8)]
sentences_validation = sentences[int(len(sentences) * 0.8):]

X = np.zeros((len(sentences_train), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences_train), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences_train):
    for t, char in enumerate(sentence):
        X[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1

X_val = np.zeros((len(sentences_validation), maxlen, len(chars)), dtype=np.bool)
y_val = np.zeros((len(sentences_validation), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences_validation):
    for t, char in enumerate(sentence):
        X_val[i, t, char_to_int[char]] = 1
    y_val[i, char_to_int[next_chars[i]]] = 1

print('Build model...')
model = Sequential()
model.add(LSTM(8, input_shape=(maxlen, len(chars))))
model.add(TimeDistributed(Dense(8)))
model.add(TimeDistributed(Activation('relu')))
model.add(Dense(len(chars), activation='softmax'))
optimizer = RMSprop(lr=0.0005)

# filename = "models/tolstoy/weights_2.5462.hdf5"
# model.load_weights(filename)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

filepath = "models/tolstoy/weights_{epoch:03d}_{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
model.fit(X, y, nb_epoch=500, batch_size=50, callbacks=[checkpoint], validation_data=(X_val, y_val))
