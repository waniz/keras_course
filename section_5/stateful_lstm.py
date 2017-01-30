import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop


filename = "data/alice.txt"
raw_text = open(filename, encoding="utf-8").read()
raw_text = raw_text.lower()

# raw_text_ru = re.sub("[^а-я ,.!–?]", "", raw_text)
raw_text_ru = re.sub("[^a-z ,.!–?]", "", raw_text)

chars = sorted(list(set(raw_text_ru)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: %s" % n_chars)
print("Total Vocab: %s" % n_vocab)

maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(raw_text_ru) - maxlen, step):
    sentences.append(raw_text_ru[i: i + maxlen])
    next_chars.append(raw_text_ru[i + maxlen])

valid_sentences = sentences[10000:20000]
sentences = sentences[:10000]

print('nb sequences:', len(sentences))
print('Vectorization...')

X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1

X_val = np.zeros((len(valid_sentences), maxlen, len(chars)), dtype=np.bool)
y_val = np.zeros((len(valid_sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(valid_sentences):
    for t, char in enumerate(sentence):
        X_val[i, t, char_to_int[char]] = 1
    y_val[i, char_to_int[next_chars[i]]] = 1

print('Build model...')
model = Sequential()
model.add(LSTM(31, batch_input_shape=(len(sentences), maxlen, len(chars)), stateful=True,
               return_sequences=True))
model.add(LSTM(31, batch_input_shape=(len(sentences), maxlen, len(chars)), stateful=True))

model.add(Dense(len(chars), activation='softmax'))
optimizer = RMSprop(lr=0.01)

# filename = "models/tolstoy/weights_2.5462.hdf5"
# model.load_weights(filename)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

filepath = "models/stateful/weights_{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
for i in range(50):
    model.fit(X, y, nb_epoch=1, batch_size=len(sentences), callbacks=[checkpoint], shuffle=False,
              validation_data=(X_val, y_val), verbose=2)
    model.reset_states()
# model.fit(X, y, nb_epoch=10, batch_size=1, callbacks=[checkpoint], validation_split=0.3)
