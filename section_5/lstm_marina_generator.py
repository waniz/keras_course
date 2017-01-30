import numpy as np
import re
import random
import sys
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop


filename = "data/war_peace.txt"
raw_text = open(filename, encoding="utf-8").read()
raw_text = raw_text.lower()

raw_text_ru = re.sub("[^а-я ,.!–?]", "", raw_text)

chars = sorted(list(set(raw_text_ru)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: %s" % n_chars)
print("Total Vocab: %s" % n_vocab)

maxlen = 50
step = 3
sentences = []
next_chars = []
for i in range(0, len(raw_text_ru) - maxlen, step):
    sentences.append(raw_text_ru[i: i + maxlen])
    next_chars.append(raw_text_ru[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1

print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(chars), activation='softmax'))
optimizer = RMSprop(lr=0.0005)

filename = "models/tolstoy/weights_034_10.5998.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds_, temperature=1.0):
    # helper function to sample an index from a probability array
    preds_ = np.asarray(preds_).astype('float64')
    preds_ = np.log(preds_) / temperature
    exp_preds = np.exp(preds_)
    preds_ = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds_, 1)
    return np.argmax(probas)


start_index = random.randint(0, len(raw_text_ru) - maxlen - 1)
for diversity in [1.0]:  # different 'meaning' + repeating himself
    print()

    generated = ''
    sentence = raw_text_ru[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"' + ' ------------')

    for i in range(2000):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_to_int[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = int_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
