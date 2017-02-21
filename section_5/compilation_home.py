import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import random
import sys


filename = "data/compilation.txt"
raw_text = open(filename, encoding="utf-8").read()
raw_text = raw_text.lower()

raw_text_ru = re.sub("[^а-я,. ]", "", raw_text)

chars = sorted(list(set(raw_text_ru)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: %s" % n_chars)
print("Total Vocab: %s" % n_vocab)

maxlen = 80
step = 7
batch_size = 1000

sentences = []
next_chars = []
for i in range(0, len(raw_text_ru) - maxlen, step):
    sentences.append(raw_text_ru[i: i + maxlen])
    next_chars.append(raw_text_ru[i + maxlen])
print('nb sequences:', len(sentences))
val_sentences = sentences[5900000:6100000]
sentences = sentences[:5900000]


print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1

# X_val = np.zeros((len(val_sentences), maxlen, len(chars)), dtype=np.bool)
# y_val = np.zeros((len(val_sentences), len(chars)), dtype=np.bool)
# for i, sentence in enumerate(val_sentences):
#     for t, char in enumerate(val_sentences):
#         X_val[i, t, char_to_int[char]] = 1
#     y_val[i, char_to_int[next_chars[i]]] = 1

print('Build model...')
model = Sequential()
model.add(LSTM(256, batch_input_shape=(batch_size, maxlen, len(chars)), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256, batch_input_shape=(batch_size, maxlen, len(chars)), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256, batch_input_shape=(batch_size, maxlen, len(chars)), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(output_dim=len(chars), activation='softmax'))
optimizer = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()

# filename = "models/compilation/weights_031_1.0184.hdf5"
# model.load_weights(filename)


def sample(a, temperature=1.0):
    a = np.log(a)/temperature
    a = np.exp(a)/np.sum(np.exp(a))
    if sum(a) > 1.0:
        a *= 1 - (sum(a) - 1)
        if sum(a) > 1.0:
            a *= 0.999999
    return np.argmax(np.random.multinomial(1, a, 1))


for iteration in range(1, 1000):
    print("==============================================================")
    print("Iteration: ", iteration)
    filepath = "models/star_wars/weights_%s_{loss:.4f}.hdf5" % iteration
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')

    model.fit(X, y, batch_size=batch_size, nb_epoch=1, callbacks=[checkpoint], shuffle=False)

    start_index = random.randint(0, len(raw_text_ru) - maxlen - 1)
    for T in [0.2, 0.5, 1.0, 1.2]:
        print("------------Temperature", T)
        generated = ''
        sentence = raw_text_ru[start_index:start_index + maxlen]
        generated += sentence
        print("Generating with seed: " + sentence)
        print('')

        for i in range(1000):
            seed = np.zeros((batch_size, maxlen, len(chars)))
            # format input
            for t, char in enumerate(sentence):
                seed[0, t, char_to_int[char]] = 1

            # get predictions
            predictions = model.predict(seed, batch_size=batch_size, verbose=2)[0]
            next_index = sample(predictions, T)
            next_char = int_to_char[next_index]
            # print next char
            sys.stdout.write(next_char)
            sys.stdout.flush()

            # use current output as input to predict the next character
            # in the sequence
            generated += next_char
            sentence = sentence[1:] + next_char
        print()

