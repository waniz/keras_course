import pandas as pd
from keras.models import Sequential
from keras.layers import Dense


dataset = pd.read_csv('data/pima-indians-diabetes.data',  delimiter=',',
                      names=['1', '2', '3', '4', '5', '6', '7', 'target'])
print(dataset.shape)
dataset = dataset.sample(frac=1)

train_test_value = int(len(dataset) * 0.8)
train_dataset = dataset[:train_test_value].copy()
test_dataset = dataset[train_test_value:].copy()
print(train_dataset.shape, test_dataset.shape)

y_train = train_dataset.as_matrix(columns=['target'])
y_test = test_dataset.as_matrix(columns=['target'])
train_dataset.drop(['target'], axis=1, inplace=True)
test_dataset.drop(['target'], axis=1, inplace=True)
x_train = train_dataset.values
x_test = test_dataset.values

model = Sequential()
model.add(Dense(input_dim=train_dataset.shape[1], output_dim=train_dataset.shape[1], activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(output_dim=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, nb_epoch=300, batch_size=1, validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, batch_size=1)

print('')
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


