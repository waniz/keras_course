import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV


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


def fit_model(neirons=1):
    model = Sequential()
    model.add(Dense(input_dim=train_dataset.shape[1], output_dim=train_dataset.shape[1], activation='relu'))
    model.add(Dense(neirons, activation='relu'))
    model.add(Dense(output_dim=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=fit_model, nb_epoch=100, batch_size=10, verbose=0)
neirons = [1, 2, 3, 4, 5, 6, 7, 8]
param_grid = dict(neirons=neirons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))


