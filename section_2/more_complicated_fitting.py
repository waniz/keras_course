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


def fit_model(input_layer_neirons=1, hidden_layer_neirons=1):
    model = Sequential()
    model.add(Dense(input_dim=train_dataset.shape[1], output_dim=input_layer_neirons, activation='relu'))
    model.add(Dense(hidden_layer_neirons, activation='relu'))
    model.add(Dense(output_dim=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=fit_model, nb_epoch=100, batch_size=10, verbose=1)
input_layer_neirons = [6, 8, 12]
hidden_layer_neirons = [2, 3, 4, 5, 6]
batch_size = [10, 50]
epochs = [100]
param_grid = dict(batch_size=batch_size, nb_epoch=epochs,
                  input_layer_neirons=input_layer_neirons, hidden_layer_neirons=hidden_layer_neirons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

# available options:
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
# momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
# weight_constraint = [1, 2, 3, 4, 5]
# dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# optimizer = SGD(lr=learn_rate, momentum=momentum)
# model.add(Dropout(dropout_rate))

# results:
# Best: 0.697068 using {'hidden_layer_neirons': 5, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 8}
# 0.351754 (0.017540) with: {'hidden_layer_neirons': 2, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 6}
# 0.646612 (0.015388) with: {'hidden_layer_neirons': 2, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 8}
# 0.667759 (0.014198) with: {'hidden_layer_neirons': 2, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 12}
# 0.644994 (0.019718) with: {'hidden_layer_neirons': 3, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 6}
# 0.685645 (0.018785) with: {'hidden_layer_neirons': 3, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 8}
# 0.653196 (0.060840) with: {'hidden_layer_neirons': 3, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 12}
# 0.544150 (0.136736) with: {'hidden_layer_neirons': 4, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 6}
# 0.680807 (0.011510) with: {'hidden_layer_neirons': 4, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 8}
# 0.679181 (0.024953) with: {'hidden_layer_neirons': 4, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 12}
# 0.682409 (0.015948) with: {'hidden_layer_neirons': 5, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 6}
# 0.697051 (0.008598) with: {'hidden_layer_neirons': 5, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 8}
# 0.473705 (0.176384) with: {'hidden_layer_neirons': 5, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 12}
# 0.674263 (0.021989) with: {'hidden_layer_neirons': 6, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 6}
# 0.677507 (0.014196) with: {'hidden_layer_neirons': 6, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 8}
# 0.688881 (0.055214) with: {'hidden_layer_neirons': 6, 'nb_epoch': 100, 'batch_size': 10, 'input_layer_neirons': 12}
# 0.651498 (0.017336) with: {'hidden_layer_neirons': 2, 'nb_epoch': 100, 'batch_size': 50, 'input_layer_neirons': 6}
# 0.661255 (0.012342) with: {'hidden_layer_neirons': 2, 'nb_epoch': 100, 'batch_size': 50, 'input_layer_neirons': 8}
# 0.653132 (0.016622) with: {'hidden_layer_neirons': 2, 'nb_epoch': 100, 'batch_size': 50, 'input_layer_neirons': 12}
# 0.640069 (0.008102) with: {'hidden_layer_neirons': 3, 'nb_epoch': 100, 'batch_size': 50, 'input_layer_neirons': 6}
# 0.662873 (0.022070) with: {'hidden_layer_neirons': 3, 'nb_epoch': 100, 'batch_size': 50, 'input_layer_neirons': 8}
# 0.651506 (0.018096) with: {'hidden_layer_neirons': 3, 'nb_epoch': 100, 'batch_size': 50, 'input_layer_neirons': 12}
# 0.651506 (0.019366) with: {'hidden_layer_neirons': 4, 'nb_epoch': 100, 'batch_size': 50, 'input_layer_neirons': 6}
# 0.641671 (0.018839) with: {'hidden_layer_neirons': 4, 'nb_epoch': 100, 'batch_size': 50, 'input_layer_neirons': 8}
# 0.545249 (0.154602) with: {'hidden_layer_neirons': 4, 'nb_epoch': 100, 'batch_size': 50, 'input_layer_neirons': 12}
# 0.669385 (0.004272) with: {'hidden_layer_neirons': 5, 'nb_epoch': 100, 'batch_size': 50, 'input_layer_neirons': 6}
# 0.659629 (0.014631) with: {'hidden_layer_neirons': 5, 'nb_epoch': 100, 'batch_size': 50, 'input_layer_neirons': 8}
# 0.566388 (0.168367) with: {'hidden_layer_neirons': 5, 'nb_epoch': 100, 'batch_size': 50, 'input_layer_neirons': 12}
# 0.662881 (0.017075) with: {'hidden_layer_neirons': 6, 'nb_epoch': 100, 'batch_size': 50, 'input_layer_neirons': 6}
# 0.677515 (0.038071) with: {'hidden_layer_neirons': 6, 'nb_epoch': 100, 'batch_size': 50, 'input_layer_neirons': 8}
# 0.672692 (0.025494) with: {'hidden_layer_neirons': 6, 'nb_epoch': 100, 'batch_size': 50, 'input_layer_neirons': 12}