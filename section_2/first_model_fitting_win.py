import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)

data = pd.read_csv('data/credit_train_equal.csv')
data = data.drop('client_id', axis=1)

usable_features = [
    'credit_sum',
    'score_shk',
    'age',
    'monthly_income',
    'tariff_id',
    'credit_count',
    'credit_month',
    'gender',
    'marital_status_MAR',
    'job_position_SPC',
    'marital_status_UNM',
    'education_GRD',
    'education_SCH',
    'overdue_credit_count'
          ]

X = data.as_matrix(usable_features)
Y = data['open_account_flg'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)


def opt_model(neurons=1):
    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], output_dim=36, activation='relu'))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


def optimizer():
    model_opt = KerasClassifier(build_fn=opt_model, nb_epoch=100, batch_size=16, verbose=1)
    hidden_neuron = [4, 8, 12, 16, 32, 64]
    param_grid = dict(neurons=hidden_neuron)
    grid = GridSearchCV(estimator=model_opt, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.grid_scores_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))


if __name__ == '__main__':
    optimizer()
