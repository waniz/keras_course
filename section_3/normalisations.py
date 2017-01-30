import pandas as pd
# 2
from keras.models import Sequential
from keras.layers import Dense
# 3
from keras.metrics import mean_squared_error, mean_absolute_error
import math
# 4
import matplotlib.pyplot as plt


data_file = 'data/energy_efficiency.csv'

raw_data = pd.read_csv(data_file)
data = pd.DataFrame()

# print(raw_data[:3])

# for column in raw_data.columns:
#     print(type(raw_data[column][1]))

columns_to_numpy = ['X1', 'X2', 'X3', 'X4', 'X5', 'X7', 'Y1', 'Y2']

for column in columns_to_numpy:
    data[column] = raw_data[column].apply(lambda x: float(x.replace(',', '.')))

data['X6'] = raw_data['X6']
data['X8'] = raw_data['X8']
print(data.shape)

train = data[:600]
test = data[600:]
print(train.shape, test.shape)

y_train = train[['Y1']].values
y_test = test[['Y1']].values
x_train = train[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']].values
x_test = test[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']].values

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = Sequential()
model.add(Dense(input_dim=x_train.shape[1], output_dim=x_train.shape[1], activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(output_dim=1, activation='softplus'))
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, nb_epoch=10, batch_size=1, validation_data=(x_test, y_test))

predicts = model.predict(x_test)
print(predicts)
# print('RMSE Y1: %s' % math.sqrt(mean_squared_error(data['Y1'], predicts[0])))
# print('RMSE Y2: %s' % math.sqrt(mean_squared_error(data['Y2'], predicts[1])))

plot_frame = pd.concat([data['Y1'], pd.DataFrame(predicts[0])], axis=1)
plot_frame.plot()
plt.show()



