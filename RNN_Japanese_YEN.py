

# LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# PART 1 : DATAS
## Training datas
dataset_train = pd.read_csv("JPY_currency_train.csv")
training_set = dataset_train[["JPY"]].values

## Scaling by normalisation (because of sigmoid)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

## Structure creation with timesteps (60) and output
X_train = []
y_train = []
for i in range (60, 5136):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)

## Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# PART 2 : RNN
regressor = Sequential()

## First layer
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

## Second layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

## Third layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

## Fourth layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

## Output layer
regressor.add(Dense(units=1))

## Compile
regressor.compile(optimizer="rmsprop", loss="mean_squared_error")

## Training
regressor.fit(X_train, y_train, epochs=100, batch_size=32)




# PART 3 : PREDICTIONS AND VISUALISATION
## Predictions
dataset_test = pd.read_csv("JPY_currency_test.csv")
test_set = dataset_test[["JPY"]].values
dataset = pd.concat((dataset_train["Euro"], dataset_test["JPY"]), axis=0)
inputs = dataset[len(dataset) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range (60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_datas = regressor.predict(X_test)
predicted_datas = sc.inverse_transform(predicted_datas)

## Visualisation
plt.plot(test_set, color="red", label="Real Price")
plt.plot(predicted_datas, color="blue", label="Predicted price")
plt.title("Predictions for the JPY currency")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()


predicted_datas = regressor.predict(X_train)
predicted_datas = sc.inverse_transform(predicted_datas)
plt.plot(training_set, color="red", label="Real Price")
plt.plot(predicted_datas, color="blue", label="Predicted price")
plt.title("Predictions for the JPY currency")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()



dataset_train[['JPY']].plot(figsize=(16,6))






