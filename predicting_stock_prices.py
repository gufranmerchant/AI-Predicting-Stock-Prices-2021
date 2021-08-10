import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# The Pandas datareader is a sub package that allows one to create a dataframe from various internet datasources, currently including: Yahoo!
import pandas_datareader as web

# The datetime module supplies classes for manipulating dates and times.
# While date and time arithmetic is supported, the focus of the implementation is on efficient attribute extraction for output formatting and manipulation.
# General calendar related functions.
import datetime as dt


# Transform features by scaling each feature to a given range.
# This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.
# This transformation is often used as an alternative to zero mean, unit variance scaling.
from sklearn.preprocessing import MinMaxScaler

#A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
#A Sequential model is not appropriate when:
#Your model has multiple inputs or multiple outputs
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, LSTM #Long Short Term Memory

# Loading the Data
company = 'AAPL'

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)

# data = web.DataReader(company, 'yahoo', start, end)

data = yf.Ticker(company)
hist = data.history(start=start, end=end)
hist

# Prepare the Data
# for this we will scale down all the values that we have so they fit in between 0 and 1.
# So if we have lowest price of 10$ and the highest price of 200$ then we will scale those values to fit in between 0 and 1. So...

scaler = MinMaxScaler(feature_range=(0,1))

# we are not going to transform the whole dataframe,we will only predict the closing prices, so when the markets are closed, hence we will only transform the 'Close'
# fit_transform() is used on the training data so that we can scale the training data and also learn the scaling parameters of that data.
# Here, the model built by us will learn the mean and variance of the features of the training set.
# These learned parameters are then used to scale our test data.
# So what actually is happening here!
# The fit method is calculating the mean and variance of each of the features present in our data.
# The transform method is transforming all the features using the respective mean and variance.

scaled_data = scaler.fit_transform(hist['Close'].values.reshape(-1,1))

# How many days i want to base my prediction on, how many days do i wanna look back on
prediction_days = 20

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

# Converting x_train and y_train into numpy array
x_train, y_train = np.array(x_train), np.array(y_train) # shape of x_train is "ndarray with shape (1857, 60)", which means that the array has 1857 dimensions, and each dimension has 60 elements.

# reshape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Building the Deep Learning Model
model = Sequential()

# first layer with 50 units, feedback if keep to true and we also defined the input shape as this is the first layer
model.add(LSTM(units=50, return_sequences=True, input_shape =(x_train.shape[1], 1)))
# dropout is used to prevent overfitting with freq of 0.2
model.add(Dropout(0.2))


# Second layer without defining the input shape as we already did for the first layer and the output of the first will be the input for the second i think
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# and this is the third lstm layer with no return sequence as it will be fed into the last layer or the dense layer
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# prediction of the next closing value
model.add(Dense(units=1))


# Compiling the model
# Adam is a replacement optimization algorithm for stochastic gradient descent for training deep learning models.
# Adam combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems.
model.compile(optimizer='adam', loss='mean_squared_error')

# The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.
# The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.
model.fit(x_train, y_train, batch_size=32, epochs=70)


''' Test the Model Accuracy on the Existing Data '''

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = yf.Ticker(company)
test_hist = test_data.history(start=test_start, end=test_end)

actual_prices = test_hist['Close'].values

total_dataset = pd.concat((hist['Close'], test_hist['Close']), axis=0)

# we will only consider the recent values, hence we are doing the operation on the total dataset(to get the latest index') and assigning it to model input

# len(total_dataset) - len(test_hist) - prediction_days = 1857, till the end
model_inputs = total_dataset[len(total_dataset) - len(test_hist) - prediction_days:].values

model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)


# Make predictions on test data
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.plot(actual_prices, color='black', label=f'Actual {company} Prices')
plt.plot(predicted_prices, color='red', label=f'Predicted {company} Price')
plt.title(f'{company} Share Price')
plt.xlabel('Time')
plt.ylabel(f' {company} Share Price')
plt.legend()
plt.show()

# len(model_inputs)+1-prediction_days: len(model_inputs+1)  #404:463
# 463-404

# Predict the next day

real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

# [[145.69354]]
print(prediction)



