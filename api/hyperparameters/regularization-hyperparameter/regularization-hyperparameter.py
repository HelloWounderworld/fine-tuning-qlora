# L2 regularization strength in a linear regression:
from sklearn.linear_model import Ridge

# Define the model with a regularization strength of 0.1
lr = Ridge(alpha=0.1)

# Dropout rate in a neural network:
from keras.layers import Dropout

# Define the model with a dropout rate of 0.5
model = Sequential()
model.add(Dense(64, input_dim=input_shape, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(output_shape, activation='softmax'))
