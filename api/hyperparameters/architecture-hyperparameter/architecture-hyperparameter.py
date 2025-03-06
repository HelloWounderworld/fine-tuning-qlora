# Number of hidden layers in a neural network:
from keras.models import Sequential
from keras.layers import Dense

# Define the model with 2 hidden layers and 64 neurons per layer
model = Sequential()
model.add(Dense(64, input_dim=input_shape, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(output_shape, activation='softmax'))

# Number of trees in a random forest:
from sklearn.ensemble import RandomForestClassifier

# Define the model with 10 trees
rf = RandomForestClassifier(n_estimators=10)
