# Learning rate in a neural network:
from keras.optimizers import Adam

# Compile the model with a learning rate of 0.001
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Number of iterations in k-means clustering:
from sklearn.cluster import KMeans

# Define the model with 20 iterations
kmeans = KMeans(n_clusters=3, max_iter=20)
