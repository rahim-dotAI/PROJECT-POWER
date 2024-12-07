import keras
from keras.models import Sequential
from keras.layers import Dense

# Create a simple Sequential model
model = Sequential()
model.add(Input(shape=(784,)))  # Define input shape with Input layer
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

print("Keras is working fine!")
