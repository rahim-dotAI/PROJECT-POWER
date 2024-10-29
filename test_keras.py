import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Create a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Print model summary
model.summary()
