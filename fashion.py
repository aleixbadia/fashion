import os
import tensorflow as tf

program_directory_name = os.path.dirname(os.path.abspath(__file__))
os.chdir(program_directory_name)

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Preprocess Fashion MNIST data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Create the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # Update input shape
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test, y_test)

# Save the model
model.save('fashion_mnist_model.h5')