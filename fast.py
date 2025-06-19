import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# Load and preprocess the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),      # Flatten input images
    tf.keras.layers.Dense(128, activation='relu'),      # Hidden layer with ReLU activation
    tf.keras.layers.Dropout(0.2),                       # Dropout for regularization
    tf.keras.layers.Dense(10, activation='softmax')     # Output layer with softmax
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

# Predict a few test images
predictions = model.predict(x_test)

# Function to display image with prediction
def display_prediction(index):
    plt.imshow(x_test[index], cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[index])}, Actual: {y_test[index]}")
    plt.axis('off')
    plt.show()

# Example usage: display prediction for the 0th test image
display_prediction(0)
# After training and evaluating
model.save('saved_model.keras')
