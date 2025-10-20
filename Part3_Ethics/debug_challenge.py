# Debug Challenge: Fix TensorFlow Script with Errors

# ORIGINAL BUGGY CODE (with intentional errors)
"""
import tensorflow as tf
import numpy as np

# Error 1: Incorrect data loading
x_train = np.random.random((1000, 28, 28))  # Missing channel dimension
y_train = np.random.randint(0, 10, (1000,))  # Missing one-hot encoding

# Error 2: Incorrect model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28)),  # Wrong input shape
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # Wrong output layer for binary classification
])

# Error 3: Incorrect compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Wrong loss function

# Error 4: Incorrect training
model.fit(x_train, y_train, epochs=10, batch_size=32)  # Missing validation data

# Error 5: Incorrect evaluation
predictions = model.predict(x_train)
accuracy = tf.keras.metrics.accuracy(y_train, predictions)  # Wrong usage of accuracy metric
print(f"Accuracy: {accuracy}")
"""

# CORRECTED CODE
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

print("=== DEBUGGING CHALLENGE: FIXED TENSORFLOW SCRIPT ===")
print("Original errors identified and corrected:")
print("1. ❌ Missing channel dimension in input data")
print("2. ❌ Wrong input shape in Conv2D layer")
print("3. ❌ Wrong loss function for multi-class classification")
print("4. ❌ Missing validation data in training")
print("5. ❌ Incorrect accuracy metric usage")
print()

# Load a real dataset for demonstration
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# FIX 1: Correct data preprocessing
print("Fixing data preprocessing...")
# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add channel dimension
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print(f"✅ Fixed: Data shapes - X: {x_train.shape}, Y: {y_train.shape}")

# FIX 2: Correct model architecture
print("Fixing model architecture...")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Fixed input shape
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Added dropout for regularization
    tf.keras.layers.Dense(10, activation='softmax')  # Correct for 10-class classification
])

print("✅ Fixed: Model architecture with correct input shape and output layer")

# FIX 3: Correct compilation
print("Fixing model compilation...")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Fixed loss function for multi-class
    metrics=['accuracy']
)

print("✅ Fixed: Compilation with correct loss function")

# FIX 4: Correct training with validation
print("Fixing training process...")
# Split training data for validation
x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

# Train with validation data
history = model.fit(
    x_train_split, y_train_split,
    validation_data=(x_val, y_val),
    epochs=5,  # Reduced for demonstration
    batch_size=32,
    verbose=1
)

print("✅ Fixed: Training with proper validation data")

# FIX 5: Correct evaluation
print("Fixing evaluation...")
# Evaluate on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

# Make predictions
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Calculate accuracy correctly
accuracy = tf.keras.metrics.Accuracy()
accuracy.update_state(true_classes, predicted_classes)
final_accuracy = accuracy.result().numpy()

print(f"✅ Fixed: Correct evaluation metrics")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Calculated Accuracy: {final_accuracy:.4f}")

# Additional improvements
print("\n=== ADDITIONAL IMPROVEMENTS MADE ===")
print("✅ Added data normalization")
print("✅ Added dropout for regularization")
print("✅ Used proper train/validation split")
print("✅ Added proper evaluation metrics")
print("✅ Used real MNIST dataset instead of random data")

# Model summary
print("\n=== MODEL SUMMARY ===")
model.summary()

print("\n=== DEBUGGING CHALLENGE COMPLETE ===")
print("All errors have been identified and fixed!")
print("The model now trains and evaluates correctly.")

