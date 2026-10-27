import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Fix random seed for reproducible results
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 1. CREATE 10 SYNTHETIC TRAINING IMAGES (10x10)
# ==========================================
X_train = []
y_train = []

# Class 0: 5 Vertical Line Images
for i in range(5):
    img = np.zeros((10, 10))
    col = np.random.randint(2, 8)  # Draw line at random vertical position
    img[:, col:col+2] = 1.0        # 2-pixel wide line
    img += np.random.normal(0, 0.05, (10, 10)) # Add slight realistic noise
    img = np.clip(img, 0, 1)
    X_train.append(img)
    y_train.append(0)

# Class 1: 5 Horizontal Line Images
for i in range(5):
    img = np.zeros((10, 10))
    row = np.random.randint(2, 8)  # Draw line at random horizontal position
    img[row:row+2, :] = 1.0        # 2-pixel wide line
    img += np.random.normal(0, 0.05, (10, 10)) # Add slight realistic noise
    img = np.clip(img, 0, 1)
    X_train.append(img)
    y_train.append(1)

# Format shape for TensorFlow: (Batch size, Height, Width, Channels)
X_train = np.array(X_train).reshape(10, 10, 10, 1)
y_train = np.array(y_train)

# ==========================================
# 2. BUILD THE CNN ARCHITECTURE
# ==========================================
model = models.Sequential([
    # Layer 1: Conv2D with 2 filters of 3x3
    layers.Conv2D(filters=2, kernel_size=(3, 3), activation='relu', input_shape=(10, 10, 1)),
    
    # Layer 2: Max Pooling (shrinks 8x8 -> 4x4)
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Layer 3: Flatten (turns 3D grid into 1D array)
    layers.Flatten(),
    
    # Layer 4: Dense Hidden Layer
    layers.Dense(4, activation='relu'),
    
    # Layer 5: Output Layer (1 neuron: 0 = Vertical, 1 = Horizontal)
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ==========================================
# 3. TRAIN THE MODEL
# ==========================================
print("Training the CNN...")
history = model.fit(X_train, y_train, epochs=50, verbose=0)
print("Training Complete!\n")

# ==========================================
# 4. TEST WITH UNSEEN IMAGES
# ==========================================
# Test Image 1: Vertical line at col 5
test_v = np.zeros((10, 10))
test_v[:, 5:7] = 1.0

# Test Image 2: Horizontal line at row 3
test_h = np.zeros((10, 10))
test_h[3:5, :] = 1.0

X_test = np.array([test_v, test_h]).reshape(2, 10, 10, 1)

# Run Predictions
predictions = model.predict(X_test)

# Plot & Display Results
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
labels = ['Vertical Line (Class 0)', 'Horizontal Line (Class 1)']

for i in range(2):
    raw_score = predictions[i][0]
    predicted_class = 1 if raw_score >= 0.5 else 0
    confidence = raw_score if predicted_class == 1 else (1 - raw_score)
    
    axes[i].imshow(X_test[i].reshape(10, 10), cmap='gray')
    axes[i].set_title(f"Predicted: {labels[predicted_class]}\nConfidence: {confidence*100:.1f}%")

plt.tight_layout()
plt.show()
