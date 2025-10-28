# train.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.ndimage import gaussian_filter, map_coordinates
import cv2
import random

def create_model():
    """Create a multi-output model for digit classification and forgery detection"""
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    digit_output = layers.Dense(10, activation='softmax', name='digit_output')(x)
    forgery_output = layers.Dense(1, activation='sigmoid', name='forgery_output')(x)
    return keras.Model(inputs=inputs, outputs=[digit_output, forgery_output], name='enhanced_forgery_detection_model')

def elastic_transform(image, alpha=34, sigma=4, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)

def add_stroke_variation(image, intensity=0.5):
    if len(image.shape) == 3:
        image = image[:, :, 0]
    rows, cols = image.shape
    result = image.copy()
    for i in range(rows):
        offset = int(intensity * np.sin(i/3.0))
        if 0 <= i+offset < rows:
            result[i] = np.roll(image[i], offset)
    return result.reshape(*image.shape, 1) if len(image.shape) == 3 else result

def cut_and_paste(image, other_image):
    h, w = image.shape[:2]
    y1, y2 = np.random.randint(0, h//2), np.random.randint(h//2, h)
    x1, x2 = np.random.randint(0, w//2), np.random.randint(w//2, w)
    result = image.copy()
    result[y1:y2, x1:x2] = other_image[y1:y2, x1:x2]
    return result

def add_erasures(image, num_erasures=1, max_size=4):
    """Adds black rectangles to simulate erasures."""
    img_erased = image.copy()
    h, w = img_erased.shape[:2]
    for _ in range(num_erasures):
        x1 = np.random.randint(0, w - max_size)
        y1 = np.random.randint(0, h - max_size)
        x2 = x1 + np.random.randint(1, max_size)
        y2 = y1 + np.random.randint(1, max_size)
        img_erased[y1:y2, x1:x2] = 0
    return img_erased

def vary_thickness(image):
    """Randomly apply dilation or erosion to change stroke thickness."""
    kernel = np.ones((2,2), np.uint8)
    if np.random.rand() > 0.5:
        return cv2.dilate(image, kernel, iterations=1)
    else:
        return cv2.erode(image, kernel, iterations=1)

def add_realistic_forgeries(x, y, num_forgeries=3000):
    print(f"Generating {num_forgeries} realistic forgeries...")
    x_forged, y_digit, y_forgery = [], [], []
    digit_to_indices = {i: [j for j, label in enumerate(y) if np.argmax(label) == i] for i in range(10)}
    
    for i in range(num_forgeries):
        original_digit = np.random.randint(0, 10)
        if not digit_to_indices[original_digit]: continue
            
        original_idx = np.random.choice(digit_to_indices[original_digit])
        original_img = x[original_idx].copy()
        forgery_type = np.random.choice(['elastic', 'stroke', 'cut_paste', 'morph', 'noise', 'erasure', 'thickness'])
        
        try:
            if forgery_type == 'elastic':
                img = elastic_transform(original_img, alpha=np.random.uniform(30, 50), sigma=np.random.uniform(4, 6))
            elif forgery_type == 'stroke':
                img = add_stroke_variation(original_img)
            elif forgery_type in ['cut_paste', 'morph']:
                other_digit = np.random.choice([d for d in range(10) if d != original_digit])
                if not digit_to_indices[other_digit]: continue
                other_idx = np.random.choice(digit_to_indices[other_digit])
                other_img = x[other_idx].copy()
                if forgery_type == 'cut_paste':
                    img = cut_and_paste(original_img, other_img)
                else:
                    img = (1 - np.random.uniform(0.3, 0.7)) * original_img + np.random.uniform(0.3, 0.7) * other_img
            elif forgery_type == 'erasure':
                img = add_erasures(original_img)
            elif forgery_type == 'thickness':
                img = vary_thickness(original_img)
            else: # noise
                img = np.clip(original_img + np.random.normal(0, 0.05, original_img.shape), 0, 1)

            img = np.clip(img, 0, 1)
            x_forged.append(img if len(img.shape) == 3 else img[..., np.newaxis])
            y_digit.append(y[original_idx])
            y_forgery.append(1)
        except Exception:
            continue
    
    x_combined = np.concatenate([x, np.array(x_forged)], axis=0)
    y_digit_combined = np.concatenate([y, np.array(y_digit)], axis=0)
    y_forgery_combined = np.concatenate([np.zeros(len(x)), np.array(y_forgery)], axis=0)
    
    indices = np.random.permutation(len(x_combined))
    return x_combined[indices], y_digit_combined[indices], y_forgery_combined[indices]

def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    print("Generating training forgeries...")
    x_train_aug, y_digit_train, y_forgery_train = add_realistic_forgeries(x_train, y_train_cat, num_forgeries=60000)
    
    print("Generating test forgeries...")
    x_test_aug, y_digit_test, y_forgery_test = add_realistic_forgeries(x_test, y_test_cat, num_forgeries=10000)

    model = create_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005), # <-- MODIFIED: Lower learning rate
        loss={'digit_output': 'categorical_crossentropy', 'forgery_output': 'binary_crossentropy'},
        metrics={
            'digit_output': 'accuracy', 
            'forgery_output': ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        },
        loss_weights={'digit_output': 1.0, 'forgery_output': 1.2} # <-- MODIFIED: Adjusted loss weight
    )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint('enhanced_mnist_forgery.keras', save_best_only=True, monitor='val_digit_output_accuracy', mode='max'),
        keras.callbacks.EarlyStopping(
            monitor='val_digit_output_accuracy',
            mode='max',
            patience=10, # <-- MODIFIED: Increased patience
            restore_best_weights=True
        )
    ]
    
    print("\nTraining model...")
    model.fit(
        x_train_aug, {'digit_output': y_digit_train, 'forgery_output': y_forgery_train},
        batch_size=128, epochs=30,
        validation_data=(x_test_aug, {'digit_output': y_digit_test, 'forgery_output': y_forgery_test}),
        callbacks=callbacks, verbose=1
    )
    
    model.save('enhanced_mnist_forgery_final.keras')
    print("Model training complete. Best model saved as 'enhanced_mnist_forgery.keras'")

if __name__ == "__main__":
    main()