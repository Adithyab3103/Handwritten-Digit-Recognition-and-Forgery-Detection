# core.py
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

def load_model(model_path='enhanced_mnist_forgery.keras'):
    """Loads the trained Keras model."""
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        # Re-raise the exception to be handled by the caller
        raise IOError(f"Error loading model from {model_path}: {e}")

def preprocess_image(image_array, target_size=(28, 28)):
    """Preprocesses a numpy image array for prediction."""
    try:
        # Ensure the image is a NumPy array
        if not isinstance(image_array, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")

        # Convert to grayscale if it has 3 or 4 channels
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:  # RGBA to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Invert colors (MNIST model expects white digits on a black background)
        inverted_image = cv2.bitwise_not(image_array)
        
        # Resize and normalize
        resized_image = cv2.resize(inverted_image, target_size, interpolation=cv2.INTER_AREA)
        normalized_image = resized_image.astype('float32') / 255.0
        
        # Reshape for the model (add batch and channel dimensions)
        model_input = normalized_image.reshape(1, *target_size, 1)
        
        return model_input, normalized_image
        
    except Exception as e:
        # Re-raise with more context
        raise ValueError(f"Error processing image: {e}")

def make_prediction(model, image_array):
    """
    Makes a prediction on a preprocessed image array.
    Returns digit prediction, confidence, forgery status, and forgery confidence.
    """
    pred_digit_probs, pred_forgery_probs = model.predict(image_array, verbose=0)
    
    pred_digit = np.argmax(pred_digit_probs[0])
    digit_confidence = np.max(pred_digit_probs[0])
    
    forgery_confidence = float(pred_forgery_probs[0][0])
    is_forged = forgery_confidence > 0.5
    
    return pred_digit, digit_confidence, is_forged, forgery_confidence

def generate_heatmap(model, img_array, last_conv_layer_name='conv2'):
    """Generates a Grad-CAM heatmap."""
    # Create a model that maps the input to the activations of the last conv layer
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.outputs[0]]
        )
    except ValueError:
        # Fallback for multi-output model structure
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output[0]]
        )

    # Compute gradient of the top predicted class for the digit output
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the convolutional output channels by the gradients
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()