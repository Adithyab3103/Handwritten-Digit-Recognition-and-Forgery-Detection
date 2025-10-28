# app.py
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Import the core functionalities
from core import load_model, preprocess_image, make_prediction, generate_heatmap

def plot_superimposed_heatmap(original_img, heatmap, alpha=0.5):
    """Overlays the heatmap on the original image."""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Ensure original image is in RGB format for color blending
    if len(original_img.shape) == 2: # if grayscale
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        original_img_rgb = original_img

    superimposed_img = cv2.addWeighted(original_img_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed_img

def main():
    # Page configuration
    st.set_page_config(
        page_title="Handwriting Analysis AI",
        page_icon="✍️",
        layout="wide"
    )

    # --- UI Styling ---
    st.markdown("""
    <style>
    /* Add your custom CSS here */
    </style>
    """, unsafe_allow_html=True)

    st.title("Handwriting Analysis: Digit & Forgery Detection")
    st.markdown("Upload an image of a handwritten digit to analyze it for authenticity.")

    # --- Sidebar ---
    with st.sidebar:
        st.header("About the AI")
        st.info(
            "This app uses a Convolutional Neural Network (CNN) to perform two tasks: "
            "recognize the digit (0-9) and detect if it shows signs of being a forgery."
        )
        st.header("How It Works")
        st.markdown(
            "- **Digit Recognition**: Classifies the image into one of the 10 digits.\n"
            "- **Forgery Detection**: Identifies subtle inconsistencies learned from thousands of synthetically forged examples.\n"
            "- **Attention Heatmap**: Visualizes where the model 'looked' to make its digit prediction."
        )

    # --- Model Loading ---
    @st.cache_resource
    def cached_load_model():
        try:
            return load_model()
        except IOError as e:
            st.error(f"Fatal Error: {e}. Make sure the model 'enhanced_mnist_forgery.keras' is in the same directory.")
            return None
    
    model = cached_load_model()
    
    if model is None:
        st.stop()

    # --- File Uploader and Main Logic ---
    uploaded_file = st.file_uploader(
        "Choose an image of a single handwritten digit...", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        try:
            # Read image from uploader
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) # Convert to RGB

            with st.spinner('Analyzing image...'):
                # Process image using core function
                img_array, processed_img_display = preprocess_image(original_img)
                
                # Get predictions using core function
                pred_digit, digit_conf, is_forged, forgery_conf = make_prediction(model, img_array)
                
                # Get heatmap using core function
                heatmap = generate_heatmap(model, img_array)
                superimposed_heatmap = plot_superimposed_heatmap(original_img, heatmap)

            # --- Display Results ---
            st.header("Analysis Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(original_img, caption="Original Image", use_container_width=True) # <-- CHANGED HERE
            with col2:
                st.image(processed_img_display, caption=f"Preprocessed (Predicted: {pred_digit})", use_container_width=True) # <-- CHANGED HERE
            with col3:
                st.image(superimposed_heatmap, caption="Attention Heatmap", use_container_width=True) # <-- CHANGED HERE

            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.subheader("Digit Recognition")
                st.metric("Predicted Digit", pred_digit)
                st.metric("Confidence", f"{digit_conf*100:.2f}%")
            with res_col2:
                st.subheader("Forgery Detection")
                if is_forged:
                    st.error(f"Potential Forgery Detected ({forgery_conf*100:.2f}% confidence)")
                    st.warning("The model found characteristics similar to known forgeries, such as unnatural strokes or manipulations.")
                else:
                    st.success(f"Likely Authentic ({ (1-forgery_conf)*100:.2f}% confidence)")
                    st.info("The writing appears natural and consistent, with no obvious signs of forgery.")

        except (ValueError, TypeError) as e:
            st.error(f"Error processing the uploaded image: {e}. Please try another image.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.info("Upload an image to begin the analysis.")

if __name__ == "__main__":
    main()