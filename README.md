# Handwritten Digit Recognition and Forgery Detection

An advanced deep learning system that not only recognizes handwritten digits but also detects potential forgeries with high accuracy. This project uses a custom Convolutional Neural Network (CNN) trained on the MNIST dataset with synthetic forgeries to identify both the digit and its authenticity.

## Features

- **Digit Recognition**: Accurately identifies handwritten digits (0-9)
- **Forgery Detection**: Detects potential forgeries with confidence scores
- **Visual Explanations**: Generates attention heatmaps to show which parts of the image influenced the prediction
- **Web Interface**: User-friendly Streamlit app for easy interaction
- **Advanced Preprocessing**: Handles various image formats and conditions

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Adithyab3103/Handwritten-Digit-Recognition-and-Forgery-Detection.git
   cd Handwritten-Digit-Recognition-and-Forgery-Detection
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Application

Run the Streamlit web app:

```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`

### Training the Model

To train a new model or retrain the existing one:

```bash
python train.py
```

This will train a new model and save it as `enhanced_mnist_forgery.keras`.

## Model Architecture

The system uses a multi-output CNN with the following structure:

1. **Feature Extraction**:
   - Two convolutional layers with max pooling and dropout
   - Flattened features
   - Dense layers for feature processing

2. **Dual Outputs**:
   - **Digit Classification**: 10 classes (0-9) with softmax activation
   - **Forgery Detection**: Binary classification (real/forged) with sigmoid activation

## 🛠️ Core Components

- **`app.py`**: Streamlit web interface for the application
- **`core.py`**: Core functionality including model loading and prediction
- **`train.py`**: Script for training the model with data augmentation
- **`requirements.txt`**: Project dependencies

## Acknowledgments

- MNIST dataset
- TensorFlow and Keras
- Streamlit for the web interface