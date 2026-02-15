# Image-Based Cattle Breed Recognition System

A deep learning project that uses convolutional neural networks (CNNs) to automatically recognize and classify cattle breeds from images.

## Features

- **Automatic Breed Classification**: Recognizes 6 different cattle breeds
- **Deep Learning Model**: Built with TensorFlow/Keras for high accuracy
- **Web Interface**: Streamlit-based user-friendly application
- **Pre-trained Models**: Includes optimized `.h5` and `.keras` model formats
- **RESTful API**: Python-based backend for integration

## Supported Cattle Breeds

1. Gir
2. Holstein Friesian
3. Jaffrabadi
4. Jersey
5. Murrah
6. Sahiwal

## Project Structure

```
Breed_Project/
├── app.py                  # Streamlit web application
├── train.py               # Model training script
├── evaluate.py            # Model evaluation script
├── utils.py               # Utility functions
├── requirements.txt       # Project dependencies
├── breed_info.json        # Breed information database
├── class_indices.json     # Class mapping indices
├── class_indices.txt      # Text version of class indices
├── model/
│   ├── best_model.h5      # Best model (HDF5 format)
│   └── best_model.keras   # Best model (Keras format)
└── dataset/
    ├── train/             # Training data
    ├── valid/             # Validation data
    └── test/              # Test data
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/DhiyaShr1385/Image-Based-Breed-Recognition-System.git
cd Breed_Project
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model from scratch:
```bash
python train.py
```

### Evaluating the Model

To evaluate the model performance:
```bash
python evaluate.py
```

### Running the Web Application

To launch the Streamlit web interface:
```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`

## Dependencies

- tensorflow
- keras
- pillow
- numpy
- streamlit
- opencv-python

See `requirements.txt` for complete list and versions.

## Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Input Size**: Standardized image dimensions
- **Output**: Probability distribution across 6 cattle breeds
- **Format**: Available in both `.h5` and `.keras` formats

## Dataset

The dataset is organized into three splits:
- **Training**: Used for model training
- **Validation**: Used for hyperparameter tuning
- **Testing**: Used for final model evaluation

Each split contains subdirectories for each cattle breed.

## Results

The trained model achieves high accuracy in classifying cattle breeds. Detailed evaluation metrics can be found by running the evaluation script.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**DhiyaShr1385**

## Acknowledgments

- TensorFlow and Keras documentation
- OpenCV community
- Streamlit framework

---

For more information or issues, please visit the [GitHub repository](https://github.com/DhiyaShr1385/Image-Based-Breed-Recognition-System)
