# EM Lab03 - TensorFlow and PyTorch

This repository contains the implementation and results for the EM Lab03 assignment, focusing on building, training, and converting neural networks using TensorFlow and PyTorch with the MNIST dataset.

## Installation Instructions

1. **Clone the Repository**
   Clone this repository to your local machine using:
   ```bash
   git clone https://github.com/yourusername/EM-Lab03.git
   cd EM-Lab03
   
2. Set Up a Virtual Environment (Recommended) Create a virtual environment to manage dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies Install the required Python packages using the provided requirements.txt file:
   ```bash
   pip install -r requirements.txt
   
4. Verify Installation Ensure all packages are installed correctly by running:
   ```bash
   python -c "import tensorflow, torch, numpy, onnx; print('All libraries installed successfully!')"

5. Download the MNIST Dataset The MNIST dataset will be automatically downloaded by the scripts when you run the code, as it is fetched from
   [MNIST Dataset](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)

6. Run the Scripts Execute the TensorFlow and PyTorch scripts to start the lab:
   TensorFlow: `python tensorflow_model.py`  
   PyTorch: `python pytorch_model.py` *(Replace `tensorflow_model.py` and `pytorch_model.py` with your actual script filenames.)*

## Notes

- Ensure you have Python 3.7+ installed.
- The code was tested on a 13th Gen Intel(R) Core(TM) i9-13900H (2.60 GHz) CPU. Performance may vary on different hardware.
- For GPU support, install the GPU versions of TensorFlow and PyTorch if applicable (e.g., `pip install tensorflow-gpu` and follow PyTorch's GPU installation guide).
- The repository includes exported model files (`model.tflite` and `model.onnx`) and training logs as per the lab submission requirements.

