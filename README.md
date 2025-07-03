# EM Lab03 - TensorFlow and PyTorch

This repository contains the implementation and results for the EM Lab03 assignment, focusing on building, training, and converting neural networks using TensorFlow and PyTorch with the MNIST dataset.

## Installation Instructions for macOS (MacBook with M4 Chip)

1. **Clone the Repository**  
   Clone this repository to your local machine using:
   ```bash
   git clone https://github.com/yourusername/File_name.git
   cd File_name
   ```

2. **Set Up a Virtual Environment (Recommended)**  
   Create a virtual environment to manage dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**  
   Install the required Python packages using the provided `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

   **Note for M4 Chip**: The M4 chip uses ARM architecture, so ensure compatibility with TensorFlow and PyTorch. Install the following for M4 support:
   ```bash
   pip install tensorflow-macos
   pip install torch torchvision torchaudio
   ```

4. **Verify Installation**  
   Ensure all packages are installed correctly by running:
   ```bash
   python3 -c "import tensorflow, torch, numpy, onnx; print('All libraries installed successfully!')"
   ```

5. **Download the MNIST Dataset**  
   The MNIST dataset will be automatically downloaded by the scripts when you run the code, as it is fetched from:
   [MNIST Dataset](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)

6. **Run the Scripts**  
   Execute the TensorFlow and PyTorch scripts to start the lab:
   - TensorFlow: `python3 tensorflow_model.py`
   - PyTorch: `python3 pytorch_model.py`

## Notes

- Ensure you have Python 3.8+ installed (Python 3.8 or later is recommended for M4 chip compatibility).
- The code was tested on a MacBook with an M4 chip running macOS. Performance may vary on different macOS versions or hardware.
- **GPU/TPU Support**: The M4 chip uses Apple's Metal Performance Shaders (MPS) for PyTorch acceleration. To enable MPS, ensure you have PyTorch 1.12 or later installed. TensorFlow on M4 uses `tensorflow-macos`, which leverages Metal for acceleration. No additional GPU packages (e.g., `tensorflow-gpu`) are needed.
- To enable MPS in PyTorch, the scripts automatically detect the M4 chip if available. If you encounter issues, verify MPS support by running:
   ```bash
   python3 -c "import torch; print(torch.backends.mps.is_available())"
   ```
   This should return `True` if MPS is supported.
- The repository includes exported model files (`model.tflite` and `model.onnx`) and training logs as per the lab submission requirements.
- If you encounter dependency issues, consider installing packages individually with `pip install tensorflow-macos torch numpy onnx`.
