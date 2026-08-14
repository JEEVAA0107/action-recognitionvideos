# Action Recognition in Videos

## Project Title and Description
**Project Title:** Action Recognition in Videos
**Description:** This project implements a deep learning-based computer vision system to accurately classify human activities in video clips. Using a subset of the UCF101 dataset, the model identifies 10 diverse actions (such as Walking, Boxing, Biking, etc.) by analyzing spatiotemporal features. It evaluates performance using metrics like Top-1 Accuracy, Top-5 Accuracy, and Confusion Matrices.

## Dependencies or Prerequisites
The following libraries are required to run this project:
- `torch` and `torchvision` (Deep Learning Framework & 3D CNNs)
- `opencv-python` (Video frame reading and preprocessing)
- `scikit-learn` (Evaluation metrics: Confusion Matrix, Classification Report)
- `matplotlib` and `seaborn` (Visualization of graphs and matrices)
- `numpy`, `pandas`, `tqdm`, `Pillow`

**Hardware Prerequisites:**
- NVIDIA GPU with at least 15GB VRAM (e.g., Google Colab T4 GPU) is highly recommended for training.
- At least 10 GB of free disk space is required for the UCF101 dataset.

## Setup and Usage Instructions

**Step 1: Clone the Repository**
```bash
git clone https://github.com/JEEVAA0107/action-recognitionvideos.git
cd action-recognitionvideos
pip install -r requirements.txt
```

**Step 2: Download the UCF101 Dataset**
Run the automated downloader script to fetch the official UCF101 dataset and split it into Train, Val, and Test folders.
```bash
python src/download_ucf101.py
```

**Step 3: Train the Model**
Train the Pre-trained R(2+1)D-18 model on the dataset. The best model weights will be saved automatically.
```bash
python src/train.py --epochs 10 --batch_size 8 --lr 1e-4
```

**Step 4: Evaluate the Model**
Evaluate the model on the test set to get Top-1 Accuracy, Top-5 Accuracy, Precision, Recall, F1-Score, and the Confusion Matrix.
```bash
python src/evaluate.py
```

**Step 5: Run Inference on a Sample Video**
Test the model on a single video to get an output video with the predicted action overlaid on the screen.
```bash
python src/infer.py --video_path data/test/Archery/v_Archery_g01_c01.avi
# (Note: Since train/test split is random, if this specific file is not found, check the folder and use any other .avi file name)
```

*(Note: For easy execution without local setup, you can run the provided standalone Colab notebook `Action_Recognition.ipynb`)*

## Brief Explanation of Your Solution Approach
Our solution approaches the problem of action recognition by treating videos not as independent images, but as spatiotemporal volumes. 
1. **Data Preprocessing:** We use OpenCV to read videos and uniformly sample exactly 16 frames evenly distributed across the entire video duration. This captures both the beginning, middle, and end of the action.
2. **Data Augmentation:** To prevent overfitting, we apply spatial resizing (112x112), Random Horizontal Flipping, and ImageNet standardization.
3. **Model Architecture:** We utilize a pre-trained **R(2+1)D-18 3D Convolutional Neural Network**. Unlike standard 2D CNNs, R(2+1)D factorizes 3D convolutions into a 2D spatial convolution followed by a 1D temporal convolution. This makes it highly efficient and exceptionally good at understanding motion over time.
4. **Transfer Learning:** By using weights pre-trained on the massive Kinetics-400 dataset, the model rapidly converges and achieves high validation accuracy (98%+) within just a few epochs on our 10-class UCF101 subset.
 
 
