# Ironhack Deep Learning Project

## Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Project Structure](#project-structure)
- [How to Run the Project](#how-to-run-the-project)
- [Model Architectures](#model-architectures)
- [Results and Evaluation](#results-and-evaluation)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Project Overview
This project is part of the Ironhack Data Science Bootcamp. The objective is to classify images from the CIFAR-10 dataset using two approaches:
1. A custom Convolutional Neural Network (CNN) architecture.
2. Transfer learning using a pre-trained `EfficientNetB0` model.

**Main Goals**:
- Understand and preprocess the CIFAR-10 dataset.
- Develop a custom CNN and experiment with different architectures.
- Apply transfer learning to leverage a pre-trained model and fine-tune it for our task.
- Evaluate and compare the performance of both approaches.

---

## Data Description
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. The classes include:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

**Data Files in `data/` Folder**:
- `batches.meta`: Metadata including label names.
- `data_batch_1` to `data_batch_5`: Training data batches, each containing 10,000 images.
- `test_batch`: Testing data batch with 10,000 images.
- `readme.html`: Official CIFAR-10 dataset documentation.

**Note**: The images are stored in binary format, and you will need to use the provided data loading scripts to process them into usable formats.

---

## Project Structure
/project-computer-vision
│
├── data/                      # Folder for your datasets
│   ├── batches.meta           # Metadata for the CIFAR-10 dataset
│   ├── data_batch_1           # Training data batch 1
│   ├── data_batch_2           # Training data batch 2
│   ├── data_batch_3           # Training data batch 3
│   ├── data_batch_4           # Training data batch 4
│   ├── data_batch_5           # Training data batch 5
│   ├── readme.html            # Dataset readme file from CIFAR-10
│   └── test_batch             # Testing data batch
│
├── models/                    # Folder for saved models or model checkpoints
│
├── notebooks/                 # Jupyter notebooks for experiments and EDA
│   ├── data_exploration.ipynb # Data exploration and analysis
│   ├── custom_cnn.ipynb       # Custom CNN development and testing
│   ├── transfer_learning.ipynb # Transfer learning experiments
│
├── src/                       # Source code for the project
│   ├── __init__.py            # Initialization file for the package
│   ├── data_utils.py          # Data preprocessing and loading scripts
│   ├── model_utils.py         # Model building, training, and evaluation scripts
│   ├── train.py               # Script to train the model
│   └── predict.py             # Script for making predictions
│
├── README.md                  # Project overview and instructions
├── requirements.txt           # List of dependencies
└── .gitignore                 # Git ignore file

---

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/ironhack-labs/project-1-deep-learning-image-classification-with-cnn.git
   cd my-ironhack-project
2. Install the required dependencies:
    pip install -r requirements.txt
3. Run the Jupyter notebooks in the notebooks/ folder to see the data exploration, custom CNN, and transfer learning experiments.
4. To train the model from the command line, execute:
    python src/train.py
5. To make predictions, run:
    python src/predict.py --image_path path/to/your/image.jpg

## Model Architectures
## Custom CNN
**Layers**: 4 convolutional layers with ReLU activation, max pooling, and dropout for regularization.
**Final Layer**: Dense layer with softmax activation for classification.

## Transfer Learning
**Base Model**: EfficientNetB0 pre-trained on ImageNet.
**Fine-Tuning**: Unfreezing the last 20 layers for better feature learning on CIFAR-10.

## Results and Evaluation
**Custom CNN Accuracy**: X% on the test set.
**Transfer Learning Accuracy**: Y% on the test set.

## Technologies Used
- Python
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## Future Improvements
Experiment with different pre-trained models (e.g., ResNet50, MobileNet).
Implement more data augmentation techniques.
Optimize hyperparameters using tools like Optuna or Hyperopt.

## Author
Tiago Ferreira da Silva: [GitHub tiagovhp](https://github.com/tiagovhp)

