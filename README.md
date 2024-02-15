# Intrusion Detection System with Machine Learning

## Overview

This repository contains a comprehensive Intrusion Detection System (IDS) implemented using various machine learning algorithms. The system is designed to classify network traffic into normal and attack categories based on features extracted from network packets.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Results](#results)
- [License](#license)

## Introduction

The goal of this project is to build and evaluate different machine learning models for intrusion detection in network traffic. The models include K-Nearest Neighbors, Random Forest Classifier, Autoencoder, One-Class SVM, and Isolation Forest. The dataset used for training and testing is the KDD Cup 1999 dataset.

## Features

- Data preprocessing: Scaling, imputing missing values, and one-hot encoding of categorical variables.
- Model evaluation: Utilizes various machine learning models for classification and evaluates them based on accuracy, precision, recall, and confusion matrices.
- Visualization: Confusion matrices are visualized using Plotly for interactive and insightful representations.
- Autoencoder: Implements an autoencoder for anomaly detection using neural networks.

## Dependencies

Ensure that you have the following dependencies installed:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- XGBoost
- Plotly
- Jupyter Notebook

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/intrusion-detection-system.git
```

2. Navigate to the project directory:

```bash
cd intrusion-detection-system
```

3. Open and run the Jupyter notebooks:

```bash
jupyter notebook
```

Follow the instructions in the notebooks for data preprocessing, model training, and evaluation.

## Notebooks

- `1_data_preprocessing.ipynb`: Data preprocessing steps including scaling, imputation, and one-hot encoding.
- `2_model_training_evaluation.ipynb`: Model training and evaluation using various machine learning algorithms.
- `3_autoencoder.ipynb`: Implementation and evaluation of an autoencoder for anomaly detection.

## Results

The evaluation results and visualizations can be found in the notebooks and the `results` directory.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to explore and contribute to the project! If you encounter any issues or have suggestions, please open an [issue](https://github.com/your-username/intrusion-detection-system/issues).