# NLP and Computer Vision Tasks

This repository contains implementations of two machine learning tasks: Named Entity Recognition (NER) and Image Classification.

## Named Entity Recognition (NER)

### Overview

Implementation of NER model to identify animals in text using BERT-based architecture.

### Components

- `dataset_creation.ipynb`: Jupyter notebook for generating training data using OpenAI API
- `train.py`: Training script for NER model using BERT
- Data format: BIO tagging scheme (Begin, Inside, Outside)

### Results

While the model showed promising metrics during training (F1 > 0.98), it failed to generalize well during real-world evaluation.

### Dataset

- Generated using [Nvidia&#39;s](https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct) API
- Contains sentences with animal mentions
- Split into train/val/test sets

## Image Classification

### Overview

Baseline implementation of animal image classifier using ResNet50.

### Features

- Uses pretrained ResNet50 architecture
- Dataset: Animals-10 from Kaggle
- Basic data augmentation pipeline
- Transfer learning approach (only final layer trained)
