# Customer Segmenter

This project classifies customers into different segments based on their shopping behavior using a K-Nearest Neighbors (KNN) algorithm.

## 📊 Description

The program reads customer data from a CSV file and segments them by training a KNN classifier. It evaluates the classification performance using a confusion matrix and a detailed classification report.

## 🧪 Features

- Reads customer data from `veriler.csv`
- Splits the data into training and test sets (67% train, 33% test)
- Uses a KNN classifier with `k=1`
- Outputs:
  - Classification report (precision, recall, f1-score)
  - Confusion matrix with visualization

## 🚀 Getting Started

### Prerequisites

Make sure you have the following Python libraries installed:

```bash
pip install pandas scikit-learn matplotlib
