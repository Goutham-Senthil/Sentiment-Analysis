# Sentiment Analysis Using SVM, Naive Bayes, and Bi-LSTM

This project demonstrates how to perform sentiment analysis using three different models: Support Vector Machine (SVM), Naive Bayes, and Bi-Directional Long Short-Term Memory (Bi-LSTM). The dataset used for this project is the [Twitter Sentiment Analysis Dataset](https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis).

## Dataset

The dataset is sourced from HuggingFace and contains tweets labeled with sentiments. You can download it [here](https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis).

## Prerequisites

Make sure you have the following packages installed:

- Python 3.7+
- scikit-learn
- pandas
- numpy
- nltk
- tensorflow
- keras

You can install these packages using pip:

```sh
pip install scikit-learn pandas numpy nltk tensorflow keras
```

## Support Vector Machine (SVM)

Support Vector Machine (SVM) is a supervised machine learning algorithm that can be used for classification or regression challenges. It works by finding the hyperplane that best divides a dataset into classes.

## Naive Bayes

Naive Bayes is a probabilistic machine learning model that's used for classification tasks. It is based on Bayes' Theorem and assumes independence between predictors. Despite its simplicity, it can be very effective for large datasets.

## Bi-Directional LSTM (Bi-LSTM)

Bi-Directional Long Short-Term Memory (Bi-LSTM) is an advanced type of Recurrent Neural Network (RNN) that can capture long-term dependencies in sequence data. Unlike standard LSTMs, Bi-LSTMs process data in both forward and backward directions, improving the model's context understanding.

## Conclusion 

We explored three different approaches to sentiment analysis: SVM, Naive Bayes, and Bi-LSTM. Each model has its own advantages and is suitable for different types of data and use cases. SVM is powerful for linear separable data, Naive Bayes is highly efficient for large datasets, and Bi-LSTM excels at capturing contextual information in sequences.
