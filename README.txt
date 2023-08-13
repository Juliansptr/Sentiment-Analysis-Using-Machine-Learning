Title: "Application of Machine Learning Classification Algorithm in Sentiment Analysis of Movie Reviews"

Concept:
This project applies Natural Language Processing using a machine learning classification model trained on a dataset of 25,000 movie reviews from the IMDB website to analyze sentiment. The application works by taking user input in the form of a review and using the trained model to classify the review as positive or negative. The result is then displayed to the user, indicating the predicted sentiment. The algorithm used as the application model is the SGDClassifier, with a prediction accuracy of 89%, implemented on the web using the Flask framework.

Dataset:
The labeled dataset consists of 50,000 IMDB movie reviews, specifically chosen for sentiment analysis. Sentiments are binary, where IMDB ratings <5 are assigned a sentiment score of 0, and ratings >=7 are assigned a sentiment score of 1. No individual movie has more than 30 reviews. The labeled training set of 25,000 reviews does not overlap with the labeled test set of 25,000 reviews. Additionally, there are 50,000 additional IMDB reviews without rating labels.

Data Preprocessing:
Data preprocessing is a crucial step for natural language processing (NLP) tasks. It transforms text into a more digestible format for machine learning algorithms to perform effectively. In this case, the following preprocessing steps were performed:

- Removed all URLs from data.
- Removed all tags from data.
- Decontracted words.
- Removed special characters from data.
- Removed stop words.

Feature Extraction: TF-IDF Vectorizer

Modeling:
The IMDB dataset was trained using various machine learning classification algorithms:

- SVM: 89%
- Naive Bayes: 76%
- SGD Classifier: 89%
- Ridge Classifier: 88%
- Decision Tree: 71%
- Logistic Regression: 88%
- Random Forest: 85%
- KNN: 81%

Based on the prediction results using multiple classification algorithms, two models suitable for web application were identified: SVM Algorithm and SGDClassifier Algorithm.

Technical Aspect:
The project is divided into two parts:
1. Training the IMDB dataset using machine learning models.
2. Implementing the model.

For training purposes, the scikit-learn library was used.
For web application implementation, Flask was employed.