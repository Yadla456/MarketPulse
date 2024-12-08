# MarketPulse
Reddit Sentiment Analysis Project
Overview
This project aims to analyze sentiment in Reddit posts and comments, particularly focusing on stock market-related discussions. Using sentiment analysis techniques, the project extracts insights from textual data and predicts sentiment trends, which could provide valuable insights into market sentiments.
Features
Data Scraping: Collects posts and comments from specified subreddits using the Reddit API.
Sentiment Analysis: Analyzes sentiment polarity of titles and comments using NLP techniques.
Data Preprocessing: Cleans text data by removing noise, punctuation, and stopwords.
Keyword Extraction: Identifies and visualizes frequently occurring words in comments.
Machine Learning: Trains a Random Forest model to classify sentiments into Positive, Neutral, and Negative.
Evaluation: Generates performance metrics, including accuracy, classification report, and confusion matrix.
Workflow
Data Collection:

Connects to the Reddit API using PRAW to scrape posts and comments from a chosen subreddit.
Saves the scraped data to a CSV file.
Data Preprocessing:

Cleans and tokenizes text.
Removes common stopwords.
Prepares the data for sentiment analysis and machine learning.
Sentiment Analysis:

Applies sentiment scoring to titles and comments using pre-trained models like VADER and TextBlob.
Feature Extraction:

Extracts top keywords using TF-IDF vectorization for further insights.
Model Training:

Trains a Random Forest classifier on sentiment-labeled data.
Evaluates model performance using test data.
Visualization:

Displays sentiment distribution, keyword frequency, and correlation matrix for insights.
Results
Accuracy: The machine learning model achieved a high accuracy score for sentiment prediction.
Insights:
Positive sentiment is often associated with optimistic stock market discussions.
Negative sentiment reflects concerns or bearish views.
Neutral sentiment correlates with balanced or speculative discussions.
Future Enhancements
Expand data sources to include platforms like Twitter or Telegram.
Implement deep learning models for more accurate sentiment analysis.
Develop a real-time sentiment tracker for live stock market monitoring.
Tools and Libraries
Languages: Python
Libraries: PRAW, Pandas, NLTK, VADER, TextBlob, Scikit-learn, Seaborn, Matplotlib
Dataset
The dataset is scraped dynamically from Reddit, focusing on subreddits such as r/stocks and r/investing. Processed data is saved in CSV files for analysis.
