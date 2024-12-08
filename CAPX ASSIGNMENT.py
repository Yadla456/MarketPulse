#!/usr/bin/env python
# coding: utf-8

# In[123]:


pip install praw


# In[124]:


import praw

# Replace with your credentials
client_id = "gzSq_bQyhG0jHYok66d3EA"
client_secret = "b7CDAP_vFb-69Q5qOWLJA6gfz2IfQw"
user_agent = "Complete_Attitude555"  # Descriptive name for your app

# Initialize Reddit instance
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

print("Connected to Reddit!")


# In[125]:


# Define the subreddit and sorting (e.g., top posts, hot posts)
subreddit = reddit.subreddit("stocks")  # Replace with your target subreddit

# Fetch posts
posts = []
for post in subreddit.hot(limit=100):  # Fetch the top 100 posts
    posts.append({
        "title": post.title,
        "score": post.score,
        "id": post.id,
        "url": post.url,
        "num_comments": post.num_comments,
        "created": post.created,
        "body": post.selftext
    })

print(f"Scraped {len(posts)} posts from r/stocks")


# In[126]:


import pandas as pd

# Convert to DataFrame
df = pd.DataFrame(posts)

# Save to CSV
df.to_csv("reddit_posts.csv", index=False)
print("Data saved to reddit_posts.csv")


# In[127]:


get_ipython().system('pip install vaderSentiment')


# In[128]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Apply sentiment analysis
df['sentiment'] = df['title'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Save with sentiment scores
df.to_csv("reddit_posts_with_sentiment.csv", index=False)


# In[129]:


post_id = "8080"  # Replace with a specific Reddit post ID
submission = reddit.submission(id=post_id)

comments = []
submission.comments.replace_more(limit=0)  # Load all comments
for comment in submission.comments.list():
    comments.append({
        "comment_id": comment.id,
        "comment_body": comment.body,
        "comment_score": comment.score,
        "created": comment.created
    })

# Save to CSV
comments_df = pd.DataFrame(comments)
comments_df.to_csv("reddit_comments.csv", index=False)


# In[130]:


for post in subreddit.search("stock market", limit=100):
    print(post.title)


# In[131]:


try:
    submission = reddit.submission(id=post_id)
    print(f"Title: {submission.title}")
except Exception as e:
    print(f"Error: {e}")


# In[132]:


post_id = "8080"  # Replace with a specific Reddit post ID

try:
    submission = reddit.submission(id=post_id)  # Fetch the submission
    print(f"Post Title: {submission.title}")  # Test if the post is accessible
    
    comments = []
    submission.comments.replace_more(limit=0)  # Load all comments
    for comment in submission.comments.list():
        comments.append({
            "comment_id": comment.id,
            "comment_body": comment.body,
            "comment_score": comment.score,
            "created": comment.created
        })

    # Save to CSV
    comments_df = pd.DataFrame(comments)
    comments_df.to_csv("reddit_comments.csv", index=False)
    print("Comments saved successfully!")

except Exception as e:
    print(f"Error: {e}")


# In[133]:


comments_df.to_csv("reddit_comments.csv", index=False)


# In[134]:


pip install pandas nltk scikit-learn textblob


# In[135]:


import nltk
nltk.download('punkt')  # Download the punkt tokenizer
nltk.download('stopwords')  # Ensure stopwords are downloaded too


# In[136]:


import nltk
nltk.download('all')  # Downloads all NLTK resources


# In[137]:


import nltk
import os

nltk_path = os.path.expanduser('~/nltk_data')  # Replace `~/nltk_data` with your actual NLTK data directory
if nltk_path not in nltk.data.path:
    nltk.data.path.append(nltk_path)
nltk.download('punkt')


# In[138]:


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer

def simple_tokenizer(text):
    words = text.lower().split()
    return [word for word in words if word not in ENGLISH_STOP_WORDS]


# In[139]:


import pandas as pd
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import praw  # Reddit API Wrapper

# Step 1: Reddit Authentication
reddit = praw.Reddit(
    client_id = "gzSq_bQyhG0jHYok66d3EA",
    client_secret = "b7CDAP_vFb-69Q5qOWLJA6gfz2IfQw",
    user_agent = "Complete_Attitude555"
)

# Step 2: Scrape Data from a Subreddit
def scrape_reddit(subreddit_name, post_limit=10):
    subreddit = reddit.subreddit(subreddit_name)
    posts = {"Title": [], "Comments": []}

    for post in subreddit.hot(limit=post_limit):
        posts["Title"].append(post.title)
        post.comments.replace_more(limit=0)
        comments = " ".join([comment.body for comment in post.comments.list()])
        posts["Comments"].append(comments)

    return pd.DataFrame(posts)

# Step 3: Preprocessing Functions
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def get_sentiment(text):
    sentiment = TextBlob(text).sentiment
    return sentiment.polarity

# Step 4: Process and Save Data
def process_and_save_data(subreddit_name, post_limit=10, output_file="reddit_data.csv"):
    # Scrape data
    df = scrape_reddit(subreddit_name, post_limit)

    # Preprocess titles and comments
    df['Cleaned_Title'] = df['Title'].apply(preprocess_text)
    df['Cleaned_Comments'] = df['Comments'].apply(preprocess_text)

    # Add sentiment scores
    df['Title_Sentiment'] = df['Cleaned_Title'].apply(get_sentiment)
    df['Comments_Sentiment'] = df['Cleaned_Comments'].apply(get_sentiment)

    # Extract keyword frequencies
    vectorizer = CountVectorizer(max_features=5)
    X = vectorizer.fit_transform(df['Cleaned_Comments'])
    keywords = vectorizer.get_feature_names_out()

    # Add keyword features
    keyword_df = pd.DataFrame(X.toarray(), columns=keywords)
    df = pd.concat([df, keyword_df], axis=1)

    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Data saved to {output_file}")

# Step 5: Run the Script
if __name__ == "__main__":
    process_and_save_data(subreddit_name="stocks", post_limit=5, output_file="reddit_data.csv")


# In[140]:


import pandas as pd
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Read the CSV File
input_file = "reddit_data.csv"  # Replace with your CSV file name
df = pd.read_csv(input_file)

# Step 2: Download NLTK Resources
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Step 3: Preprocessing Function
def preprocess_text(text):
    if pd.isna(text):  # Handle missing values
        return ""
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])  # Remove punctuation/numbers
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Apply Preprocessing to Titles and Comments
df['Cleaned_Title'] = df['Title'].apply(preprocess_text)
df['Cleaned_Comments'] = df['Comments'].apply(preprocess_text)

# Step 4: Sentiment Analysis Function
def get_sentiment(text):
    if not text:
        return 0  # Neutral for empty text
    sentiment = TextBlob(text).sentiment
    return sentiment.polarity  # Polarity score between -1 (negative) and 1 (positive)

# Add Sentiment Analysis Columns
df['Title_Sentiment'] = df['Cleaned_Title'].apply(get_sentiment)
df['Comments_Sentiment'] = df['Cleaned_Comments'].apply(get_sentiment)

# Step 5: Keyword Frequency Extraction
vectorizer = CountVectorizer(max_features=5)  # Extract top 5 keywords
X = vectorizer.fit_transform(df['Cleaned_Comments'])
keywords = vectorizer.get_feature_names_out()

# Add Keyword Frequency Features to DataFrame
keyword_df = pd.DataFrame(X.toarray(), columns=keywords)
df = pd.concat([df, keyword_df], axis=1)

# Step 6: Save Processed Data to a New CSV File
output_file = "processed_reddit_data.csv"  # Specify your output file name
df.to_csv(output_file, index=False, encoding='utf-8')
print(f"Processed data saved to {output_file}")


# In[141]:


from IPython.display import HTML

# File path
file_path = "processed_reddit_data.csv"

# Create a download link
def create_download_link(filename):
    return HTML(f'<a href="{filename}" download>Click here to download {filename}</a>')

# Display the download link
create_download_link(file_path)


# In[142]:


# Load the preprocessed CSV file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file_path = "processed_reddit_data.csv"  # Replace with your file path
df = pd.read_csv(file_path)


# In[143]:


# Basic summary statistics
print(df.describe())  # Numerical features
print(df.info())  # Overview of the dataset
print(df.head())  # First 5 rows of the data


# In[144]:


# Plot the distribution of title and comment sentiment
plt.figure(figsize=(10, 5))
sns.histplot(df['Title_Sentiment'], kde=True, color='blue', label='Title Sentiment', bins=20)
sns.histplot(df['Comments_Sentiment'], kde=True, color='green', label='Comments Sentiment', bins=20)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[145]:


# Extract and plot top keywords
keywords = [col for col in df.columns if col not in ['Title', 'Comments', 'Cleaned_Title', 
                                                     'Cleaned_Comments', 'Title_Sentiment', 'Comments_Sentiment']]

keyword_sums = df[keywords].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=keyword_sums.index, y=keyword_sums.values, palette="viridis")
plt.title("Top Keyword Frequencies")
plt.xlabel("Keywords")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()


# In[146]:


# Scatter plot of sentiment scores vs. keyword frequencies
for keyword in keywords:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[keyword], y=df['Comments_Sentiment'], label=f'{keyword} vs. Sentiment')
    plt.title(f'Keyword: {keyword} vs. Comment Sentiment')
    plt.xlabel('Keyword Frequency')
    plt.ylabel('Comment Sentiment')
    plt.legend()
    plt.show()


# In[147]:


# Correlation matrix
correlation_matrix = df[['Title_Sentiment', 'Comments_Sentiment'] + keywords].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[148]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
df['Text'] = df['Cleaned_Title'] + " " + df['Cleaned_Comments']


# In[149]:


# Use Text (Combined Title and Comments) as features and Comments_Sentiment as labels
X = df['Text']
y = df['Comments_Sentiment'].apply(lambda x: 1 if x > 0 else (0 if x == 0 else -1))  # Convert to classes: -1, 0, 1

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[150]:


# Convert text data to TF-IDF vectors
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')  # Limit features to 5000 for simplicity
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[151]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train_tfidf, y_train)


# In[152]:


# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Check for unique classes in test labels
print("Unique classes in y_test:", y_test.unique())

# Generate a classification report with dynamic labels
labels = y_test.unique()  # Fetch unique labels
print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=labels, target_names=["Negative", "Neutral", "Positive"][:len(labels)]))

# Plot a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Negative", "Neutral", "Positive"][:len(labels)],
            yticklabels=["Negative", "Neutral", "Positive"][:len(labels)])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[153]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Check for unique classes in test labels
print("Unique classes in y_test:", y_test.unique())

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Generate a classification report with dynamic labels
labels = y_test.unique()  # Fetch unique labels
print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=labels, target_names=["Negative", "Neutral", "Positive"][:len(labels)]))

# Plot a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Negative", "Neutral", "Positive"][:len(labels)],
            yticklabels=["Negative", "Neutral", "Positive"][:len(labels)])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[ ]:


import numpy as np

# Ensure you have your trained model and TF-IDF vectorizer loaded or defined
# model: Trained machine learning model
# tfidf_vectorizer: Fitted TF-IDF vectorizer

# Example new text data to predict
new_texts = [
    "Tesla stock is rising rapidly!",
    "There is uncertainty about Google shares.",
    "Apple's new product launch could boost their stock prices.",
    "Economic slowdown might affect the stock market negatively."
]

# Preprocess and vectorize the new text using the trained TF-IDF vectorizer
new_texts_tfidf = tfidf_vectorizer.transform(new_texts)

# Predict the sentiment for the new text
predictions = model.predict(new_texts_tfidf)

# Map predictions to labels (e.g., Negative, Neutral, Positive)
def map_prediction_to_label(pred):
    if pred == 1:
        return "Positive"
    elif pred == 0:
        return "Neutral"
    else:
        return "Negative"

# Display predictions
for text, pred in zip(new_texts, predictions):
    label = map_prediction_to_label(pred)
    print(f"Text: {text}\nPredicted Sentiment: {label}\n")

