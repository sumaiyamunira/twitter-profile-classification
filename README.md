# Twitter Data Analysis - Gender Classification and Profile Analysis

## Overview
This project performs a comprehensive analysis of Twitter user data, primarily focusing on identifying gender classifications (human vs. brand profiles) based on various features from user profiles. It utilizes natural language processing (NLP) techniques, sentiment analysis, text preprocessing and machine learning models to classify Twitter profiles. This is achieved by analyzing the user's description, profile image and other metadata to predict whether a profile represents a human or a brand.

The dataset used for this analysis is **twitter_user_data.csv**, which contains **20,050 rows** of data, including user information such as profile descriptions, tweet counts and follower counts.

## Key Features

### 1. **Data Preprocessing**
- Missing values in the `description` column are filled with a placeholder value ("no description").
- Profiles with "no description" or "unknown" gender are identified and stored in separate CSV files for further investigation.

### 2. **Text Preprocessing**
- User descriptions are cleaned by:
  - Converting text to lowercase.
  - Removing URLs and non-alphabetic characters.
  - Tokenizing the text.
  - Removing stopwords using the NLTK stopword list.
  - Lemmatizing words to reduce them to their base form.

### 3. **Sentiment Analysis**
- Sentiment analysis is performed on user descriptions using both **BiGru** and **VADER SentimentIntensityAnalyzer**.
- The sentiment score, ranging from -1 (negative) to 1 (positive), is calculated for each user profile description.

### 4. **Keyword Detection**
- Key terms for **brand** and **human** profiles are dynamically identified using TF-IDF vectorization, which highlights the most significant terms based on frequency and relevance.
- Profiles are checked for the presence of these key terms in their descriptions. If keywords associated with brands are found, the profile is flagged as a **brand**. If human-related terms are found, the profile is flagged as a **human**.

### 5. **Profile Image Analysis**
- Profiles that use default profile images are likely non-human accounts (e.g., bots or automated accounts). This heuristic is incorporated into the classification process.
- A set of URLs representing default profile images is used to identify such profiles.

### 6. **Profile Classification**
- The classification of a profile as "human" or "brand" is based on:
  - The presence of **brand-related keywords** in the description.
  - The presence of **human-related keywords** in the description.
  - The **sentiment score** of the description (neutral or positive sentiment may indicate a human profile).
  - **Profile image analysis** (default images are more likely to be bots or non-human).

### 7. **Performance Evaluation**
- The model's performance is evaluated using **accuracy**, **confusion matrix** and **classification report**.
- Accuracy is calculated based on a subset of high-confidence profiles (those with a `gender:confidence` score above 0.9).
- Metrics are generated to assess the classification quality and identify areas of improvement.

## Requirements
To run this project, the following Python libraries are required:
- **pandas**: For data manipulation and analysis.
- **re**: For regular expression-based text processing.
- **nltk**: For natural language processing tasks (tokenization, stopwords, lemmatization).
- **scikit-learn**: For machine learning models and vectorization (TF-IDF).
- **vaderSentiment**: For sentiment analysis.

Install the required libraries with:

```bash

pip install -r requirements.txt

```



