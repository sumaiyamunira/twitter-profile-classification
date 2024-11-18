import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
url = 'twitter_user_data.csv'
df = pd.read_csv(url, encoding='ISO-8859-1')

df['description'] = df['description'].fillna('no description')

# Identify profiles with "no description"
no_description_profiles = df[df['description'] == 'no description']

# Identify profiles with "unknown" gender
unknown_gender_profiles = df[df['gender'] == 'unknown']

# Save profiles with no description and unknown gender to separate files
no_description_profiles.to_csv('no_description_profiles.csv', index=False)
unknown_gender_profiles.to_csv('unknown_gender_profiles.csv', index=False)

# Remove profiles with "no description" and "unknown" gender from the main dataset
data = df[(df['description'] != 'no description') & (df['gender'] != 'unknown')]

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ''  # Return an empty string if text is not a string
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    tokens = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize tokens
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_description'] = df['description'].apply(preprocess_text)

# Initialize SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Sentiment analysis function
def get_sentiment(text):
    if not isinstance(text, str):
        return 0  # Return neutral sentiment if text is not a string
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

# Apply sentiment analysis
df['description_sentiment'] = df['description'].apply(get_sentiment)

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(df['cleaned_description'])

# Convert TF-IDF results to DataFrame
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out(), index=df.index)

# Aggregate top terms for brand and person profiles
brand_terms = tfidf_df[df['gender'] == 'brand'].mean().sort_values(ascending=False)
person_terms = tfidf_df[df['gender'] != 'brand'].mean().sort_values(ascending=False)

print("Top terms for brand profiles:")
print(brand_terms.head(20))

print("\nTop terms for person profiles:")
print(person_terms.head(20))

# Define dynamic keywords based on top terms
brand_keywords = brand_terms.head(50).index.tolist()
person_keywords = person_terms.head(50).index.tolist()

# Function to detect keywords
def contains_keywords(text, keywords):
    return any(word in text for word in keywords)

# Apply brand and person keyword detection
df['contains_brand_keywords'] = df['cleaned_description'].apply(lambda x: contains_keywords(x, brand_keywords))
df['contains_person_keywords'] = df['cleaned_description'].apply(lambda x: contains_keywords(x, person_keywords))

# Identify non-brand accounts based on default profile image pattern
default_profile_images = [
    "https://abs.twimg.com/sticky/default_profile_images/default_profile_1_normal.png",
    "https://abs.twimg.com/sticky/default_profile_images/default_profile_0_normal.png",
    "https://abs.twimg.com/sticky/default_profile_images/default_profile_2_normal.png",
    "https://abs.twimg.com/sticky/default_profile_images/default_profile_3_normal.png",
    "https://abs.twimg.com/sticky/default_profile_images/default_profile_4_normal.png",
    "https://abs.twimg.com/sticky/default_profile_images/default_profile_5_normal.png",
    "https://abs.twimg.com/sticky/default_profile_images/default_profile_6_normal.png"
]

def is_non_brand_profileimage(profile_image_url):
    return any(default_image in profile_image_url for default_image in default_profile_images)

df['is_non_brand_profileimage'] = df['profileimage'].apply(is_non_brand_profileimage)

# Updated logic based on sentiment, keyword detection, and profile image
def classify_profile(row):
    if row['is_non_brand_profileimage']:
        return 'human'
    elif row['contains_brand_keywords']:
        return 'brand'
    elif row['contains_person_keywords']:
        return 'human'
    else:
        sentiment = row.get('description_sentiment', 0) 
        if -0.5 <= sentiment <= 0.4:
            return 'human'
        elif 0.5 <= sentiment <= 1.0:
            return 'brand'
        else:
            # Default to 'human' if sentiment does not fall into the specified ranges
            return 'human'

# Apply classification logic
df['classified_gender'] = df.apply(classify_profile, axis=1)


# # Evaluate performance on the entire dataset
# true_labels = df['gender']  # True labels for the entire dataset
# predicted_labels = df['classified_gender']  # Predictions based on classification logic

# # Convert true_labels to match the predicted_labels
# true_labels = true_labels.apply(lambda x: 'brand' if x == 'brand' else 'human')

# # Evaluate accuracy and generate metrics
# accuracy = accuracy_score(true_labels, predicted_labels)
# print(f'Accuracy: {accuracy:.2f}')

# conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=['human', 'brand'])
# print('Confusion Matrix:')
# print(conf_matrix)

# class_report = classification_report(true_labels, predicted_labels, labels=['human', 'brand'])
# print('Classification Report:')
# print(class_report)



# # Filter high-confidence true labels
high_confidence_df = df[
    (df['_unit_state'] == 'golden') & (df['gender:confidence'] > 0.9)
]

true_labels = high_confidence_df['gender']  # True labels based on high confidence
predicted_labels = high_confidence_df['classified_gender']  # Predictions based on classification logic


# Convert true_labels to match the predicted_labels
true_labels = true_labels.apply(lambda x: 'brand' if x == 'brand' else 'human')


# Evaluate accuracy and generate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
print(f'Accuracy: {accuracy:.2f}')


conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=['human', 'brand'])
print('Confusion Matrix:')
print(conf_matrix)


class_report = classification_report(true_labels, predicted_labels, labels=['human', 'brand'])
print('Classification Report:')
print(class_report)


