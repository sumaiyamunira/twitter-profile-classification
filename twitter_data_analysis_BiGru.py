import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout
from sklearn.preprocessing import LabelEncoder

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

# Encode gender labels
label_encoder = LabelEncoder()
data['gender_encoded'] = label_encoder.fit_transform(data['gender'])

# Ensure that 'unknown' gender is removed from the dataset
df = df[df['gender'] != 'unknown']

# Encode labels for the entire dataset
df['gender_encoded'] = label_encoder.transform(df['gender'])

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

# Tokenize and pad text sequences
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(df['cleaned_description'])  # Fit tokenizer on cleaned text
description_sequences = tokenizer.texts_to_sequences(df['cleaned_description'])
description_padded = pad_sequences(description_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    description_padded, df['gender_encoded'], test_size=0.2, random_state=42)

# Build the BiGRU model
embedding_dim = 100

model = Sequential()
model.add(Embedding(MAX_NUM_WORDS, embedding_dim, input_length=MAX_SEQUENCE_LENGTH))
model.add(Bidirectional(GRU(64, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Multi-class classification

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Predict on all data
description_sentiment_pred = model.predict(description_padded)
description_sentiment_pred_classes = description_sentiment_pred.argmax(axis=-1)

# Add predictions to dataframe
df['description_sentiment_bigru'] = label_encoder.inverse_transform(description_sentiment_pred_classes)

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
        sentiment = row.get('description_sentiment_bigru', 'human') 
        if sentiment == 'brand':
            return 'brand'
        else:
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


# Filter high-confidence true labels
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

# Optionally save the results
df.to_csv('classified_profiles.csv', index=False)


