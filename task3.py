
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Six support tickets
tickets = [
    "I was charged twice for my subscription",
    "The app crashes when I open settings",
    "Please reset my password, cannot log in",
    "Refund not received for cancelled order",
    "Error 500 when uploading a file",
    "How do I change my registered email"
]

# Lowercase, remove punctuation, and tokenize
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

clean_tickets = [preprocess(t) for t in tickets]

for ticket in clean_tickets:
    print(ticket.split())

# 2. Build TF-IDF features
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(clean_tickets)

print("\nTF-IDF shape:", tfidf.shape)

# Top 5 words in ticket 1
words = vectorizer.get_feature_names_out()
scores = tfidf[0].toarray()[0]
top_indices = np.argsort(scores)[::-1][:5]

print("\nTop 5 words in ticket 1:")
for i in top_indices:
    print(words[i], round(scores[i], 3))

# 3. Find the most similar ticket
query = preprocess("cannot log in to my account")
query_vector = vectorizer.transform([query])

similarities = cosine_similarity(query_vector, tfidf)[0]
best_index = similarities.argmax()

print("\nMost similar ticket:", tickets[best_index])
print("Ticket number:", best_index + 1)
print("Similarity score:", round(similarities[best_index], 3))

# 4. Create 30 labelled tickets (6 classes, 5 each)
training_tickets = [
    # Billing
    "I was charged twice",
    "Duplicate subscription payment",
    "My card was billed two times",
    "Incorrect amount charged",
    "Payment taken twice from account",

    # Technical
    "The app crashes on settings",
    "Application keeps crashing",
    "Settings page closes unexpectedly",
    "App freezes when opened",
    "Software crashes every time",

    # Login
    "I cannot log in to my account",
    "Please reset my password",
    "Unable to sign in",
    "Forgot my login password",
    "Account login is not working",

    # Refund
    "My cancelled order refund is missing",
    "I have not received my refund",
    "Refund for cancelled purchase",
    "Where is my refund payment",
    "Order cancelled but money not returned",

    # Upload
    "Error 500 while uploading a file",
    "File upload keeps failing",
    "Cannot upload my document",
    "Upload shows a server error",
    "File upload is not working",

    # Account
    "I need to change my registered email",
    "Update my email address",
    "How can I change account email",
    "Please update my registered email",
    "Change the email on my profile"
]

labels = (
    ["billing"] * 5 +
    ["technical"] * 5 +
    ["login"] * 5 +
    ["refund"] * 5 +
    ["upload"] * 5 +
    ["account"] * 5
)

# 5. Train and evaluate Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(
    training_tickets, labels,
    test_size=0.2, random_state=42, stratify=labels
)

model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("\nClassification accuracy:", round(accuracy * 100, 2), "%")
