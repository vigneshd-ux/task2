
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# 1. Load pretrained GloVe vectors
print("Loading GloVe model...")
model = api.load("glove-wiki-gigaword-50")
print("Model loaded successfully!")

# 2. Five nearest words
print("\n5 nearest words to king:")
print(model.most_similar("king", topn=5))

print("\n5 nearest words to computer:")
print(model.most_similar("computer", topn=5))

# 3. Word analogy: king - man + woman
print("\nAnalogy: king - man + woman")
print(model.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
))

# 4. Average word vectors for a sentence
def sentence_vector(sentence):
    words = sentence.lower().split()
    vectors = [model[word] for word in words if word in model]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)

sentence1 = "the king rules the kingdom"
sentence2 = "the queen rules the country"

vector1 = sentence_vector(sentence1)
vector2 = sentence_vector(sentence2)

similarity = cosine_similarity(
    vector1.reshape(1, -1),
    vector2.reshape(1, -1)
)[0][0]

print("\nSentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Cosine similarity:", round(float(similarity), 4))

# 5. Bonus: Plot 30 word vectors using PCA
words = [
    "king", "queen", "man", "woman", "boy", "girl",
    "computer", "software", "internet", "keyboard",
    "mouse", "phone", "science", "data", "school",
    "teacher", "student", "book", "city", "country",
    "car", "bus", "train", "apple", "orange",
    "happy", "sad", "love", "friend", "family"
]

vectors = np.array([model[word] for word in words])
pca = PCA(n_components=2)
points = pca.fit_transform(vectors)

plt.figure(figsize=(12, 8))
plt.scatter(points[:, 0], points[:, 1])

for i, word in enumerate(words):
    plt.annotate(word, (points[i, 0], points[i, 1]))

plt.title("GloVe Word Embeddings using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()
