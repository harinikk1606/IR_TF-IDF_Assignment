import os
import math
import re
import nltk

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

DOCUMENT_FOLDER = "documents"

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess(text):
    text = text.lower()
    words = re.findall(r"[a-z]+", text)

    processed_words = []

    for word in words:
        if word not in stop_words:
            processed_words.append(stemmer.stem(word))

    return processed_words


def load_documents():
    documents = {}

    for filename in sorted(os.listdir(DOCUMENT_FOLDER)):
        if filename.endswith(".txt"):
            file_path = os.path.join(DOCUMENT_FOLDER, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                documents[filename] = file.read()

    return documents


def calculate_tf(words):
    tf = {}
    total_words = len(words )

    if total_words == 0:
        return tf

    for word in words:
        tf[word] = tf.get(word, 0) + 1

    for word in tf:
        tf[word] = tf[word] / total_words

    return tf


def calculate_idf(processed_documents):
    idf = {}
    total_documents = len(processed_documents)
    vocabulary = set()

    for words in processed_documents.values():
        vocabulary.update(words)

    for term in vocabulary: 
        document_count = sum(
            1 for words in processed_documents.values()
            if term in words
        )

        idf[term] = math.log(
            (total_documents + 1) / (document_count + 1)
        ) + 1

    return idf


def calculate_tfidf(tf, idf):
    tfidf = {}

    for term in tf:
        if term in idf:
            tfidf[term] = tf[term] * idf[term]

    return tfidf


def cosine_similarity(vector1, vector2):
    common_terms = set(vector1.keys()) & set(vector2.keys())

    if not common_terms:
        return 0.0

    dot_product = sum(
        vector1[term] * vector2[term]
        for term in common_terms
    )

    magnitude1 = math.sqrt(
        sum(value ** 2 for value in vector1.values())
    )

    magnitude2 = math.sqrt(
        sum(value ** 2 for value in vector2.values())
    )

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def main():
    print("=" * 60)
    print("INFORMATION RETRIEVAL SYSTEM")
    print("TF-IDF DOCUMENT SEARCH")
    print("=" * 60)

    documents = load_documents()

    if not documents:
        print("No TXT documents found.")
        return

    print(f"\nNumber of documents loaded: {len(documents)}")

    processed_documents = {
        filename: preprocess(text)
        for filename, text in documents.items()
    }

    print("Text preprocessing completed.")

    idf = calculate_idf(processed_documents)

    document_vectors = {}

    for filename, words in processed_documents.items():
        tf = calculate_tf(words)
        document_vectors[filename] = calculate_tfidf(tf, idf)

    print("TF-IDF calculation completed.")

    print("\n" + "-" * 60)

    query = input("Enter your search query: ")

    if not query.strip():
        print("Query cannot be empty.")
        return

    processed_query = preprocess(query)

    query_tf = calculate_tf(processed_query)

    query_vector = calculate_tfidf(query_tf, idf)

    results = []

    for filename, document_vector in document_vectors.items():
        score = cosine_similarity(
            query_vector,
            document_vector
        )

        results.append((filename, score))

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    print(f"\nQuery: {query}")
    print("\nDocuments ranked by TF-IDF cosine similarity:\n")

    for rank, (filename, score) in enumerate(results, start=1):
        print(f"{rank}. {filename}")
        print(f"   Score: {score:.4f}\n")

    print("=" * 60)


if __name__ == "__main__":
    main()