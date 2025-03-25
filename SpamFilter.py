# Bencze Marton

import os
import string
from collections import defaultdict
import math


def load_stopwords(filepath):
    file = open(filepath, 'r')
    return set(word.strip().lower() for word in file.readlines())


def preprocess_text(text, stopwords):
    tokens = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    # tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [word for word in tokens if word not in stopwords]
    return tokens


def load_file_list(filepath):
    file = open(filepath, 'r')
    return [line.strip() for line in file.readlines()]


def load_files(file_list, base_path):
    contents = []
    for filename in file_list:
        file = open(os.path.join(base_path, filename), 'r', encoding='latin-1')
        contents.append(file.read())
    return contents


def train_naive_bayes(train_files, train_labels, stopwords, base_path):
    word_counts = {'spam': defaultdict(int), 'ham': defaultdict(int)}
    class_counts = {'spam': 0, 'ham': 0}
    total_words = {'spam': 0, 'ham': 0}
    vocab = set()

    for filename, label in zip(train_files, train_labels):
        file_path = os.path.join(base_path, 'spam' if label == 'spam' else 'ham', filename)
        file = open(file_path, 'r', encoding='latin-1')
        text = file.read()
        tokens = preprocess_text(text, stopwords)
        class_counts[label] += 1
        for word in tokens:
            word_counts[label][word] += 1
            if word not in vocab:
                vocab.add(word)
            total_words[label] += 1

    return word_counts, class_counts, total_words, vocab


def calculate_word_probability(word, label, word_counts, total_words, vocab, alpha):
    return (word_counts[label][word] + alpha) / (total_words[label] + alpha * len(vocab))


def calculate_class_probability(label, class_counts):
    total_documents = sum(class_counts.values())
    return class_counts[label] / total_documents


def classify(text, word_counts, class_counts, total_words, vocab, stopwords, alpha):
    tokens = preprocess_text(text, stopwords)
    spam_score = math.log(calculate_class_probability('spam', class_counts))
    ham_score = math.log(calculate_class_probability('ham', class_counts))

    for word in tokens:
        if word in vocab:
            spam_score += math.log(calculate_word_probability(word, 'spam', word_counts, total_words, vocab, alpha))
            ham_score += math.log(calculate_word_probability(word, 'ham', word_counts, total_words, vocab, alpha))

    return 'spam' if spam_score > ham_score else 'ham'


def calculate_error(files, labels, word_counts, class_counts, total_words, vocab, stopwords, alpha, base_path):
    errors = 0
    for filename, true_label in zip(files, labels):
        file_path = os.path.join(base_path, 'spam' if true_label == 'spam' else 'ham', filename)
        file = open(file_path, 'r', encoding='latin-1')
        text = file.read()
        predicted_label = classify(text, word_counts, class_counts, total_words, vocab, stopwords, alpha)
        if predicted_label != true_label:
            errors += 1
    return errors / len(files)


def additiv(train_files, train_labels, test_files, test_labels, stopwords, alphas, base_path):
    # train
    word_counts, class_counts, total_words, vocab = train_naive_bayes(train_files, train_labels, stopwords, base_path)
    results = {}

    for alpha in alphas:
        train_error = calculate_error(train_files, train_labels, word_counts, class_counts, total_words, vocab,
                                      stopwords, alpha, base_path)
        test_error = calculate_error(test_files, test_labels, word_counts, class_counts, total_words, vocab, stopwords,
                                     alpha, base_path)
        results[alpha] = (train_error, test_error)
    return results


stopwords = load_stopwords('stopwords.txt')
train_files = load_file_list('train.txt')
train_labels = ['ham' if 'ham' in file else 'spam' for file in train_files]

test_files = load_file_list('test.txt')
test_labels = ['ham' if 'ham' in file else 'spam' for file in test_files]
alphas = [0.01, 0.1, 1]
results = additiv(train_files, train_labels, test_files, test_labels, stopwords, alphas, 'enron6')

for alpha, (train_error, test_error) in results.items():
    print(f'Alpha: {alpha}, Train Error: {train_error}, Test Error: {test_error}')
