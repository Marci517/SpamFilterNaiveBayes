# Naive Bayes Spam Classifier

This project implements a Naive Bayes classifier for detecting spam messages using email text data. The classifier uses additive (Laplace) smoothing and is trained and tested on labeled examples of spam and ham (non-spam) emails.

A core component of this project is the **machine learning-based training phase**, where the classifier learns from previously labeled examples to generalize and classify unseen messages. This approach leverages fundamental concepts from natural language processing and probabilistic modeling to distinguish between spam and legitimate content.

## Functionality

- Loads and filters stopwords from a given file
- Preprocesses text data by removing punctuation, converting to lowercase, and excluding stopwords
- **Trains a Naive Bayes classifier** using labeled data, implementing a supervised learning approach
- Evaluates classification performance on both training and test data
- Supports multiple smoothing (alpha) values to compare error rates and observe model generalization

## File Structure

- `train.txt`: List of filenames used for training, each line referring to either a ham or spam email
- `test.txt`: List of filenames used for testing
- `stopwords.txt`: List of stopwords to remove during preprocessing
- `SpamFilter.py`: Main Python script containing the Naive Bayes implementation and training logic
- `README.md`: This documentation file
- `enron6/`
  - `ham/`: Contains ham (non-spam) email files
  - `spam/`: Contains spam email files
