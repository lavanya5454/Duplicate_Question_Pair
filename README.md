# Duplicate Question Pairs Detection

A machine learning application that identifies whether two questions are duplicates or paraphrases of each other. Built with Python and deployed using Streamlit for an interactive web interface.

## Overview

This project uses natural language processing and machine learning techniques to determine if two input questions have the same semantic meaning. It's particularly useful for:
- Content moderation on Q&A platforms
- Reducing duplicate questions in forums
- Improving search relevance
- Database deduplication

## Features

- **Text Preprocessing**: Comprehensive text cleaning including contraction expansion, HTML tag removal, and special character handling
- **Feature Engineering**: Multiple feature extraction techniques including:
  - Token-based features (common words, stopwords analysis)
  - Length-based features (absolute differences, longest common substring)
  - Fuzzy matching features (using FuzzyWuzzy library)
  - Bag-of-Words (BoW) vectorization
- **Interactive Web Interface**: User-friendly Streamlit application
- **Real-time Prediction**: Instant duplicate detection results

## Project Structure

```
├── app.py                 # Main Streamlit application
├── helper.py              # Feature extraction and preprocessing utilities
├── model.pkl              # Trained machine learning model
├── cv.pkl                 # CountVectorizer for BoW features
└── README.md              # Project documentation
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd duplicate-question-pairs
   ```

2. **Install required dependencies**
   ```bash
   pip install streamlit
   pip install beautifulsoup4
   pip install distance
   pip install fuzzywuzzy
   pip install numpy
   pip install nltk
   pip install scikit-learn
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## Usage

1. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

2. **Using the web interface**
   - Enter the first question in the "Enter question 1" field
   - Enter the second question in the "Enter question 2" field
   - Click the "Find" button
   - View the result: "Duplicate" or "Not Duplicate"

## Feature Engineering Details

### Preprocessing Steps
- **Special Character Replacement**: Converts symbols (%, $, ₹, €, @) to text equivalents
- **Number Normalization**: Converts large numbers to abbreviated forms (k, m, b)
- **Contraction Expansion**: Expands contractions (don't → do not, I'm → I am)
- **HTML Tag Removal**: Strips HTML tags using BeautifulSoup
- **Punctuation Removal**: Removes special characters and punctuation

### Feature Categories

1. **Basic Features** (7 features)
   - Character length of both questions
   - Word count of both questions
   - Common word count
   - Total unique words
   - Common words ratio

2. **Token Features** (8 features)
   - Common non-stopword ratios (min/max normalized)
   - Common stopword ratios (min/max normalized)
   - Common token ratios (min/max normalized)
   - First word match indicator
   - Last word match indicator

3. **Length Features** (3 features)
   - Absolute difference in token count
   - Average token length
   - Longest common substring ratio

4. **Fuzzy Features** (4 features)
   - Fuzzy ratio (overall similarity)
   - Partial ratio (substring matching)
   - Token sort ratio (sorted token matching)
   - Token set ratio (unique token matching)

5. **Bag-of-Words Features**
   - TF-IDF vectorized representation of both questions

## Model Architecture

The system uses a pre-trained machine learning model that processes 22 engineered features plus BoW vectors to make predictions. The model returns a binary classification:
- `1`: Questions are duplicates
- `0`: Questions are not duplicates

## Dependencies

- **streamlit**: Web application framework
- **beautifulsoup4**: HTML parsing and cleaning
- **distance**: String distance calculations
- **fuzzywuzzy**: Fuzzy string matching
- **numpy**: Numerical computations
- **nltk**: Natural language processing
- **scikit-learn**: Machine learning utilities
- **pickle**: Model serialization

## Example Usage

```python
# Example question pairs
q1 = "What is the capital of France?"
q2 = "What is France's capital city?"
# Result: Duplicate

q1 = "How to learn Python programming?"
q2 = "What is the weather like today?"
# Result: Not Duplicate
```


## Acknowledgments

- NLTK for natural language processing tools
- FuzzyWuzzy for string matching algorithms
- Streamlit for the web application framework
