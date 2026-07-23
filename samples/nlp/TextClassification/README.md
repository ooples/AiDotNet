# Text Classification - News Article Categorization

This sample demonstrates multi-class text classification using AiDotNet to categorize news articles into different topics.

## What You'll Learn

- How to preprocess text data for machine learning (TF-IDF vectorization)
- How to build a multi-class text classifier using Multinomial Naive Bayes
- How to evaluate classification performance with precision, recall, and F1-score
- How to interpret confusion matrices for multi-class problems

## Dataset

The sample uses a synthetic news article dataset with 4 categories:

| Category | Description | Examples |
|----------|-------------|----------|
| Technology | Tech news, gadgets, software | "Apple announces new iPhone...", "Google releases Android update..." |
| Sports | Sports events, athletes, teams | "Lakers defeat Celtics...", "World Cup final draws..." |
| Politics | Government, elections, policy | "President signs legislation...", "Congress debates bill..." |
| Business | Markets, finance, economy | "Stock market reaches highs...", "Fed announces rate decision..." |

## Running the Sample

```bash
cd samples/nlp/TextClassification
dotnet run
```

## Expected Output

```
=== AiDotNet Text Classification ===
News Article Categorization with Multi-Class Classification

Loaded 60 news articles across 4 categories

Categories:
  0. Technology: 15 articles
  1. Sports: 15 articles
  2. Politics: 15 articles
  3. Business: 15 articles

Training set: 48 articles
Test set: 12 articles

Preprocessing: Converting text to TF-IDF features...
  Vocabulary size: 312 terms
  Feature vector size: 312

Building Multinomial Naive Bayes classifier...
  - Smoothing: Laplace (alpha=1.0)
  - Suitable for: Multi-class text classification

Training classifier...

Evaluation Results:
------------------------------------------------------------

Per-Category Metrics:
Category          Precision     Recall   F1-Score    Support
------------------------------------------------------------
Technology            90.0%      90.0%      90.0%          3
Sports               100.0%     100.0%     100.0%          3
Politics              85.7%     100.0%      92.3%          3
Business             100.0%      85.7%      92.3%          3
------------------------------------------------------------
Weighted Avg          93.9%      93.9%      93.7%         12

Overall Accuracy: 93.94%

Confusion Matrix:
                Predicted ->
Actual          Technology     Sports   Politics   Business
---------------------------------------------------------------
Technology               3          0          0          0
Sports                   0          3          0          0
Politics                 0          0          3          0
Business                 0          0          0          3

Sample Predictions:
------------------------------------------------------------

Article 1: "Samsung unveils foldable smartphone with improved du..."
  Predicted: Technology
  Actual:    Technology [Correct]

Article 2: "Golf champion wins fourth major tournament of the se..."
  Predicted: Sports
  Actual:    Sports [Correct]
```

## How It Works

### 1. Text Preprocessing (TF-IDF)

Text is converted to numerical features using TF-IDF (Term Frequency-Inverse Document Frequency):

```
TF-IDF(term, document) = TF(term, document) * IDF(term)

Where:
- TF = frequency of term in document
- IDF = log(total_documents / documents_containing_term)
```

This gives higher weights to distinctive terms and lower weights to common words.

### 2. Multinomial Naive Bayes

The classifier uses Bayes' theorem:

```
P(category | document) proportional to P(category) * P(document | category)
```

- **P(category)**: Prior probability of each category
- **P(document | category)**: Likelihood based on word frequencies
- **Laplace smoothing**: Handles unseen words (alpha=1.0)

### 3. Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| Precision | TP / (TP + FP) | Of predicted positives, how many are correct? |
| Recall | TP / (TP + FN) | Of actual positives, how many did we find? |
| F1-Score | 2 * (P * R) / (P + R) | Harmonic mean of precision and recall |

## Code Highlights

```csharp
// Create and train classifier
var classifier = new MultinomialNaiveBayes<double, double[], double>(alpha: 1.0);
classifier.Train(trainFeatures, trainLabels);

// Make predictions
double prediction = classifier.Predict(features);
```

## Architecture

```
                Input Text
                    |
                    v
          +-----------------+
          |   Tokenization  |
          |  (split, lower) |
          +-----------------+
                    |
                    v
          +-----------------+
          | Stop Word       |
          | Removal         |
          +-----------------+
                    |
                    v
          +-----------------+
          | TF-IDF          |
          | Vectorization   |
          +-----------------+
                    |
                    v
          +-----------------+
          | Multinomial     |
          | Naive Bayes     |
          +-----------------+
                    |
                    v
              Category Label
```

## Customization

### Different Classifiers

Replace Naive Bayes with other classifiers:

```csharp
// Support Vector Machine
var svm = new LinearSupportVectorClassifier<double, double[], double>();

// Random Forest
var rf = new RandomForestClassifier<double, double[], double>(nEstimators: 100);

// Logistic Regression via SGD
var sgd = new SGDClassifier<double, double[], double>();
```

### Better Text Features

For production use, consider:

- **N-grams**: Capture word sequences ("machine learning" vs "machine" + "learning")
- **Word embeddings**: Use pre-trained embeddings (Word2Vec, GloVe)
- **Transformer embeddings**: Use BERT or similar for semantic understanding

## Next Steps

- [Embeddings](../Embeddings/) - Learn about text embeddings for semantic similarity
- [BasicRAG](../RAG/BasicRAG/) - Build question-answering systems
