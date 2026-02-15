using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a text vectorizer that converts text documents to numeric feature matrices.
/// </summary>
/// <remarks>
/// <para>
/// Text vectorizers transform collections of text documents into numeric representations
/// suitable for machine learning algorithms. They follow the sklearn-style Fit/Transform
/// pattern where the vectorizer first learns from training data, then transforms any text.
/// </para>
/// <para><b>For Beginners:</b> A text vectorizer converts words into numbers that ML models can understand.
/// Different vectorizers use different strategies:
/// - TF-IDF: Weights words by importance (rare words score higher)
/// - Count: Simply counts how many times each word appears
/// - Hashing: Uses hashing for memory efficiency with large vocabularies
/// - BM25: Improved TF-IDF used by search engines
/// - Word2Vec/Doc2Vec: Creates dense embeddings capturing semantic meaning
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for the output matrix (e.g., float, double).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("TextVectorizer")]
public interface ITextVectorizer<T>
{
    /// <summary>
    /// Gets whether this vectorizer has been fitted to data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns true after <see cref="Fit"/> or <see cref="FitTransform"/> has been called.
    /// Some vectorizers (like HashingVectorizer) don't require fitting and always return true.
    /// </para>
    /// <para><b>For Beginners:</b> Check this before calling Transform() to ensure
    /// the vectorizer has learned the vocabulary from training data.
    /// </para>
    /// </remarks>
    bool IsFitted { get; }

    /// <summary>
    /// Gets the vocabulary mapping (token to index).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns null if the vectorizer hasn't been fitted yet, or if the vectorizer
    /// doesn't maintain an explicit vocabulary (like HashingVectorizer).
    /// </para>
    /// <para><b>For Beginners:</b> The vocabulary tells you which words the vectorizer knows
    /// and what column index each word maps to in the output matrix.
    /// </para>
    /// </remarks>
    Dictionary<string, int>? Vocabulary { get; }

    /// <summary>
    /// Gets the feature names (vocabulary terms in order).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns the terms in the order they appear as columns in the output matrix.
    /// Returns null if not fitted, or synthetic names for vectorizers without explicit vocabulary.
    /// </para>
    /// <para><b>For Beginners:</b> These are the column names for your feature matrix.
    /// Each name corresponds to a word (or n-gram) that the vectorizer learned.
    /// </para>
    /// </remarks>
    string[]? FeatureNames { get; }

    /// <summary>
    /// Gets the number of features (vocabulary size) this vectorizer produces.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns the number of columns in the output matrix. For vocabulary-based vectorizers,
    /// this equals the vocabulary size. For hashing vectorizers, this is the configured hash size.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many numeric features each document
    /// will be converted into. Use this to configure neural network input sizes.
    /// </para>
    /// </remarks>
    int FeatureCount { get; }

    /// <summary>
    /// Fits the vectorizer to the training documents, learning the vocabulary.
    /// </summary>
    /// <param name="documents">The training documents to learn from.</param>
    /// <remarks>
    /// <para>
    /// This method analyzes the documents to build a vocabulary and compute any
    /// statistics needed for transformation (like IDF weights for TF-IDF).
    /// </para>
    /// <para><b>For Beginners:</b> Call this once on your training data. The vectorizer
    /// will learn which words exist and how important they are. After fitting,
    /// you can transform any text using the same vocabulary.
    /// </para>
    /// </remarks>
    void Fit(IEnumerable<string> documents);

    /// <summary>
    /// Transforms documents to a numeric feature matrix.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>A matrix where each row is a document and each column is a feature.</returns>
    /// <exception cref="InvalidOperationException">Thrown if Fit() has not been called (for vectorizers that require fitting).</exception>
    /// <remarks>
    /// <para>
    /// Converts each document into a numeric vector based on the learned vocabulary.
    /// Words not in the vocabulary are typically ignored (except for hashing vectorizers).
    /// </para>
    /// <para><b>For Beginners:</b> Use this to convert new text (like test data or user input)
    /// into numbers using the same rules learned from training data.
    /// </para>
    /// </remarks>
    Matrix<T> Transform(IEnumerable<string> documents);

    /// <summary>
    /// Fits the vectorizer and transforms the documents in one step.
    /// </summary>
    /// <param name="documents">The documents to fit and transform.</param>
    /// <returns>A matrix where each row is a document and each column is a feature.</returns>
    /// <remarks>
    /// <para>
    /// Convenience method that combines Fit and Transform. Use this for training data
    /// where you want to learn the vocabulary and transform in one call.
    /// </para>
    /// <para><b>For Beginners:</b> Use this for your training data. It's equivalent to
    /// calling Fit() then Transform(), but more convenient and potentially more efficient.
    /// </para>
    /// </remarks>
    Matrix<T> FitTransform(IEnumerable<string> documents);

    /// <summary>
    /// Gets the output feature names.
    /// </summary>
    /// <returns>Array of feature names corresponding to output columns.</returns>
    /// <remarks>
    /// <para>
    /// Returns meaningful names for interpreting the output features.
    /// For vocabulary-based vectorizers, these are the actual terms.
    /// For hashing vectorizers, these are synthetic names like "hash_0", "hash_1", etc.
    /// </para>
    /// <para><b>For Beginners:</b> Use this to understand what each column in your
    /// feature matrix represents. Helpful for model interpretability.
    /// </para>
    /// </remarks>
    string[] GetFeatureNamesOut();
}
