using AiDotNet.Interfaces;
using AiDotNet.Tensors;

namespace AiDotNet.Finance.Interfaces;

/// <summary>
/// Interface for financial Natural Language Processing models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> with NLP-specific capabilities
/// for processing financial text, including sentiment analysis, entity extraction, and text classification.
/// </para>
/// <para>
/// <b>For Beginners:</b> Financial NLP models analyze text documents like:
///
/// - <b>News Articles:</b> Analyze market sentiment from financial news
/// - <b>SEC Filings:</b> Extract information from 10-K, 10-Q, 8-K filings
/// - <b>Earnings Calls:</b> Process transcripts for sentiment and key information
/// - <b>Social Media:</b> Monitor Twitter/Reddit for market-moving sentiment
/// - <b>Analyst Reports:</b> Extract recommendations and price targets
///
/// These models convert unstructured text into structured signals that can inform
/// trading decisions, risk assessment, and market analysis.
///
/// <b>Common Use Cases:</b>
/// - Sentiment scoring of news to predict price movements
/// - Named entity recognition for company names, stock tickers
/// - Document classification (earnings, M&A, legal, regulatory)
/// - Relationship extraction from financial documents
/// </para>
/// </remarks>
public interface IFinancialNLPModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets whether this model uses native mode (true) or ONNX mode (false).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Native mode allows training and uses pure C# layers.
    /// ONNX mode loads a pre-trained model for inference only.
    /// </para>
    /// </remarks>
    bool UseNativeMode { get; }

    /// <summary>
    /// Gets whether training is supported (only in native mode).
    /// </summary>
    bool SupportsTraining { get; }

    /// <summary>
    /// Gets the maximum sequence length (in tokens) that the model can process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the maximum number of tokens (roughly words/subwords)
    /// that can be processed in a single input. BERT-based models typically support 512 tokens.
    /// Longer documents need to be chunked or summarized.
    /// </para>
    /// </remarks>
    int MaxSequenceLength { get; }

    /// <summary>
    /// Gets the vocabulary size of the model's tokenizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The tokenizer converts text to numbers. The vocabulary size
    /// is how many unique tokens it knows. BERT uses ~30,000 tokens.
    /// </para>
    /// </remarks>
    int VocabularySize { get; }

    /// <summary>
    /// Gets the hidden dimension of the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The internal representation size for each token.
    /// BERT-base uses 768, BERT-large uses 1024.
    /// </para>
    /// </remarks>
    int HiddenDimension { get; }

    /// <summary>
    /// Gets the number of sentiment classes the model predicts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Most financial sentiment models use 3 classes:
    /// Positive, Negative, Neutral. Some use 5 classes including Very Positive/Negative.
    /// </para>
    /// </remarks>
    int NumSentimentClasses { get; }

    /// <summary>
    /// Analyzes sentiment from tokenized input.
    /// </summary>
    /// <param name="tokenIds">Tensor of token IDs [batch_size, sequence_length].</param>
    /// <returns>Sentiment probabilities [batch_size, num_sentiment_classes].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Takes tokenized text and returns sentiment probabilities.
    /// For a 3-class model: [P(negative), P(neutral), P(positive)].
    ///
    /// Example for financial news:
    /// - "Company beats earnings expectations" -> [0.05, 0.15, 0.80] (positive)
    /// - "Stock drops on weak guidance" -> [0.75, 0.20, 0.05] (negative)
    /// - "Company reports quarterly results" -> [0.10, 0.80, 0.10] (neutral)
    /// </para>
    /// </remarks>
    Tensor<T> AnalyzeSentiment(Tensor<T> tokenIds);

    /// <summary>
    /// Analyzes sentiment from raw text strings.
    /// </summary>
    /// <param name="texts">Array of text strings to analyze.</param>
    /// <returns>Sentiment results with class probabilities for each text.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Convenient method that handles tokenization internally.
    /// Returns a structured result with the predicted sentiment class and confidence.
    /// </para>
    /// </remarks>
    SentimentResult<T>[] AnalyzeSentiment(string[] texts);

    /// <summary>
    /// Gets embeddings (vector representations) for input tokens.
    /// </summary>
    /// <param name="tokenIds">Tensor of token IDs [batch_size, sequence_length].</param>
    /// <returns>Token embeddings [batch_size, sequence_length, hidden_dim].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Embeddings are dense vector representations of text.
    /// These can be used for similarity search, clustering, or as features for downstream tasks.
    /// The [CLS] token embedding (position 0) often represents the entire sequence.
    /// </para>
    /// </remarks>
    Tensor<T> GetEmbeddings(Tensor<T> tokenIds);

    /// <summary>
    /// Gets the [CLS] token embedding representing the entire input sequence.
    /// </summary>
    /// <param name="tokenIds">Tensor of token IDs [batch_size, sequence_length].</param>
    /// <returns>Sequence embeddings [batch_size, hidden_dim].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In BERT-style models, the [CLS] token is a special token
    /// at the beginning whose embedding represents the entire sequence. This is commonly
    /// used for classification tasks and document similarity.
    /// </para>
    /// </remarks>
    Tensor<T> GetSequenceEmbedding(Tensor<T> tokenIds);

    /// <summary>
    /// Tokenizes raw text into token IDs.
    /// </summary>
    /// <param name="text">The text to tokenize.</param>
    /// <param name="maxLength">Maximum sequence length (will pad or truncate).</param>
    /// <returns>Array of token IDs.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Tokenization converts text to numbers that the model understands.
    /// BERT uses WordPiece tokenization where rare words are split into subwords.
    ///
    /// Example: "investing" might become ["invest", "##ing"]
    /// </para>
    /// </remarks>
    int[] Tokenize(string text, int? maxLength = null);

    /// <summary>
    /// Converts token IDs back to text.
    /// </summary>
    /// <param name="tokenIds">Array of token IDs.</param>
    /// <returns>Reconstructed text string.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Converts the numeric token IDs back into human-readable text.
    /// Useful for debugging and understanding what the model sees.
    /// </para>
    /// </remarks>
    string Detokenize(int[] tokenIds);

    /// <summary>
    /// Gets financial-specific NLP metrics from the model.
    /// </summary>
    /// <returns>Dictionary containing NLP metrics like accuracy, F1 score, etc.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns metrics relevant to NLP performance:
    /// - Accuracy: Percentage of correct predictions
    /// - F1 Score: Harmonic mean of precision and recall
    /// - AUC: Area under ROC curve
    /// - ParameterCount: Number of model parameters
    /// </para>
    /// </remarks>
    Dictionary<string, T> GetFinancialMetrics();
}

/// <summary>
/// Result of sentiment analysis on a single text.
/// </summary>
/// <typeparam name="T">The numeric type for probabilities.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class holds the sentiment prediction for a piece of text,
/// including the predicted class (positive/negative/neutral), the confidence score,
/// and the probability distribution over all classes.
/// </para>
/// </remarks>
public class SentimentResult<T>
{
    /// <summary>
    /// Gets or sets the predicted sentiment class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The most likely sentiment: "positive", "negative", or "neutral".
    /// For FinBERT, these correspond to financial sentiment, not general sentiment.
    /// </para>
    /// </remarks>
    public string PredictedClass { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence score (probability of predicted class).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How confident the model is in its prediction (0 to 1).
    /// Higher values indicate more confident predictions. A score of 0.95 means
    /// the model is 95% confident in its prediction.
    /// </para>
    /// </remarks>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Gets or sets the probability distribution over all sentiment classes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Shows the probability for each possible class.
    /// Useful when you need to see how close the decision was.
    /// Example: { "negative": 0.05, "neutral": 0.15, "positive": 0.80 }
    /// </para>
    /// </remarks>
    public Dictionary<string, T> ClassProbabilities { get; set; } = new();

    /// <summary>
    /// Gets or sets the original text that was analyzed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The input text for reference, useful when
    /// analyzing multiple texts in batch.
    /// </para>
    /// </remarks>
    public string OriginalText { get; set; } = string.Empty;
}
