using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Base class for text vectorizers providing common functionality.
/// </summary>
/// <remarks>
/// <para>
/// This base class provides shared implementation for text vectorization including:
/// - Tokenization (splitting text into words/tokens)
/// - N-gram generation (creating word sequences)
/// - Stop word filtering
/// - Vocabulary management
/// </para>
/// <para>
/// Supports two tokenization approaches:
/// <list type="bullet">
/// <item><b>Simple Tokenizer (default):</b> Uses word-level tokenization via a custom function
/// or the built-in whitespace/punctuation splitter. Best for traditional NLP tasks.</item>
/// <item><b>ITokenizer Integration:</b> Uses advanced subword tokenization (BPE, WordPiece, SentencePiece)
/// from the AiDotNet.Tokenization module. Enables consistent tokenization between classical ML
/// and neural network approaches.</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all text vectorizers build upon.
/// It handles the common task of breaking text into tokens and managing which words
/// the vectorizer knows about. Specific vectorizers (TF-IDF, Count, etc.) add their
/// own logic for how to convert those tokens into numbers.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for the output matrix (e.g., float, double).</typeparam>
public abstract class TextVectorizerBase<T> : ITextVectorizer<T>
{
    /// <summary>
    /// Numeric operations helper for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Minimum document frequency (absolute count).
    /// </summary>
    protected readonly int _minDf;

    /// <summary>
    /// Maximum document frequency (proportion 0-1).
    /// </summary>
    protected readonly double _maxDf;

    /// <summary>
    /// Maximum number of features (vocabulary size limit).
    /// </summary>
    protected readonly int? _maxFeatures;

    /// <summary>
    /// N-gram range (min, max) for token generation.
    /// </summary>
    protected readonly (int Min, int Max) _nGramRange;

    /// <summary>
    /// Whether to convert text to lowercase.
    /// </summary>
    protected readonly bool _lowercase;

    /// <summary>
    /// Custom tokenizer function (simple word-level tokenization).
    /// </summary>
    protected readonly Func<string, IEnumerable<string>>? _tokenizer;

    /// <summary>
    /// Advanced tokenizer implementing ITokenizer (BPE, WordPiece, SentencePiece).
    /// </summary>
    /// <remarks>
    /// <para>
    /// When set, this takes precedence over the simple tokenizer function.
    /// Enables using the same subword tokenization used by neural networks (BERT, GPT, etc.)
    /// with traditional text vectorizers like TF-IDF and BM25.
    /// </para>
    /// </remarks>
    protected readonly ITokenizer? _advancedTokenizer;

    /// <summary>
    /// Set of stop words to exclude.
    /// </summary>
    protected readonly HashSet<string>? _stopWords;

    /// <summary>
    /// The learned vocabulary (token to index mapping).
    /// </summary>
    protected Dictionary<string, int>? _vocabulary;

    /// <summary>
    /// The feature names (tokens in order).
    /// </summary>
    protected string[]? _featureNames;

    /// <summary>
    /// Number of documents seen during fitting.
    /// </summary>
    protected int _nDocs;

    /// <inheritdoc/>
    public virtual bool IsFitted => _vocabulary is not null;

    /// <inheritdoc/>
    public Dictionary<string, int>? Vocabulary => _vocabulary;

    /// <inheritdoc/>
    public string[]? FeatureNames => _featureNames;

    /// <inheritdoc/>
    public virtual int FeatureCount => _featureNames?.Length ?? 0;

    /// <summary>
    /// Initializes a new instance of the text vectorizer base class.
    /// </summary>
    /// <param name="minDf">Minimum document frequency (absolute count). Defaults to 1.</param>
    /// <param name="maxDf">Maximum document frequency (proportion 0-1). Defaults to 1.0.</param>
    /// <param name="maxFeatures">Maximum vocabulary size. Null for unlimited.</param>
    /// <param name="nGramRange">N-gram range (min, max). Defaults to (1, 1) for unigrams.</param>
    /// <param name="lowercase">Convert all text to lowercase. Defaults to true.</param>
    /// <param name="tokenizer">Custom tokenizer function. Null for default word-level tokenization.</param>
    /// <param name="stopWords">Words to exclude. Null for no filtering.</param>
    /// <param name="advancedTokenizer">
    /// Optional ITokenizer for subword tokenization (BPE, WordPiece, SentencePiece).
    /// When provided, this takes precedence over the simple tokenizer function.
    /// Enables consistent tokenization between classical ML and neural network approaches.
    /// </param>
    /// <remarks>
    /// <para><b>Tokenization Priority:</b></para>
    /// <list type="number">
    /// <item>If <paramref name="advancedTokenizer"/> is provided, it is used (subword tokens).</item>
    /// <item>Else if <paramref name="tokenizer"/> function is provided, it is used (custom word-level).</item>
    /// <item>Else the default whitespace/punctuation splitter is used (simple word-level).</item>
    /// </list>
    /// <para><b>For Beginners:</b> For most traditional NLP tasks, the default tokenization works well.
    /// Use <paramref name="advancedTokenizer"/> if you want to match the tokenization used by
    /// neural network models like BERT, GPT, or if you need to handle rare words via subword splitting.
    /// </para>
    /// </remarks>
    protected TextVectorizerBase(
        int minDf = 1,
        double maxDf = 1.0,
        int? maxFeatures = null,
        (int Min, int Max)? nGramRange = null,
        bool lowercase = true,
        Func<string, IEnumerable<string>>? tokenizer = null,
        HashSet<string>? stopWords = null,
        ITokenizer? advancedTokenizer = null)
    {
        if (minDf < 1)
            throw new ArgumentException("Minimum document frequency must be at least 1.", nameof(minDf));
        if (maxDf < 0 || maxDf > 1)
            throw new ArgumentException("Maximum document frequency must be between 0 and 1.", nameof(maxDf));

        _minDf = minDf;
        _maxDf = maxDf;
        _maxFeatures = maxFeatures;
        _nGramRange = nGramRange ?? (1, 1);
        _lowercase = lowercase;
        _tokenizer = tokenizer;
        _stopWords = stopWords;
        _advancedTokenizer = advancedTokenizer;
    }

    /// <inheritdoc/>
    public abstract void Fit(IEnumerable<string> documents);

    /// <inheritdoc/>
    public abstract Matrix<T> Transform(IEnumerable<string> documents);

    /// <inheritdoc/>
    public virtual Matrix<T> FitTransform(IEnumerable<string> documents)
    {
        var docList = documents.ToList();
        Fit(docList);
        return Transform(docList);
    }

    /// <inheritdoc/>
    public virtual string[] GetFeatureNamesOut()
    {
        return _featureNames ?? Array.Empty<string>();
    }

    /// <summary>
    /// Tokenizes a document into individual tokens.
    /// </summary>
    /// <param name="text">The text to tokenize.</param>
    /// <returns>Enumerable of tokens.</returns>
    /// <remarks>
    /// <para>
    /// Tokenization priority:
    /// <list type="number">
    /// <item>If an ITokenizer (advancedTokenizer) is configured, uses subword tokenization.</item>
    /// <item>If a custom tokenizer function is configured, uses that.</item>
    /// <item>Otherwise, uses the default whitespace/punctuation splitter.</item>
    /// </list>
    /// </para>
    /// <para><b>For Beginners:</b> Tokenization splits text into individual words or tokens.
    /// By default, it splits on whitespace and punctuation, converts to lowercase if configured,
    /// and removes stop words if configured. When using an ITokenizer, you get subword tokens
    /// (like "un", "##happy" for "unhappy" in WordPiece) which can handle rare words better.
    /// </para>
    /// </remarks>
    protected virtual IEnumerable<string> Tokenize(string text)
    {
        // Priority 1: Use advanced ITokenizer if provided (subword tokenization)
        if (_advancedTokenizer is not null)
        {
            // Use the ITokenizer.Tokenize method which returns subword tokens as strings
            var tokens = _advancedTokenizer.Tokenize(text);
            IEnumerable<string> result = tokens;

            if (_lowercase)
                result = result.Select(t => t.ToLowerInvariant());
            if (_stopWords is not null)
                result = result.Where(t => !_stopWords.Contains(t));

            return result;
        }

        // Priority 2: Use custom tokenizer function if provided
        if (_tokenizer is not null)
        {
            var tokens = _tokenizer(text);
            if (_lowercase)
                tokens = tokens.Select(t => t.ToLowerInvariant());
            if (_stopWords is not null)
                tokens = tokens.Where(t => !_stopWords.Contains(t));
            return tokens;
        }

        // Priority 3: Default tokenizer - split on whitespace and punctuation
        string processedText = _lowercase ? text.ToLowerInvariant() : text;
        var tokenList = new List<string>();
        var currentToken = new System.Text.StringBuilder();

        foreach (char c in processedText)
        {
            if (char.IsLetterOrDigit(c))
            {
                currentToken.Append(c);
            }
            else if (currentToken.Length > 0)
            {
                string token = currentToken.ToString();
                if (_stopWords is null || !_stopWords.Contains(token))
                    tokenList.Add(token);
                currentToken.Clear();
            }
        }

        if (currentToken.Length > 0)
        {
            string token = currentToken.ToString();
            if (_stopWords is null || !_stopWords.Contains(token))
                tokenList.Add(token);
        }

        return tokenList;
    }

    /// <summary>
    /// Generates n-grams from a sequence of tokens.
    /// </summary>
    /// <param name="tokens">The tokens to generate n-grams from.</param>
    /// <returns>Enumerable of n-grams.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> N-grams are sequences of N consecutive words.
    /// For example, with nGramRange=(1,2), the text "the cat sat" produces:
    /// - Unigrams (1-grams): "the", "cat", "sat"
    /// - Bigrams (2-grams): "the cat", "cat sat"
    ///
    /// N-grams help capture word context and phrases.
    /// </para>
    /// </remarks>
    protected virtual IEnumerable<string> GenerateNGrams(IEnumerable<string> tokens)
    {
        var tokenList = tokens.ToList();
        var ngrams = new List<string>();

        for (int n = _nGramRange.Min; n <= _nGramRange.Max; n++)
        {
            for (int i = 0; i <= tokenList.Count - n; i++)
            {
                string ngram = string.Join(" ", tokenList.Skip(i).Take(n));
                ngrams.Add(ngram);
            }
        }

        return ngrams;
    }

    /// <summary>
    /// Builds vocabulary from document frequency counts.
    /// </summary>
    /// <param name="dfCounts">Document frequency counts for each token.</param>
    /// <param name="totalCounts">Total occurrence counts for each token (for sorting).</param>
    /// <remarks>
    /// <para>
    /// Filters tokens by document frequency, sorts by total count, applies max features limit,
    /// and builds the final vocabulary dictionary.
    /// </para>
    /// </remarks>
    protected virtual void BuildVocabulary(
        Dictionary<string, int> dfCounts,
        Dictionary<string, int> totalCounts)
    {
        // Filter by document frequency
        double maxDfCount = _maxDf * _nDocs;
        var filteredTokens = dfCounts
            .Where(kvp => kvp.Value >= _minDf && kvp.Value <= maxDfCount)
            .Select(kvp => kvp.Key)
            .ToList();

        // Sort by total count (descending) for consistent ordering
        filteredTokens = filteredTokens
            .OrderByDescending(t => totalCounts.GetValueOrDefault(t, 0))
            .ThenBy(t => t)
            .ToList();

        // Apply max features limit
        if (_maxFeatures.HasValue && filteredTokens.Count > _maxFeatures.Value)
            filteredTokens = filteredTokens.Take(_maxFeatures.Value).ToList();

        // Sort alphabetically for final vocabulary
        filteredTokens = filteredTokens.OrderBy(t => t).ToList();

        // Build vocabulary
        _vocabulary = new Dictionary<string, int>();
        for (int i = 0; i < filteredTokens.Count; i++)
            _vocabulary[filteredTokens[i]] = i;

        _featureNames = filteredTokens.ToArray();
    }

    /// <summary>
    /// Common English stop words.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Stop words are common words like "the", "is", "at"
    /// that usually don't carry much meaning for classification tasks. Removing them
    /// can reduce noise and improve model performance.
    /// </para>
    /// </remarks>
    public static HashSet<string> EnglishStopWords => new HashSet<string>
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "the", "this", "but", "they",
        "have", "had", "what", "when", "where", "who", "which", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "can", "just", "should", "now", "also", "into", "over", "after",
        "before", "between", "under", "again", "further", "then", "once", "here",
        "there", "any", "about", "above", "below", "up", "down", "out", "off",
        "being", "been", "having", "does", "did", "doing", "would", "could",
        "shall", "might", "must", "need", "dare", "ought", "used", "am", "i",
        "you", "your", "yours", "yourself", "yourselves", "we", "our", "ours",
        "ourselves", "me", "my", "mine", "myself", "him", "his", "himself",
        "she", "her", "hers", "herself", "them", "their", "theirs", "themselves"
    };
}
