using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Converts a collection of text documents to a matrix of token counts.
/// </summary>
/// <remarks>
/// <para>
/// CountVectorizer tokenizes text documents and builds a vocabulary, then
/// encodes each document as a count vector (bag of words representation).
/// </para>
/// <para>
/// Features include:
/// - Customizable tokenization
/// - N-gram support (unigrams, bigrams, etc.)
/// - Minimum and maximum document frequency thresholds
/// - Maximum vocabulary size
/// </para>
/// <para><b>For Beginners:</b> CountVectorizer turns text into numbers:
/// - Each unique word becomes a column
/// - Each document becomes a row
/// - Values are word counts
///
/// Example: ["I like cats", "I like dogs"] becomes:
///          | I | like | cats | dogs |
/// Doc 1:   | 1 |  1   |  1   |  0   |
/// Doc 2:   | 1 |  1   |  0   |  1   |
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class CountVectorizer<T>
{
    private readonly int _minDf;
    private readonly double _maxDf;
    private readonly int? _maxFeatures;
    private readonly (int Min, int Max) _nGramRange;
    private readonly bool _lowercase;
    private readonly bool _binary;
    private readonly Func<string, IEnumerable<string>>? _tokenizer;
    private readonly HashSet<string>? _stopWords;

    // Fitted parameters
    private Dictionary<string, int>? _vocabulary;
    private string[]? _featureNames;

    /// <summary>
    /// Gets the vocabulary (token to index mapping).
    /// </summary>
    public Dictionary<string, int>? Vocabulary => _vocabulary;

    /// <summary>
    /// Gets the feature names (tokens).
    /// </summary>
    public string[]? FeatureNames => _featureNames;

    /// <summary>
    /// Gets whether this vectorizer has been fitted.
    /// </summary>
    public bool IsFitted => _vocabulary is not null;

    /// <summary>
    /// Creates a new instance of <see cref="CountVectorizer{T}"/>.
    /// </summary>
    /// <param name="minDf">Minimum document frequency (absolute count). Defaults to 1.</param>
    /// <param name="maxDf">Maximum document frequency (proportion 0-1). Defaults to 1.0.</param>
    /// <param name="maxFeatures">Maximum vocabulary size. Null for unlimited.</param>
    /// <param name="nGramRange">N-gram range (min, max). Defaults to (1, 1) for unigrams.</param>
    /// <param name="lowercase">Convert all text to lowercase. Defaults to true.</param>
    /// <param name="binary">If true, use binary counts (1 if present, 0 if absent). Defaults to false.</param>
    /// <param name="tokenizer">Custom tokenizer function. Null for default whitespace/punctuation tokenizer.</param>
    /// <param name="stopWords">Words to exclude from vocabulary. Null for no filtering.</param>
    public CountVectorizer(
        int minDf = 1,
        double maxDf = 1.0,
        int? maxFeatures = null,
        (int Min, int Max)? nGramRange = null,
        bool lowercase = true,
        bool binary = false,
        Func<string, IEnumerable<string>>? tokenizer = null,
        HashSet<string>? stopWords = null)
    {
        if (minDf < 1)
        {
            throw new ArgumentException("Minimum document frequency must be at least 1.", nameof(minDf));
        }

        if (maxDf < 0 || maxDf > 1)
        {
            throw new ArgumentException("Maximum document frequency must be between 0 and 1.", nameof(maxDf));
        }

        _minDf = minDf;
        _maxDf = maxDf;
        _maxFeatures = maxFeatures;
        _nGramRange = nGramRange ?? (1, 1);
        _lowercase = lowercase;
        _binary = binary;
        _tokenizer = tokenizer;
        _stopWords = stopWords;
    }

    /// <summary>
    /// Fits the vectorizer to the corpus.
    /// </summary>
    /// <param name="documents">The text documents to learn vocabulary from.</param>
    public void Fit(IEnumerable<string> documents)
    {
        var docList = documents.ToList();
        int nDocs = docList.Count;

        // Count document frequency for each token
        var dfCounts = new Dictionary<string, int>();
        var allTokenCounts = new Dictionary<string, int>();

        foreach (string doc in docList)
        {
            var tokens = Tokenize(doc);
            var ngrams = GenerateNGrams(tokens);
            var uniqueNGrams = new HashSet<string>(ngrams);

            foreach (string ngram in uniqueNGrams)
            {
                dfCounts.TryGetValue(ngram, out int df);
                dfCounts[ngram] = df + 1;
            }

            foreach (string ngram in ngrams)
            {
                allTokenCounts.TryGetValue(ngram, out int count);
                allTokenCounts[ngram] = count + 1;
            }
        }

        // Filter by document frequency
        double maxDfCount = _maxDf * nDocs;
        var filteredTokens = dfCounts
            .Where(kvp => kvp.Value >= _minDf && kvp.Value <= maxDfCount)
            .Select(kvp => kvp.Key)
            .ToList();

        // Sort by total count (descending) for consistent ordering
        filteredTokens = filteredTokens
            .OrderByDescending(t => allTokenCounts[t])
            .ThenBy(t => t)
            .ToList();

        // Apply max features limit
        if (_maxFeatures.HasValue && filteredTokens.Count > _maxFeatures.Value)
        {
            filteredTokens = filteredTokens.Take(_maxFeatures.Value).ToList();
        }

        // Sort alphabetically for final vocabulary
        filteredTokens = filteredTokens.OrderBy(t => t).ToList();

        // Build vocabulary
        _vocabulary = new Dictionary<string, int>();
        for (int i = 0; i < filteredTokens.Count; i++)
        {
            _vocabulary[filteredTokens[i]] = i;
        }

        _featureNames = filteredTokens.ToArray();
    }

    /// <summary>
    /// Transforms documents to count vectors.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix where each row is a document and each column is a token count.</returns>
    public Matrix<T> Transform(IEnumerable<string> documents)
    {
        if (_vocabulary is null)
        {
            throw new InvalidOperationException("CountVectorizer has not been fitted.");
        }

        var docList = documents.ToList();
        int nDocs = docList.Count;
        int nFeatures = _vocabulary.Count;
        var result = new T[nDocs, nFeatures];

        for (int i = 0; i < nDocs; i++)
        {
            var tokens = Tokenize(docList[i]);
            var ngrams = GenerateNGrams(tokens);

            foreach (string ngram in ngrams)
            {
                if (_vocabulary.TryGetValue(ngram, out int idx))
                {
                    if (_binary)
                    {
                        result[i, idx] = NumOps<T>.One;
                    }
                    else
                    {
                        double current = NumOps<T>.ToDouble(result[i, idx]);
                        result[i, idx] = NumOps<T>.FromDouble(current + 1);
                    }
                }
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Fits the vectorizer and transforms the documents.
    /// </summary>
    /// <param name="documents">The documents to fit and transform.</param>
    /// <returns>Matrix of token counts.</returns>
    public Matrix<T> FitTransform(IEnumerable<string> documents)
    {
        var docList = documents.ToList();
        Fit(docList);
        return Transform(docList);
    }

    private IEnumerable<string> Tokenize(string text)
    {
        if (_tokenizer is not null)
        {
            var tokens = _tokenizer(text);
            if (_lowercase)
            {
                tokens = tokens.Select(t => t.ToLowerInvariant());
            }
            if (_stopWords is not null)
            {
                tokens = tokens.Where(t => !_stopWords.Contains(t));
            }
            return tokens;
        }

        // Default tokenizer: split on whitespace and punctuation
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
                {
                    tokenList.Add(token);
                }
                currentToken.Clear();
            }
        }

        if (currentToken.Length > 0)
        {
            string token = currentToken.ToString();
            if (_stopWords is null || !_stopWords.Contains(token))
            {
                tokenList.Add(token);
            }
        }

        return tokenList;
    }

    private IEnumerable<string> GenerateNGrams(IEnumerable<string> tokens)
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
    /// Gets the feature names (vocabulary terms).
    /// </summary>
    public string[] GetFeatureNamesOut()
    {
        return _featureNames ?? Array.Empty<string>();
    }

    /// <summary>
    /// Common English stop words.
    /// </summary>
    public static HashSet<string> EnglishStopWords => new HashSet<string>
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "the", "this", "but", "they",
        "have", "had", "what", "when", "where", "who", "which", "why", "how"
    };
}

/// <summary>
/// Helper class for numeric operations in text vectorizers.
/// </summary>
internal static class NumOps<T>
{
    private static readonly INumericOperations<T> _ops = MathHelper.GetNumericOperations<T>();

    public static T Zero => _ops.Zero;
    public static T One => _ops.One;

    public static double ToDouble(T value) => _ops.ToDouble(value);
    public static T FromDouble(double value) => _ops.FromDouble(value);
}
