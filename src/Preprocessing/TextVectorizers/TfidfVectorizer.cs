using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Converts a collection of text documents to a TF-IDF weighted matrix.
/// </summary>
/// <remarks>
/// <para>
/// TfidfVectorizer converts text to TF-IDF (Term Frequency-Inverse Document Frequency)
/// representation, which weights terms by their importance in a document relative
/// to the entire corpus.
/// </para>
/// <para>
/// TF-IDF = TF × IDF where:
/// - TF (Term Frequency) = count of term in document / total terms in document
/// - IDF (Inverse Document Frequency) = log(total documents / documents containing term)
/// </para>
/// <para><b>For Beginners:</b> TF-IDF makes rare but meaningful words more important:
/// - Common words like "the" appear everywhere, so they get low weight
/// - Rare words that only appear in specific documents get high weight
/// - This helps distinguish documents by their unique content
///
/// Example: "machine learning" in a tech article is more meaningful
/// than "the" which appears in every document.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class TfidfVectorizer<T>
{
    private readonly int _minDf;
    private readonly double _maxDf;
    private readonly int? _maxFeatures;
    private readonly (int Min, int Max) _nGramRange;
    private readonly bool _lowercase;
    private readonly TfidfNorm _norm;
    private readonly bool _useIdf;
    private readonly bool _smoothIdf;
    private readonly bool _sublinearTf;
    private readonly Func<string, IEnumerable<string>>? _tokenizer;
    private readonly HashSet<string>? _stopWords;

    // Fitted parameters
    private Dictionary<string, int>? _vocabulary;
    private string[]? _featureNames;
    private double[]? _idfWeights;
    private int _nDocs;

    /// <summary>
    /// Gets the vocabulary (token to index mapping).
    /// </summary>
    public Dictionary<string, int>? Vocabulary => _vocabulary;

    /// <summary>
    /// Gets the feature names (tokens).
    /// </summary>
    public string[]? FeatureNames => _featureNames;

    /// <summary>
    /// Gets the IDF weights for each term.
    /// </summary>
    public double[]? IdfWeights => _idfWeights;

    /// <summary>
    /// Gets whether this vectorizer has been fitted.
    /// </summary>
    public bool IsFitted => _vocabulary is not null;

    /// <summary>
    /// Creates a new instance of <see cref="TfidfVectorizer{T}"/>.
    /// </summary>
    /// <param name="minDf">Minimum document frequency (absolute count). Defaults to 1.</param>
    /// <param name="maxDf">Maximum document frequency (proportion 0-1). Defaults to 1.0.</param>
    /// <param name="maxFeatures">Maximum vocabulary size. Null for unlimited.</param>
    /// <param name="nGramRange">N-gram range (min, max). Defaults to (1, 1) for unigrams.</param>
    /// <param name="lowercase">Convert all text to lowercase. Defaults to true.</param>
    /// <param name="norm">Normalization method for output vectors. Defaults to L2.</param>
    /// <param name="useIdf">Whether to use IDF weighting. Defaults to true.</param>
    /// <param name="smoothIdf">Smooth IDF by adding 1 to df. Defaults to true.</param>
    /// <param name="sublinearTf">Apply sublinear TF (1 + log(tf)). Defaults to false.</param>
    /// <param name="tokenizer">Custom tokenizer function. Null for default.</param>
    /// <param name="stopWords">Words to exclude. Null for no filtering.</param>
    public TfidfVectorizer(
        int minDf = 1,
        double maxDf = 1.0,
        int? maxFeatures = null,
        (int Min, int Max)? nGramRange = null,
        bool lowercase = true,
        TfidfNorm norm = TfidfNorm.L2,
        bool useIdf = true,
        bool smoothIdf = true,
        bool sublinearTf = false,
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
        _norm = norm;
        _useIdf = useIdf;
        _smoothIdf = smoothIdf;
        _sublinearTf = sublinearTf;
        _tokenizer = tokenizer;
        _stopWords = stopWords;
    }

    /// <summary>
    /// Fits the vectorizer to the corpus.
    /// </summary>
    /// <param name="documents">The text documents to learn vocabulary and IDF from.</param>
    public void Fit(IEnumerable<string> documents)
    {
        var docList = documents.ToList();
        _nDocs = docList.Count;

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
        double maxDfCount = _maxDf * _nDocs;
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

        // Compute IDF weights
        _idfWeights = new double[filteredTokens.Count];
        for (int i = 0; i < filteredTokens.Count; i++)
        {
            string term = filteredTokens[i];
            int df = dfCounts[term];

            if (_smoothIdf)
            {
                // Smooth IDF: log((n + 1) / (df + 1)) + 1
                _idfWeights[i] = Math.Log((double)(_nDocs + 1) / (df + 1)) + 1;
            }
            else
            {
                // Standard IDF: log(n / df) + 1
                _idfWeights[i] = Math.Log((double)_nDocs / df) + 1;
            }
        }
    }

    /// <summary>
    /// Transforms documents to TF-IDF weighted vectors.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix where each row is a document's TF-IDF vector.</returns>
    public Matrix<T> Transform(IEnumerable<string> documents)
    {
        if (_vocabulary is null || _idfWeights is null)
        {
            throw new InvalidOperationException("TfidfVectorizer has not been fitted.");
        }

        var docList = documents.ToList();
        int nDocs = docList.Count;
        int nFeatures = _vocabulary.Count;
        var result = new double[nDocs, nFeatures];

        for (int i = 0; i < nDocs; i++)
        {
            var tokens = Tokenize(docList[i]);
            var ngrams = GenerateNGrams(tokens).ToList();

            // Count term frequencies
            var tfCounts = new Dictionary<int, int>();
            foreach (string ngram in ngrams)
            {
                if (_vocabulary.TryGetValue(ngram, out int idx))
                {
                    tfCounts.TryGetValue(idx, out int count);
                    tfCounts[idx] = count + 1;
                }
            }

            // Apply TF weighting
            foreach (var kvp in tfCounts)
            {
                int idx = kvp.Key;
                double tf = kvp.Value;

                if (_sublinearTf && tf > 0)
                {
                    tf = 1 + Math.Log(tf);
                }

                // Apply IDF
                double tfidf = _useIdf ? tf * _idfWeights[idx] : tf;
                result[i, idx] = tfidf;
            }

            // Normalize
            if (_norm != TfidfNorm.None)
            {
                double normValue = 0;

                if (_norm == TfidfNorm.L1)
                {
                    for (int j = 0; j < nFeatures; j++)
                    {
                        normValue += Math.Abs(result[i, j]);
                    }
                }
                else if (_norm == TfidfNorm.L2)
                {
                    for (int j = 0; j < nFeatures; j++)
                    {
                        normValue += result[i, j] * result[i, j];
                    }
                    normValue = Math.Sqrt(normValue);
                }

                if (normValue > 1e-10)
                {
                    for (int j = 0; j < nFeatures; j++)
                    {
                        result[i, j] /= normValue;
                    }
                }
            }
        }

        // Convert to output type
        var output = new T[nDocs, nFeatures];
        for (int i = 0; i < nDocs; i++)
        {
            for (int j = 0; j < nFeatures; j++)
            {
                output[i, j] = NumOps<T>.FromDouble(result[i, j]);
            }
        }

        return new Matrix<T>(output);
    }

    /// <summary>
    /// Fits the vectorizer and transforms the documents.
    /// </summary>
    /// <param name="documents">The documents to fit and transform.</param>
    /// <returns>Matrix of TF-IDF values.</returns>
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
/// Specifies the normalization method for TF-IDF vectors.
/// </summary>
public enum TfidfNorm
{
    /// <summary>
    /// No normalization.
    /// </summary>
    None,

    /// <summary>
    /// L1 normalization (sum of absolute values = 1).
    /// </summary>
    L1,

    /// <summary>
    /// L2 normalization (Euclidean length = 1).
    /// </summary>
    L2
}
