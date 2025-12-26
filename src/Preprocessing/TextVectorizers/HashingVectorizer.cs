using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Converts text documents to a fixed-size hash-based feature matrix.
/// </summary>
/// <remarks>
/// <para>
/// HashingVectorizer uses the hashing trick to map tokens to a fixed number of features.
/// Unlike CountVectorizer, it doesn't require storing a vocabulary, making it memory-efficient
/// for very large datasets or streaming applications.
/// </para>
/// <para>
/// The trade-off is that hash collisions can occur (different tokens map to same feature),
/// and you cannot retrieve original tokens from the hash indices.
/// </para>
/// <para><b>For Beginners:</b> HashingVectorizer is like CountVectorizer but:
/// - Doesn't need to see all data first (no vocabulary to learn)
/// - Uses fixed memory regardless of vocabulary size
/// - Slight accuracy loss from hash collisions
/// - Great for very large or streaming text data
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class HashingVectorizer<T>
{
    private readonly int _nFeatures;
    private readonly (int Min, int Max) _nGramRange;
    private readonly bool _lowercase;
    private readonly bool _binary;
    private readonly bool _alternateSign;
    private readonly HashingNorm _norm;
    private readonly Func<string, IEnumerable<string>>? _tokenizer;
    private readonly HashSet<string>? _stopWords;

    /// <summary>
    /// Gets the number of output features (hash buckets).
    /// </summary>
    public int NFeatures => _nFeatures;

    /// <summary>
    /// Creates a new instance of <see cref="HashingVectorizer{T}"/>.
    /// </summary>
    /// <param name="nFeatures">Number of features (hash buckets). Defaults to 2^20 (1048576).</param>
    /// <param name="nGramRange">N-gram range (min, max). Defaults to (1, 1) for unigrams.</param>
    /// <param name="lowercase">Convert all text to lowercase. Defaults to true.</param>
    /// <param name="binary">If true, use binary counts (1 if present, 0 if absent). Defaults to false.</param>
    /// <param name="alternateSign">Alternate sign of hash to reduce collision impact. Defaults to true.</param>
    /// <param name="norm">Normalization method for output vectors. Defaults to L2.</param>
    /// <param name="tokenizer">Custom tokenizer function. Null for default.</param>
    /// <param name="stopWords">Words to exclude. Null for no filtering.</param>
    public HashingVectorizer(
        int nFeatures = 1048576,
        (int Min, int Max)? nGramRange = null,
        bool lowercase = true,
        bool binary = false,
        bool alternateSign = true,
        HashingNorm norm = HashingNorm.L2,
        Func<string, IEnumerable<string>>? tokenizer = null,
        HashSet<string>? stopWords = null)
    {
        if (nFeatures < 1)
        {
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeatures));
        }

        _nFeatures = nFeatures;
        _nGramRange = nGramRange ?? (1, 1);
        _lowercase = lowercase;
        _binary = binary;
        _alternateSign = alternateSign;
        _norm = norm;
        _tokenizer = tokenizer;
        _stopWords = stopWords;
    }

    /// <summary>
    /// Transforms documents to hash-based feature vectors.
    /// No fitting required - this method can be called directly.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix where each row is a document's hashed feature vector.</returns>
    public Matrix<T> Transform(IEnumerable<string> documents)
    {
        var docList = documents.ToList();
        int nDocs = docList.Count;
        var result = new double[nDocs, _nFeatures];

        for (int i = 0; i < nDocs; i++)
        {
            var tokens = Tokenize(docList[i]);
            var ngrams = GenerateNGrams(tokens);

            foreach (string ngram in ngrams)
            {
                int hash = ComputeHash(ngram);
                int idx = Math.Abs(hash) % _nFeatures;

                if (_binary)
                {
                    result[i, idx] = 1;
                }
                else if (_alternateSign)
                {
                    // Use sign of hash to reduce collision impact
                    double sign = hash >= 0 ? 1.0 : -1.0;
                    result[i, idx] += sign;
                }
                else
                {
                    result[i, idx] += 1;
                }
            }

            // Normalize
            if (_norm != HashingNorm.None)
            {
                double normValue = 0;

                if (_norm == HashingNorm.L1)
                {
                    for (int j = 0; j < _nFeatures; j++)
                    {
                        normValue += Math.Abs(result[i, j]);
                    }
                }
                else if (_norm == HashingNorm.L2)
                {
                    for (int j = 0; j < _nFeatures; j++)
                    {
                        normValue += result[i, j] * result[i, j];
                    }
                    normValue = Math.Sqrt(normValue);
                }

                if (normValue > 1e-10)
                {
                    for (int j = 0; j < _nFeatures; j++)
                    {
                        result[i, j] /= normValue;
                    }
                }
            }
        }

        // Convert to output type
        var output = new T[nDocs, _nFeatures];
        for (int i = 0; i < nDocs; i++)
        {
            for (int j = 0; j < _nFeatures; j++)
            {
                output[i, j] = NumOps<T>.FromDouble(result[i, j]);
            }
        }

        return new Matrix<T>(output);
    }

    /// <summary>
    /// Fits and transforms (for API compatibility - no actual fitting needed).
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix of hashed feature values.</returns>
    public Matrix<T> FitTransform(IEnumerable<string> documents)
    {
        return Transform(documents);
    }

    private int ComputeHash(string input)
    {
        // Use a simple but effective hash function
        // MurmurHash-inspired implementation for good distribution
        unchecked
        {
            const int seed = 0x5bd1e995;
            int hash = seed ^ input.Length;

            foreach (char c in input)
            {
                int k = c;
                k *= 0x5bd1e995;
                k ^= k >> 24;
                k *= 0x5bd1e995;

                hash *= 0x5bd1e995;
                hash ^= k;
            }

            hash ^= hash >> 13;
            hash *= 0x5bd1e995;
            hash ^= hash >> 15;

            return hash;
        }
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
    /// Gets feature names (hash bucket indices).
    /// </summary>
    public string[] GetFeatureNamesOut()
    {
        return Enumerable.Range(0, _nFeatures).Select(i => $"hash_{i}").ToArray();
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
/// Specifies the normalization method for HashingVectorizer.
/// </summary>
public enum HashingNorm
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
