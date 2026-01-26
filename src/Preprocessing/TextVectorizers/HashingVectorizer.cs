using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Converts text documents to a fixed-size hash-based feature matrix.
/// </summary>
/// <remarks>
/// <para>
/// HashingVectorizer uses the hashing trick to map tokens to a fixed number of features.
/// Unlike CountVectorizer and TfidfVectorizer, it doesn't require storing a vocabulary,
/// making it memory-efficient for very large datasets or streaming applications.
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
public class HashingVectorizer<T> : TextVectorizerBase<T>
{
    private readonly int _nFeatures;
    private readonly bool _binary;
    private readonly bool _alternateSign;
    private readonly HashingNorm _norm;

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
    /// <param name="advancedTokenizer">
    /// Optional ITokenizer for subword tokenization (BPE, WordPiece, SentencePiece).
    /// When provided, takes precedence over the simple tokenizer function.
    /// </param>
    public HashingVectorizer(
        int nFeatures = 1048576,
        (int Min, int Max)? nGramRange = null,
        bool lowercase = true,
        bool binary = false,
        bool alternateSign = true,
        HashingNorm norm = HashingNorm.L2,
        Func<string, IEnumerable<string>>? tokenizer = null,
        HashSet<string>? stopWords = null,
        ITokenizer? advancedTokenizer = null)
        : base(minDf: 1, maxDf: 1.0, maxFeatures: null, nGramRange, lowercase, tokenizer, stopWords, advancedTokenizer)
    {
        if (nFeatures < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeatures));

        _nFeatures = nFeatures;
        _binary = binary;
        _alternateSign = alternateSign;
        _norm = norm;

        // Don't pre-generate feature names - do it lazily if needed
        // With default nFeatures = 1048576, eagerly allocating would consume tens of megabytes
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Feature names are generated lazily on first call to avoid memory allocation
    /// when feature names are not needed. With the default 1 million features,
    /// this saves significant memory.
    /// </remarks>
    public override string[] GetFeatureNamesOut()
    {
        // Generate synthetic feature names on demand
        _featureNames ??= Enumerable.Range(0, _nFeatures).Select(i => $"hash_{i}").ToArray();
        return _featureNames;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// HashingVectorizer is always fitted - it doesn't learn from data.
    /// </remarks>
    public override bool IsFitted => true;

    /// <inheritdoc/>
    public override int FeatureCount => _nFeatures;

    /// <summary>
    /// Fits the vectorizer (no-op for HashingVectorizer - fitting is not required).
    /// </summary>
    /// <param name="documents">The documents (ignored).</param>
    public override void Fit(IEnumerable<string> documents)
    {
        // HashingVectorizer doesn't need fitting - this is a no-op for API compatibility
        _nDocs = documents.Count();
    }

    /// <summary>
    /// Transforms documents to hash-based feature vectors.
    /// No fitting required - this method can be called directly.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix where each row is a document's hashed feature vector.</returns>
    public override Matrix<T> Transform(IEnumerable<string> documents)
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
                output[i, j] = NumOps.FromDouble(result[i, j]);
            }
        }

        return new Matrix<T>(output);
    }

    /// <inheritdoc/>
    public override Matrix<T> FitTransform(IEnumerable<string> documents)
    {
        // HashingVectorizer doesn't need fitting, just transform
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
}
