using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Converts text documents to BM25-weighted feature vectors.
/// </summary>
/// <remarks>
/// <para>
/// BM25 (Best Matching 25) is an advanced ranking function used by search engines like
/// Elasticsearch and Lucene. It improves upon TF-IDF by adding document length normalization
/// and term frequency saturation.
/// </para>
/// <para>
/// The BM25 formula is:
/// <code>
/// score(D,Q) = IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))
/// </code>
/// Where:
/// - f(qi,D) = term frequency of qi in document D
/// - |D| = document length
/// - avgdl = average document length in the corpus
/// - k1 = term frequency saturation parameter (typically 1.2-2.0)
/// - b = document length normalization parameter (typically 0.75)
/// </para>
/// <para><b>For Beginners:</b> BM25 is like TF-IDF but smarter:
/// - Long documents don't unfairly dominate (length normalization)
/// - Repeating a word 100 times doesn't score 100x better than once (saturation)
/// - Used by Google, Elasticsearch, and most modern search engines
/// - Generally performs better than TF-IDF for search and retrieval tasks
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class BM25Vectorizer<T> : TextVectorizerBase<T>
{
    private readonly double _k1;
    private readonly double _b;
    private readonly double _delta;
    private readonly BM25Norm _norm;

    private double[]? _idfWeights;
    private double _avgDocLength;
    private int[]? _docLengths;

    /// <summary>
    /// Gets the IDF weights for each term.
    /// </summary>
    public double[]? IdfWeights => _idfWeights;

    /// <summary>
    /// Gets the average document length in the corpus.
    /// </summary>
    public double AverageDocumentLength => _avgDocLength;

    /// <summary>
    /// Creates a new instance of <see cref="BM25Vectorizer{T}"/>.
    /// </summary>
    /// <param name="k1">Term frequency saturation parameter. Higher values mean term frequency has more impact. Default: 1.5</param>
    /// <param name="b">Document length normalization parameter (0-1). 0 = no normalization, 1 = full normalization. Default: 0.75</param>
    /// <param name="delta">BM25+ modification parameter. Adds a small constant to prevent zero scores. Default: 0 (standard BM25)</param>
    /// <param name="minDf">Minimum document frequency (absolute count). Defaults to 1.</param>
    /// <param name="maxDf">Maximum document frequency (proportion 0-1). Defaults to 1.0.</param>
    /// <param name="maxFeatures">Maximum vocabulary size. Null for unlimited.</param>
    /// <param name="nGramRange">N-gram range (min, max). Defaults to (1, 1) for unigrams.</param>
    /// <param name="lowercase">Convert all text to lowercase. Defaults to true.</param>
    /// <param name="norm">Normalization method for output vectors. Defaults to None.</param>
    /// <param name="tokenizer">Custom tokenizer function. Null for default.</param>
    /// <param name="stopWords">Words to exclude. Null for no filtering.</param>
    /// <param name="advancedTokenizer">
    /// Optional ITokenizer for subword tokenization (BPE, WordPiece, SentencePiece).
    /// When provided, takes precedence over the simple tokenizer function.
    /// </param>
    public BM25Vectorizer(
        double k1 = 1.5,
        double b = 0.75,
        double delta = 0,
        int minDf = 1,
        double maxDf = 1.0,
        int? maxFeatures = null,
        (int Min, int Max)? nGramRange = null,
        bool lowercase = true,
        BM25Norm norm = BM25Norm.None,
        Func<string, IEnumerable<string>>? tokenizer = null,
        HashSet<string>? stopWords = null,
        ITokenizer? advancedTokenizer = null)
        : base(minDf, maxDf, maxFeatures, nGramRange, lowercase, tokenizer, stopWords, advancedTokenizer)
    {
        if (k1 < 0)
            throw new ArgumentException("k1 must be non-negative.", nameof(k1));
        if (b < 0 || b > 1)
            throw new ArgumentException("b must be between 0 and 1.", nameof(b));

        _k1 = k1;
        _b = b;
        _delta = delta;
        _norm = norm;
    }

    /// <inheritdoc/>
    public override bool IsFitted => _vocabulary is not null && _idfWeights is not null;

    /// <summary>
    /// Fits the vectorizer to the corpus.
    /// </summary>
    /// <param name="documents">The text documents to learn vocabulary and statistics from.</param>
    public override void Fit(IEnumerable<string> documents)
    {
        var docList = documents.ToList();
        _nDocs = docList.Count;

        // Count document frequency for each token and track document lengths
        var dfCounts = new Dictionary<string, int>();
        var allTokenCounts = new Dictionary<string, int>();
        _docLengths = new int[_nDocs];
        int totalLength = 0;

        for (int i = 0; i < docList.Count; i++)
        {
            var tokens = Tokenize(docList[i]);
            var ngrams = GenerateNGrams(tokens).ToList();
            var uniqueNGrams = new HashSet<string>(ngrams);

            _docLengths[i] = ngrams.Count;
            totalLength += ngrams.Count;

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

        _avgDocLength = _nDocs > 0 ? (double)totalLength / _nDocs : 0;

        // Build vocabulary using base class method
        BuildVocabulary(dfCounts, allTokenCounts);

        // Compute IDF weights using BM25 formula
        _idfWeights = new double[_featureNames!.Length];
        for (int i = 0; i < _featureNames.Length; i++)
        {
            string term = _featureNames[i];
            int df = dfCounts[term];

            // BM25 IDF: log((N - df + 0.5) / (df + 0.5))
            // Add 1 to avoid negative IDF for very common terms
            _idfWeights[i] = Math.Log(1 + (_nDocs - df + 0.5) / (df + 0.5));
        }
    }

    /// <summary>
    /// Transforms documents to BM25-weighted vectors.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix where each row is a document's BM25 vector.</returns>
    public override Matrix<T> Transform(IEnumerable<string> documents)
    {
        if (_vocabulary is null || _idfWeights is null)
        {
            throw new InvalidOperationException("BM25Vectorizer has not been fitted. Call Fit() or FitTransform() first.");
        }

        var docList = documents.ToList();
        int nDocs = docList.Count;
        int nFeatures = _vocabulary.Count;
        var result = new double[nDocs, nFeatures];

        for (int i = 0; i < nDocs; i++)
        {
            var tokens = Tokenize(docList[i]);
            var ngrams = GenerateNGrams(tokens).ToList();
            int docLength = ngrams.Count;

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

            // Compute BM25 scores
            foreach (var kvp in tfCounts)
            {
                int idx = kvp.Key;
                double tf = kvp.Value;

                // BM25 term weight
                double numerator = tf * (_k1 + 1);
                double denominator = tf + _k1 * (1 - _b + _b * docLength / _avgDocLength);
                double termWeight = numerator / denominator + _delta;

                // BM25 score = IDF * term weight
                result[i, idx] = _idfWeights[idx] * termWeight;
            }

            // Normalize if requested
            if (_norm != BM25Norm.None)
            {
                double normValue = 0;

                if (_norm == BM25Norm.L1)
                {
                    for (int j = 0; j < nFeatures; j++)
                    {
                        normValue += Math.Abs(result[i, j]);
                    }
                }
                else if (_norm == BM25Norm.L2)
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
                output[i, j] = NumOps.FromDouble(result[i, j]);
            }
        }

        return new Matrix<T>(output);
    }
}
