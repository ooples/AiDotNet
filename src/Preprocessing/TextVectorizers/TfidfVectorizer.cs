using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;

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
public class TfidfVectorizer<T> : TextVectorizerBase<T>
{
    private readonly TfidfNorm _norm;
    private readonly bool _useIdf;
    private readonly bool _smoothIdf;
    private readonly bool _sublinearTf;

    // Fitted parameters specific to TF-IDF
    private double[]? _idfWeights;

    /// <summary>
    /// Gets the IDF weights for each term.
    /// </summary>
    /// <remarks>
    /// <para>
    /// IDF (Inverse Document Frequency) weights indicate how rare/important each term is.
    /// Higher values mean the term appears in fewer documents and is more discriminative.
    /// </para>
    /// <para><b>For Beginners:</b> Terms that appear in many documents get low IDF weights
    /// (they're common and not very distinctive), while terms in few documents get high weights
    /// (they're rare and help distinguish documents).
    /// </para>
    /// </remarks>
    public double[]? IdfWeights => _idfWeights;

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
    /// <param name="tokenizer">Custom tokenizer function. Null for default word-level tokenization.</param>
    /// <param name="stopWords">Words to exclude. Null for no filtering.</param>
    /// <param name="advancedTokenizer">
    /// Optional ITokenizer for subword tokenization (BPE, WordPiece, SentencePiece).
    /// When provided, takes precedence over the simple tokenizer function.
    /// </param>
    /// <example>
    /// <code>
    /// // Basic usage with default tokenization
    /// var tfidf = new TfidfVectorizer&lt;double&gt;();
    /// var matrix = tfidf.FitTransform(documents);
    ///
    /// // Using BERT tokenization for consistency with neural network models
    /// var tokenizer = AutoTokenizer.FromPretrained("bert-base-uncased");
    /// var tfidf = new TfidfVectorizer&lt;double&gt;(advancedTokenizer: tokenizer);
    /// var matrix = tfidf.FitTransform(documents);
    /// </code>
    /// </example>
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
        HashSet<string>? stopWords = null,
        ITokenizer? advancedTokenizer = null)
        : base(minDf, maxDf, maxFeatures, nGramRange, lowercase, tokenizer, stopWords, advancedTokenizer)
    {
        _norm = norm;
        _useIdf = useIdf;
        _smoothIdf = smoothIdf;
        _sublinearTf = sublinearTf;
    }

    /// <inheritdoc/>
    public override bool IsFitted => _vocabulary is not null && _idfWeights is not null;

    /// <summary>
    /// Fits the vectorizer to the corpus.
    /// </summary>
    /// <param name="documents">The text documents to learn vocabulary and IDF from.</param>
    public override void Fit(IEnumerable<string> documents)
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

        // Build vocabulary using base class method
        BuildVocabulary(dfCounts, allTokenCounts);

        // Compute IDF weights
        _idfWeights = new double[_featureNames!.Length];
        for (int i = 0; i < _featureNames.Length; i++)
        {
            string term = _featureNames[i];
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
    public override Matrix<T> Transform(IEnumerable<string> documents)
    {
        if (_vocabulary is null || _idfWeights is null)
        {
            throw new InvalidOperationException("TfidfVectorizer has not been fitted. Call Fit() or FitTransform() first.");
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
                output[i, j] = NumOps.FromDouble(result[i, j]);
            }
        }

        return new Matrix<T>(output);
    }
}
