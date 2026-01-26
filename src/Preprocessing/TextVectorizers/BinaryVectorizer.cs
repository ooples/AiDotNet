using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Converts text documents to binary feature vectors (presence/absence encoding).
/// </summary>
/// <remarks>
/// <para>
/// BinaryVectorizer creates a simple presence/absence encoding where each feature
/// is 1 if the term appears in the document and 0 otherwise. Unlike CountVectorizer
/// with binary=true, this vectorizer is optimized specifically for binary encoding.
/// </para>
/// <para><b>For Beginners:</b> Binary encoding is the simplest text representation:
/// - Each word is either present (1) or absent (0) in a document
/// - Word frequency is ignored (appearing 10 times = appearing once)
/// - Fast and memory-efficient
/// - Works well when word presence matters more than frequency
/// - Common in document classification and spam detection
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class BinaryVectorizer<T> : TextVectorizerBase<T>
{
    /// <summary>
    /// Creates a new instance of <see cref="BinaryVectorizer{T}"/>.
    /// </summary>
    /// <param name="minDf">Minimum document frequency (absolute count). Defaults to 1.</param>
    /// <param name="maxDf">Maximum document frequency (proportion 0-1). Defaults to 1.0.</param>
    /// <param name="maxFeatures">Maximum vocabulary size. Null for unlimited.</param>
    /// <param name="nGramRange">N-gram range (min, max). Defaults to (1, 1) for unigrams.</param>
    /// <param name="lowercase">Convert all text to lowercase. Defaults to true.</param>
    /// <param name="tokenizer">Custom tokenizer function. Null for default.</param>
    /// <param name="stopWords">Words to exclude. Null for no filtering.</param>
    /// <param name="advancedTokenizer">Optional ITokenizer for subword tokenization.</param>
    public BinaryVectorizer(
        int minDf = 1,
        double maxDf = 1.0,
        int? maxFeatures = null,
        (int Min, int Max)? nGramRange = null,
        bool lowercase = true,
        Func<string, IEnumerable<string>>? tokenizer = null,
        HashSet<string>? stopWords = null,
        ITokenizer? advancedTokenizer = null)
        : base(minDf, maxDf, maxFeatures, nGramRange, lowercase, tokenizer, stopWords, advancedTokenizer)
    {
    }

    /// <summary>
    /// Fits the vectorizer to the corpus.
    /// </summary>
    /// <param name="documents">The text documents to learn vocabulary from.</param>
    public override void Fit(IEnumerable<string> documents)
    {
        var docList = documents.ToList();
        _nDocs = docList.Count;

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

        BuildVocabulary(dfCounts, allTokenCounts);
    }

    /// <summary>
    /// Transforms documents to binary vectors.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix where each row is a document's binary feature vector.</returns>
    public override Matrix<T> Transform(IEnumerable<string> documents)
    {
        if (_vocabulary is null)
        {
            throw new InvalidOperationException("BinaryVectorizer has not been fitted. Call Fit() or FitTransform() first.");
        }

        var docList = documents.ToList();
        int nDocs = docList.Count;
        int nFeatures = _vocabulary.Count;
        var result = new T[nDocs, nFeatures];

        for (int i = 0; i < nDocs; i++)
        {
            var tokens = Tokenize(docList[i]);
            var ngrams = GenerateNGrams(tokens);
            var uniqueNGrams = new HashSet<string>(ngrams);

            foreach (string ngram in uniqueNGrams)
            {
                if (_vocabulary.TryGetValue(ngram, out int idx))
                {
                    result[i, idx] = NumOps.One;
                }
            }
        }

        return new Matrix<T>(result);
    }
}
