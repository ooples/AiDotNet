using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

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
public class CountVectorizer<T> : TextVectorizerBase<T>
{
    private readonly bool _binary;

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
        : base(minDf, maxDf, maxFeatures, nGramRange, lowercase, tokenizer, stopWords)
    {
        _binary = binary;
    }

    /// <summary>
    /// Fits the vectorizer to the corpus.
    /// </summary>
    /// <param name="documents">The text documents to learn vocabulary from.</param>
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
    }

    /// <summary>
    /// Transforms documents to count vectors.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix where each row is a document and each column is a token count.</returns>
    public override Matrix<T> Transform(IEnumerable<string> documents)
    {
        if (_vocabulary is null)
        {
            throw new InvalidOperationException("CountVectorizer has not been fitted. Call Fit() or FitTransform() first.");
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
                        result[i, idx] = NumOps.One;
                    }
                    else
                    {
                        double current = NumOps.ToDouble(result[i, idx]);
                        result[i, idx] = NumOps.FromDouble(current + 1);
                    }
                }
            }
        }

        return new Matrix<T>(result);
    }
}
