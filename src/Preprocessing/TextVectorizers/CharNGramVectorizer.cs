using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Converts text documents to character-level n-gram feature vectors.
/// </summary>
/// <remarks>
/// <para>
/// CharNGramVectorizer creates features from character sequences rather than words.
/// This is particularly useful for:
/// - Handling misspellings and typos
/// - Languages without clear word boundaries
/// - Capturing subword patterns
/// - Short text classification (tweets, SMS)
/// </para>
/// <para><b>For Beginners:</b> Character n-grams capture patterns at the letter level:
/// - "hello" with (2,3) n-grams produces: "he", "el", "ll", "lo", "hel", "ell", "llo"
/// - Robust to spelling mistakes ("color" and "colour" share many character n-grams)
/// - Works well for author identification and language detection
/// - Can capture morphological patterns (prefixes, suffixes)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class CharNGramVectorizer<T> : TextVectorizerBase<T>
{
    private readonly (int Min, int Max) _charNGramRange;
    private readonly bool _wordBoundaries;
    private readonly CharNGramNorm _norm;

    /// <summary>
    /// Creates a new instance of <see cref="CharNGramVectorizer{T}"/>.
    /// </summary>
    /// <param name="charNGramRange">Character n-gram range (min, max). Defaults to (2, 4).</param>
    /// <param name="wordBoundaries">Add word boundary markers. Defaults to true.</param>
    /// <param name="minDf">Minimum document frequency (absolute count). Defaults to 1.</param>
    /// <param name="maxDf">Maximum document frequency (proportion 0-1). Defaults to 1.0.</param>
    /// <param name="maxFeatures">Maximum vocabulary size. Null for unlimited.</param>
    /// <param name="lowercase">Convert all text to lowercase. Defaults to true.</param>
    /// <param name="norm">Normalization method for output vectors. Defaults to L2.</param>
    /// <param name="stopWords">Words to exclude before generating character n-grams. Null for no filtering.</param>
    /// <param name="advancedTokenizer">Optional ITokenizer for subword tokenization.</param>
    public CharNGramVectorizer(
        (int Min, int Max)? charNGramRange = null,
        bool wordBoundaries = true,
        int minDf = 1,
        double maxDf = 1.0,
        int? maxFeatures = null,
        bool lowercase = true,
        CharNGramNorm norm = CharNGramNorm.L2,
        HashSet<string>? stopWords = null,
        ITokenizer? advancedTokenizer = null)
        : base(minDf, maxDf, maxFeatures, (1, 1), lowercase, null, stopWords, advancedTokenizer)
    {
        _charNGramRange = charNGramRange ?? (2, 4);
        _wordBoundaries = wordBoundaries;
        _norm = norm;

        if (_charNGramRange.Min < 1)
            throw new ArgumentException("Minimum n-gram length must be at least 1.", nameof(charNGramRange));
        if (_charNGramRange.Max < _charNGramRange.Min)
            throw new ArgumentException("Maximum n-gram length must be >= minimum.", nameof(charNGramRange));
    }

    /// <summary>
    /// Generates character n-grams from text.
    /// </summary>
    private IEnumerable<string> GenerateCharNGrams(string text)
    {
        string processedText = _lowercase ? text.ToLowerInvariant() : text;

        // Optionally add word boundary markers
        if (_wordBoundaries)
        {
            var words = processedText.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            if (_stopWords != null)
            {
                words = words.Where(w => !_stopWords.Contains(w)).ToArray();
            }
            processedText = string.Join(" ", words.Select(w => $" {w} "));
        }

        var ngrams = new List<string>();

        for (int n = _charNGramRange.Min; n <= _charNGramRange.Max; n++)
        {
            for (int i = 0; i <= processedText.Length - n; i++)
            {
                ngrams.Add(processedText.Substring(i, n));
            }
        }

        return ngrams;
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
        var allNGramCounts = new Dictionary<string, int>();

        foreach (string doc in docList)
        {
            var ngrams = GenerateCharNGrams(doc).ToList();
            var uniqueNGrams = new HashSet<string>(ngrams);

            foreach (string ngram in uniqueNGrams)
            {
                dfCounts.TryGetValue(ngram, out int df);
                dfCounts[ngram] = df + 1;
            }

            foreach (string ngram in ngrams)
            {
                allNGramCounts.TryGetValue(ngram, out int count);
                allNGramCounts[ngram] = count + 1;
            }
        }

        BuildVocabulary(dfCounts, allNGramCounts);
    }

    /// <summary>
    /// Transforms documents to character n-gram vectors.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix where each row is a document's character n-gram vector.</returns>
    public override Matrix<T> Transform(IEnumerable<string> documents)
    {
        if (_vocabulary is null)
        {
            throw new InvalidOperationException("CharNGramVectorizer has not been fitted. Call Fit() or FitTransform() first.");
        }

        var docList = documents.ToList();
        int nDocs = docList.Count;
        int nFeatures = _vocabulary.Count;
        var result = new double[nDocs, nFeatures];

        for (int i = 0; i < nDocs; i++)
        {
            var ngrams = GenerateCharNGrams(docList[i]);

            foreach (string ngram in ngrams)
            {
                if (_vocabulary.TryGetValue(ngram, out int idx))
                {
                    result[i, idx] += 1;
                }
            }

            // Normalize
            if (_norm != CharNGramNorm.None)
            {
                double normValue = 0;

                if (_norm == CharNGramNorm.L1)
                {
                    for (int j = 0; j < nFeatures; j++)
                    {
                        normValue += Math.Abs(result[i, j]);
                    }
                }
                else if (_norm == CharNGramNorm.L2)
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
