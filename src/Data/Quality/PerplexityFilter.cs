namespace AiDotNet.Data.Quality;

/// <summary>
/// Filters documents based on perplexity scores from a simple n-gram language model.
/// </summary>
/// <remarks>
/// <para>
/// Perplexity measures how "surprised" a language model is by a document.
/// Very high perplexity indicates gibberish or foreign language text.
/// Very low perplexity may indicate boilerplate or repetitive content.
/// Commonly used in C4 and CCNet data cleaning pipelines.
/// </para>
/// </remarks>
public class PerplexityFilter
{
    private readonly PerplexityFilterOptions _options;
    private Dictionary<string, int> _ngramCounts;
    private Dictionary<string, int> _contextCounts;
    private int _vocabularySize;
    private bool _isTrained;

    public PerplexityFilter(PerplexityFilterOptions? options = null)
    {
        _options = options ?? new PerplexityFilterOptions();
        _ngramCounts = new Dictionary<string, int>();
        _contextCounts = new Dictionary<string, int>();
        _vocabularySize = 0;
        _isTrained = false;
    }

    /// <summary>
    /// Trains the n-gram language model on reference text.
    /// </summary>
    /// <param name="referenceDocuments">High-quality reference documents to build the model from.</param>
    public void Train(IReadOnlyList<string> referenceDocuments)
    {
        if (referenceDocuments.Count == 0)
            throw new ArgumentException("Reference documents cannot be empty.", nameof(referenceDocuments));

        _ngramCounts.Clear();
        _contextCounts.Clear();
        var uniqueTokens = new HashSet<string>();

        foreach (string doc in referenceDocuments)
        {
            string[] tokens = Tokenize(doc);
            foreach (string token in tokens)
                uniqueTokens.Add(token);

            for (int i = 0; i <= tokens.Length - _options.NGramOrder; i++)
            {
                string ngram = string.Join(" ", tokens, i, _options.NGramOrder);
                string context = string.Join(" ", tokens, i, _options.NGramOrder - 1);

                _ngramCounts[ngram] = _ngramCounts.GetValueOrDefault(ngram, 0) + 1;
                _contextCounts[context] = _contextCounts.GetValueOrDefault(context, 0) + 1;
            }
        }

        _vocabularySize = uniqueTokens.Count;
        _isTrained = true;
    }

    /// <summary>
    /// Computes the perplexity of a document under the trained language model.
    /// </summary>
    /// <param name="text">The document text.</param>
    /// <returns>The perplexity score. Lower means more predictable text.</returns>
    public double ComputePerplexity(string text)
    {
        if (!_isTrained)
            throw new InvalidOperationException("Language model must be trained before computing perplexity. Call Train() first.");

        string[] tokens = Tokenize(text);
        if (tokens.Length < _options.NGramOrder)
            return double.MaxValue;

        double logProb = 0;
        int count = 0;

        for (int i = 0; i <= tokens.Length - _options.NGramOrder; i++)
        {
            string ngram = string.Join(" ", tokens, i, _options.NGramOrder);
            string context = string.Join(" ", tokens, i, _options.NGramOrder - 1);

            int ngramCount = _ngramCounts.GetValueOrDefault(ngram, 0);
            int contextCount = _contextCounts.GetValueOrDefault(context, 0);

            // Laplace smoothing
            double prob = (ngramCount + _options.SmoothingFactor) /
                         (contextCount + _options.SmoothingFactor * _vocabularySize);

            logProb += Math.Log(prob);
            count++;
        }

        if (count == 0) return double.MaxValue;
        return Math.Exp(-logProb / count);
    }

    /// <summary>
    /// Filters documents by perplexity, returning indices of documents that should be removed.
    /// </summary>
    /// <param name="documents">Documents to filter.</param>
    /// <returns>Set of indices that fail the perplexity check (should be removed).</returns>
    public HashSet<int> Filter(IReadOnlyList<string> documents)
    {
        if (!_isTrained)
            throw new InvalidOperationException("Language model must be trained before filtering. Call Train() first.");

        var filtered = new HashSet<int>();

        for (int i = 0; i < documents.Count; i++)
        {
            double perplexity = ComputePerplexity(documents[i]);
            if (perplexity > _options.MaxPerplexity ||
                (_options.MinPerplexity > 0 && perplexity < _options.MinPerplexity))
            {
                filtered.Add(i);
            }
        }

        return filtered;
    }

    private static string[] Tokenize(string text)
    {
        return text.ToLowerInvariant()
            .Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
    }
}
