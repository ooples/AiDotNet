using System.Text;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.ContextCompression;

/// <summary>
/// Auto-compressor using rule-based text compression for document content reduction.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// Compresses documents by extracting the most relevant sentences based on keyword importance
/// and position in the document. This is a production implementation that doesn't require
/// external ML models.
/// </remarks>
public class AutoCompressor<T> : ContextCompressorBase<T>
{
    private readonly int _maxOutputLength;
    private readonly double _compressionRatio;

    /// <summary>
    /// Initializes a new instance of the <see cref="AutoCompressor{T}"/> class.
    /// </summary>
    /// <param name="maxOutputLength">Maximum length of compressed output in characters.</param>
    /// <param name="compressionRatio">Target compression ratio (0-1).</param>
    public AutoCompressor(int maxOutputLength = 500, double compressionRatio = 0.5)
    {
        if (maxOutputLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxOutputLength), "Max output length must be positive");

        if (compressionRatio <= 0 || compressionRatio > 1)
            throw new ArgumentOutOfRangeException(nameof(compressionRatio), "Compression ratio must be between 0 and 1");

        _maxOutputLength = maxOutputLength;
        _compressionRatio = compressionRatio;
    }

    /// <summary>
    /// Compresses documents using rule-based sentence extraction.
    /// </summary>
    protected override List<Document<T>> CompressCore(
        List<Document<T>> documents,
        string query,
        Dictionary<string, object>? options = null)
    {
        var queryTokens = new HashSet<string>(query.ToLowerInvariant().Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries));
        var compressedDocs = new List<Document<T>>();

        foreach (var doc in documents)
        {
            var compressed = CompressDocument(doc.Content, queryTokens);
            var compressedDoc = new Document<T>(doc.Id, compressed)
            {
                RelevanceScore = doc.RelevanceScore,
                HasRelevanceScore = doc.HasRelevanceScore
            };
            foreach (var kvp in doc.Metadata)
            {
                compressedDoc.Metadata[kvp.Key] = kvp.Value;
            }
            compressedDocs.Add(compressedDoc);
        }

        return compressedDocs;
    }

    private string CompressDocument(string content, HashSet<string> queryTokens)
    {
        if (string.IsNullOrWhiteSpace(content))
            return string.Empty;

        var sentences = SplitIntoSentences(content);
        if (sentences.Count == 0)
            return string.Empty;

        var targetSentenceCount = Math.Max(1, (int)(sentences.Count * _compressionRatio));

        var scoredSentences = sentences
            .Select((sentence, index) => new
            {
                Sentence = sentence,
                Score = ScoreSentence(sentence, queryTokens, index, sentences.Count)
            })
            .OrderByDescending(x => x.Score)
            .Take(targetSentenceCount)
            .OrderBy(x => sentences.IndexOf(x.Sentence))
            .Select(x => x.Sentence)
            .ToList();

        var result = string.Join(" ", scoredSentences);

        if (result.Length > _maxOutputLength)
        {
            result = result.Substring(0, _maxOutputLength);
            var lastSpace = result.LastIndexOf(' ');
            if (lastSpace > _maxOutputLength / 2)
                result = result.Substring(0, lastSpace);
        }

        return result;
    }

    private double ScoreSentence(string sentence, HashSet<string> queryTokens, int position, int totalSentences)
    {
        var tokens = sentence.ToLowerInvariant().Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
        var matchCount = tokens.Count(t => queryTokens.Contains(t));
        var matchRatio = tokens.Length > 0 ? (double)matchCount / tokens.Length : 0;

        var positionScore = 1.0 - ((double)position / totalSentences);
        positionScore = positionScore * 0.3;

        return (matchRatio * 0.7) + positionScore;
    }

    private List<string> SplitIntoSentences(string text)
    {
        return Helpers.TextProcessingHelper.SplitIntoSentences(text);
    }
}
