using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Detects potential copyright violations using embedding-based semantic similarity
/// to known copyrighted works.
/// </summary>
/// <remarks>
/// <para>
/// While <see cref="NgramCopyrightDetector{T}"/> catches verbatim copying, this detector catches
/// paraphrased or semantically similar reproductions of copyrighted content. Each reference work
/// is split into passages, embedded into a fixed-dimensional vector space, and compared against
/// the model output using cosine similarity. High similarity to specific passages indicates
/// potential memorization even when exact wording differs.
/// </para>
/// <para>
/// <b>For Beginners:</b> If someone rewrites a copyrighted book using different words but the
/// same ideas and structure, n-gram matching won't catch it. This module converts text into
/// mathematical representations that capture meaning, so it can detect "same content, different
/// words" situations.
/// </para>
/// <para>
/// <b>References:</b>
/// - DE-COP: Detecting copyrighted content via paraphrased permutations (2024, arxiv:2402.09910)
/// - CopyBench: Measuring literal and non-literal copyright memorization (2024, arxiv:2407.07087)
/// - BookMIA: Practical membership inference for book-level memorization (2024, arxiv:2401.15588)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EmbeddingCopyrightDetector<T> : TextSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _threshold;
    private readonly int _embeddingDim;
    private readonly int _passageLength;
    private readonly Vector<T>[][] _passageEmbeddings; // [work][passage]
    private readonly string[] _sourceNames;

    /// <inheritdoc />
    public override string ModuleName => "EmbeddingCopyrightDetector";

    /// <summary>
    /// Initializes a new embedding-based copyright detector.
    /// </summary>
    /// <param name="copyrightedTexts">Full texts of copyrighted works to check against.</param>
    /// <param name="sourceNames">Names of the copyrighted works (parallel array).</param>
    /// <param name="threshold">Similarity threshold (0-1). Default: 0.7.</param>
    /// <param name="embeddingDim">Embedding dimension. Default: 128.</param>
    /// <param name="passageLength">Words per passage for chunking. Default: 50.</param>
    public EmbeddingCopyrightDetector(
        string[]? copyrightedTexts = null,
        string[]? sourceNames = null,
        double threshold = 0.7,
        int embeddingDim = 128,
        int passageLength = 50)
    {
        _threshold = threshold;
        _embeddingDim = embeddingDim;
        _passageLength = passageLength;
        _sourceNames = sourceNames ?? Array.Empty<string>();

        var texts = copyrightedTexts ?? Array.Empty<string>();
        _passageEmbeddings = new Vector<T>[texts.Length][];

        for (int w = 0; w < texts.Length; w++)
        {
            var passages = ChunkIntoPassages(texts[w]);
            _passageEmbeddings[w] = new Vector<T>[passages.Length];
            for (int p = 0; p < passages.Length; p++)
            {
                _passageEmbeddings[w][p] = ComputeEmbedding(passages[p].ToLowerInvariant());
            }
        }
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        if (string.IsNullOrWhiteSpace(text) || _passageEmbeddings.Length == 0)
        {
            return findings;
        }

        // Chunk the output text into passages
        string[] outputPassages = ChunkIntoPassages(text);
        if (outputPassages.Length == 0)
        {
            return findings;
        }

        // Compare each output passage against each copyrighted passage
        for (int w = 0; w < _passageEmbeddings.Length; w++)
        {
            double maxSimilarity = 0;
            int matchingPassages = 0;

            foreach (var outPassage in outputPassages)
            {
                var outEmb = ComputeEmbedding(outPassage.ToLowerInvariant());
                double bestPassageSim = 0;

                for (int p = 0; p < _passageEmbeddings[w].Length; p++)
                {
                    double sim = ComputeCosineSimilarity(outEmb, _passageEmbeddings[w][p]);
                    if (sim > bestPassageSim) bestPassageSim = sim;
                }

                if (bestPassageSim >= _threshold)
                {
                    matchingPassages++;
                }
                if (bestPassageSim > maxSimilarity)
                {
                    maxSimilarity = bestPassageSim;
                }
            }

            if (matchingPassages > 0)
            {
                double matchRate = (double)matchingPassages / outputPassages.Length;
                string sourceName = w < _sourceNames.Length ? _sourceNames[w] : $"Work #{w + 1}";

                // Single high-similarity passage = warning; multiple = higher severity
                SafetySeverity severity = matchRate > 0.3
                    ? SafetySeverity.High
                    : matchingPassages > 1
                        ? SafetySeverity.Medium
                        : SafetySeverity.Low;

                SafetyAction action = matchRate > 0.3
                    ? SafetyAction.Block
                    : SafetyAction.Warn;

                findings.Add(new SafetyFinding
                {
                    Category = SafetyCategory.CopyrightViolation,
                    Severity = severity,
                    Confidence = maxSimilarity,
                    Description = $"Semantic similarity to '{sourceName}': {matchingPassages}/{outputPassages.Length} " +
                                  $"passages exceed threshold (max similarity: {maxSimilarity:F3}). " +
                                  $"This may indicate paraphrased reproduction of copyrighted content.",
                    RecommendedAction = action,
                    SourceModule = ModuleName
                });
            }
        }

        return findings;
    }

    private string[] ChunkIntoPassages(string text)
    {
        string[] words = text.Split(new[] { ' ', '\t', '\n', '\r' },
            StringSplitOptions.RemoveEmptyEntries);

        if (words.Length <= _passageLength)
        {
            return words.Length >= 5 ? new[] { text } : Array.Empty<string>();
        }

        var passages = new List<string>();
        int stride = _passageLength / 2; // 50% overlap

        for (int i = 0; i <= words.Length - _passageLength; i += stride)
        {
            int count = Math.Min(_passageLength, words.Length - i);
            passages.Add(string.Join(" ", words, i, count));
        }

        return passages.ToArray();
    }

    private Vector<T> ComputeEmbedding(string text)
    {
        var embedding = new Vector<T>(_embeddingDim);

        // Character trigrams
        for (int i = 0; i <= text.Length - 3; i++)
        {
            int hash = HashNgram(text, i, 3);
            int idx = ((hash % _embeddingDim) + _embeddingDim) % _embeddingDim;
            T delta = (hash & 0x80000) != 0 ? NumOps.One : NumOps.Negate(NumOps.One);
            embedding[idx] = NumOps.Add(embedding[idx], delta);
        }

        // Character 4-grams (captures more structure)
        for (int i = 0; i <= text.Length - 4; i++)
        {
            int hash = HashNgram(text, i, 4);
            int idx = ((hash % _embeddingDim) + _embeddingDim) % _embeddingDim;
            T delta = (hash & 0x80000) != 0 ? NumOps.One : NumOps.Negate(NumOps.One);
            embedding[idx] = NumOps.Add(embedding[idx], delta);
        }

        // L2 normalize
        T sumSq = NumOps.Zero;
        for (int i = 0; i < _embeddingDim; i++)
        {
            sumSq = NumOps.Add(sumSq, NumOps.Multiply(embedding[i], embedding[i]));
        }

        double norm = Math.Sqrt(NumOps.ToDouble(sumSq));
        if (norm > 1e-10)
        {
            T normT = NumOps.FromDouble(norm);
            for (int i = 0; i < _embeddingDim; i++)
            {
                embedding[i] = NumOps.Divide(embedding[i], normT);
            }
        }

        return embedding;
    }

    private static double ComputeCosineSimilarity(Vector<T> a, Vector<T> b)
    {
        double dot = 0, normA = 0, normB = 0;
        int len = Math.Min(a.Length, b.Length);

        for (int i = 0; i < len; i++)
        {
            double ai = NumOps.ToDouble(a[i]);
            double bi = NumOps.ToDouble(b[i]);
            dot += ai * bi;
            normA += ai * ai;
            normB += bi * bi;
        }

        double denom = Math.Sqrt(normA * normB);
        return denom > 1e-10 ? Math.Max(0, dot / denom) : 0;
    }

    private static int HashNgram(string text, int start, int length)
    {
        unchecked
        {
            int hash = (int)2166136261;
            for (int i = start; i < start + length && i < text.Length; i++)
            {
                hash ^= text[i];
                hash *= 16777619;
            }
            return hash;
        }
    }
}
