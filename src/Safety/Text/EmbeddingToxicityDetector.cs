using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Detects toxic text using embedding-based cosine similarity to known toxic concept vectors.
/// </summary>
/// <remarks>
/// <para>
/// Computes a lightweight embedding (character n-gram hash vectors) of the input text and
/// measures cosine similarity against pre-built concept vectors for known toxic categories.
/// This approach catches semantically similar content that regex-based approaches miss.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of looking for specific bad words, this module converts text
/// into a mathematical representation (embedding) and measures how "close" it is to known
/// examples of toxic content. This catches rephrasings, misspellings, and subtle toxicity
/// that keyword matching would miss.
/// </para>
/// <para>
/// <b>References:</b>
/// - MetaTox knowledge graph for enhanced LLM toxicity detection (2024, arxiv:2412.15268)
/// - LLM-extracted rationales for interpretable hate speech detection (2024, arxiv:2403.12403)
/// - GPT-4o/LLaMA-3 zero-shot hate speech detection (2025, arxiv:2506.12744)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EmbeddingToxicityDetector<T> : TextSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T _threshold;
    private readonly int _embeddingDim;

    // Pre-built toxic concept vectors (character n-gram hash space)
    private readonly Vector<T>[] _toxicConceptVectors;
    private readonly SafetyCategory[] _toxicCategories;

    /// <inheritdoc />
    public override string ModuleName => "EmbeddingToxicityDetector";

    /// <summary>
    /// Initializes a new embedding-based toxicity detector.
    /// </summary>
    /// <param name="threshold">Cosine similarity threshold (0-1). Default: 0.6.</param>
    /// <param name="embeddingDim">Dimension of the hash embedding. Default: 128.</param>
    public EmbeddingToxicityDetector(double threshold = 0.6, int embeddingDim = 128)
    {
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be between 0 and 1.");
        }

        if (embeddingDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(embeddingDim),
                "Embedding dimension must be positive.");
        }

        _threshold = NumOps.FromDouble(threshold);
        _embeddingDim = embeddingDim;

        // Build toxic concept vectors from known toxic phrase sets
        var concepts = BuildToxicConcepts();
        _toxicConceptVectors = new Vector<T>[concepts.Count];
        _toxicCategories = new SafetyCategory[concepts.Count];

        for (int i = 0; i < concepts.Count; i++)
        {
            _toxicConceptVectors[i] = ComputeConceptEmbedding(concepts[i].Phrases);
            _toxicCategories[i] = concepts[i].Category;
        }
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        if (string.IsNullOrWhiteSpace(text))
        {
            return findings;
        }

        // Compute embedding for input text
        var inputEmbedding = ComputeTextEmbedding(text);

        // Compare against each toxic concept
        for (int i = 0; i < _toxicConceptVectors.Length; i++)
        {
            T similarity = CosineSimilarity(inputEmbedding, _toxicConceptVectors[i]);

            if (NumOps.GreaterThanOrEquals(similarity, _threshold))
            {
                double simDouble = NumOps.ToDouble(similarity);
                findings.Add(new SafetyFinding
                {
                    Category = _toxicCategories[i],
                    Severity = simDouble >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                    Confidence = simDouble,
                    Description = $"Text embedding similarity to {_toxicCategories[i]} concept: {simDouble:F3}.",
                    RecommendedAction = simDouble >= 0.8 ? SafetyAction.Block : SafetyAction.Warn,
                    SourceModule = ModuleName
                });
            }
        }

        return findings;
    }

    /// <summary>
    /// Computes a character n-gram hash embedding for the given text.
    /// Uses feature hashing (hashing trick) to map n-grams to a fixed-size vector.
    /// </summary>
    private Vector<T> ComputeTextEmbedding(string text)
    {
        var embedding = new Vector<T>(_embeddingDim);
        string normalized = text.ToLowerInvariant().Trim();

        // Character trigram hashing
        for (int i = 0; i <= normalized.Length - 3; i++)
        {
            int hash = HashNgram(normalized, i, 3);
            int idx = ((hash % _embeddingDim) + _embeddingDim) % _embeddingDim;
            // Use sign of hash for random projection
            T delta = (hash & 0x80000) != 0 ? NumOps.One : NumOps.Negate(NumOps.One);
            embedding[idx] = NumOps.Add(embedding[idx], delta);
        }

        // Character 4-gram hashing
        for (int i = 0; i <= normalized.Length - 4; i++)
        {
            int hash = HashNgram(normalized, i, 4);
            int idx = ((hash % _embeddingDim) + _embeddingDim) % _embeddingDim;
            T delta = (hash & 0x80000) != 0 ? NumOps.One : NumOps.Negate(NumOps.One);
            embedding[idx] = NumOps.Add(embedding[idx], delta);
        }

        // L2 normalize
        NormalizeVector(embedding);

        return embedding;
    }

    /// <summary>
    /// Computes a concept embedding by averaging embeddings of representative phrases.
    /// </summary>
    private Vector<T> ComputeConceptEmbedding(string[] phrases)
    {
        var centroid = new Vector<T>(_embeddingDim);

        foreach (var phrase in phrases)
        {
            var phraseEmb = ComputeTextEmbedding(phrase);
            for (int i = 0; i < _embeddingDim; i++)
            {
                centroid[i] = NumOps.Add(centroid[i], phraseEmb[i]);
            }
        }

        NormalizeVector(centroid);
        return centroid;
    }

    private static T CosineSimilarity(Vector<T> a, Vector<T> b)
    {
        T dot = NumOps.Zero;
        T normA = NumOps.Zero;
        T normB = NumOps.Zero;

        for (int i = 0; i < a.Length && i < b.Length; i++)
        {
            dot = NumOps.Add(dot, NumOps.Multiply(a[i], b[i]));
            normA = NumOps.Add(normA, NumOps.Multiply(a[i], a[i]));
            normB = NumOps.Add(normB, NumOps.Multiply(b[i], b[i]));
        }

        T denominator = NumOps.FromDouble(
            Math.Sqrt(NumOps.ToDouble(normA) * NumOps.ToDouble(normB)));

        T epsilon = NumOps.FromDouble(1e-10);
        if (NumOps.LessThan(denominator, epsilon)) return NumOps.Zero;

        T similarity = NumOps.Divide(dot, denominator);

        // Clamp to [0, 1]
        if (NumOps.LessThan(similarity, NumOps.Zero)) return NumOps.Zero;
        if (NumOps.GreaterThan(similarity, NumOps.One)) return NumOps.One;
        return similarity;
    }

    private static void NormalizeVector(Vector<T> v)
    {
        T sumSq = NumOps.Zero;
        for (int i = 0; i < v.Length; i++)
        {
            sumSq = NumOps.Add(sumSq, NumOps.Multiply(v[i], v[i]));
        }

        double norm = Math.Sqrt(NumOps.ToDouble(sumSq));
        if (norm < 1e-10) return;

        T normT = NumOps.FromDouble(norm);
        for (int i = 0; i < v.Length; i++)
        {
            v[i] = NumOps.Divide(v[i], normT);
        }
    }

    private static int HashNgram(string text, int start, int length)
    {
        // FNV-1a hash for character n-grams
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

    private static List<ToxicConcept> BuildToxicConcepts()
    {
        return new List<ToxicConcept>
        {
            new(SafetyCategory.ViolenceThreat, new[]
            {
                "i will hurt you", "kill you", "make you suffer",
                "destroy you", "find where you live", "physical harm",
                "beat you up", "coming after you"
            }),
            new(SafetyCategory.HateSpeech, new[]
            {
                "all of them are", "those people are inferior",
                "should be removed from", "don't belong here",
                "subhuman", "vermin", "parasites", "infestation"
            }),
            new(SafetyCategory.Harassment, new[]
            {
                "you are worthless", "nobody likes you",
                "you should be ashamed", "ugly and stupid",
                "waste of space", "pathetic loser"
            }),
            new(SafetyCategory.ViolenceSelfHarm, new[]
            {
                "hurt yourself", "end it all", "not worth living",
                "better off without you", "no reason to go on"
            }),
            new(SafetyCategory.SocialEngineering, new[]
            {
                "send me your password", "click this link urgently",
                "verify your account immediately", "your account will be suspended",
                "wire transfer urgent", "act now or lose access"
            })
        };
    }

    private readonly struct ToxicConcept
    {
        public readonly SafetyCategory Category;
        public readonly string[] Phrases;

        public ToxicConcept(SafetyCategory category, string[] phrases)
        {
            Category = category;
            Phrases = phrases;
        }
    }
}
