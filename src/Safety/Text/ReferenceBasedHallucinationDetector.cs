using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Detects hallucinations by comparing model output against provided reference documents.
/// </summary>
/// <remarks>
/// <para>
/// Measures the overlap between claims in the model output and information present in reference
/// documents. Claims that cannot be grounded in the reference material are flagged as potential
/// hallucinations. Uses n-gram overlap and embedding similarity for grounding verification.
/// </para>
/// <para>
/// <b>For Beginners:</b> When an AI makes a claim, this module checks whether that claim
/// is supported by the source documents. If the AI says something that isn't in the sources,
/// it's likely a "hallucination" — something the AI made up.
/// </para>
/// <para>
/// <b>References:</b>
/// - RefChecker: Knowledge triplet-based detection (Amazon, 2024, arxiv:2405.14486)
/// - HHEM 2.1/2.3: Production-grade detection beating GPT-4 (Vectara, 2024-2025)
/// - FaithBench: Benchmarking hallucination in summarization (2025, arxiv:2505.04847)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ReferenceBasedHallucinationDetector<T> : TextSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _threshold;
    private readonly string[] _referenceDocuments;
    private readonly int _embeddingDim;

    /// <inheritdoc />
    public override string ModuleName => "ReferenceBasedHallucinationDetector";

    /// <summary>
    /// Initializes a new reference-based hallucination detector.
    /// </summary>
    /// <param name="referenceDocuments">Source documents to check against.</param>
    /// <param name="threshold">Hallucination score threshold (0-1). Default: 0.5.</param>
    /// <param name="embeddingDim">Embedding dimension for similarity. Default: 64.</param>
    public ReferenceBasedHallucinationDetector(
        string[]? referenceDocuments = null,
        double threshold = 0.5,
        int embeddingDim = 64)
    {
        _threshold = threshold;
        _referenceDocuments = referenceDocuments ?? Array.Empty<string>();
        _embeddingDim = embeddingDim;
    }

    /// <summary>
    /// Sets or updates the reference documents for grounding checks.
    /// </summary>
    public ReferenceBasedHallucinationDetector<T> WithReferences(string[] documents)
    {
        return new ReferenceBasedHallucinationDetector<T>(documents, _threshold, _embeddingDim);
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        if (string.IsNullOrWhiteSpace(text) || _referenceDocuments.Length == 0)
        {
            return findings;
        }

        // Split output into sentences (claims)
        string[] sentences = SplitIntoSentences(text);

        int ungroundedCount = 0;
        int totalClaims = 0;
        var ungroundedClaims = new List<string>();

        foreach (var sentence in sentences)
        {
            if (sentence.Length < 10) continue; // Skip very short sentences
            totalClaims++;

            double groundingScore = ComputeGroundingScore(sentence);

            if (groundingScore < _threshold)
            {
                ungroundedCount++;
                if (ungroundedClaims.Count < 3) // Limit reported examples
                {
                    ungroundedClaims.Add(sentence.Length > 80
                        ? sentence.Substring(0, 77) + "..."
                        : sentence);
                }
            }
        }

        if (totalClaims > 0 && ungroundedCount > 0)
        {
            double hallucinationRate = (double)ungroundedCount / totalClaims;

            if (hallucinationRate > 0.2) // More than 20% ungrounded claims
            {
                findings.Add(new SafetyFinding
                {
                    Category = SafetyCategory.Hallucination,
                    Severity = hallucinationRate > 0.5 ? SafetySeverity.High : SafetySeverity.Medium,
                    Confidence = hallucinationRate,
                    Description = $"Potential hallucination detected: {ungroundedCount}/{totalClaims} claims " +
                                  $"not grounded in reference documents ({hallucinationRate:P0}). " +
                                  $"Examples: {string.Join("; ", ungroundedClaims)}",
                    RecommendedAction = SafetyAction.Warn,
                    SourceModule = ModuleName
                });
            }
        }

        return findings;
    }

    private double ComputeGroundingScore(string claim)
    {
        double maxScore = 0;
        string claimLower = claim.ToLowerInvariant();

        // Extract claim n-grams
        var claimNgrams = ExtractNgrams(claimLower, 3);
        if (claimNgrams.Count == 0) return 1.0; // Trivial claim

        foreach (var doc in _referenceDocuments)
        {
            string docLower = doc.ToLowerInvariant();

            // N-gram overlap score
            var docNgrams = ExtractNgrams(docLower, 3);
            int overlapCount = 0;
            foreach (var ngram in claimNgrams)
            {
                if (docNgrams.Contains(ngram))
                {
                    overlapCount++;
                }
            }

            double ngramScore = claimNgrams.Count > 0
                ? (double)overlapCount / claimNgrams.Count
                : 0;

            // Embedding similarity score
            var claimEmb = ComputeEmbedding(claimLower);
            var docEmb = ComputeEmbedding(docLower);
            double embScore = ComputeCosineSimilarity(claimEmb, docEmb);

            // Combined: 60% n-gram overlap + 40% embedding similarity
            double combined = 0.6 * ngramScore + 0.4 * embScore;

            if (combined > maxScore)
            {
                maxScore = combined;
            }
        }

        return maxScore;
    }

    private Vector<T> ComputeEmbedding(string text)
    {
        var embedding = new Vector<T>(_embeddingDim);

        for (int i = 0; i <= text.Length - 3; i++)
        {
            int hash = HashNgram(text, i, 3);
            int idx = ((hash % _embeddingDim) + _embeddingDim) % _embeddingDim;
            T delta = (hash & 0x80000) != 0 ? NumOps.One : NumOps.Negate(NumOps.One);
            embedding[idx] = NumOps.Add(embedding[idx], delta);
        }

        // Normalize
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

    private static HashSet<string> ExtractNgrams(string text, int n)
    {
        var ngrams = new HashSet<string>();
        string[] words = text.Split(new[] { ' ', '\t', '\n', '\r' },
            StringSplitOptions.RemoveEmptyEntries);

        for (int i = 0; i <= words.Length - n; i++)
        {
            ngrams.Add(string.Join(" ", words, i, n));
        }

        return ngrams;
    }

    private static string[] SplitIntoSentences(string text)
    {
        var sentences = new List<string>();
        int start = 0;

        for (int i = 0; i < text.Length; i++)
        {
            if (text[i] == '.' || text[i] == '!' || text[i] == '?')
            {
                string sentence = text.Substring(start, i - start + 1).Trim();
                if (sentence.Length > 0)
                {
                    sentences.Add(sentence);
                }
                start = i + 1;
            }
        }

        // Add remaining text
        if (start < text.Length)
        {
            string remaining = text.Substring(start).Trim();
            if (remaining.Length > 0)
            {
                sentences.Add(remaining);
            }
        }

        return sentences.ToArray();
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
