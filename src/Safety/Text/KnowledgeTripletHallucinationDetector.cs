using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Detects hallucinations by extracting (subject, predicate, object) knowledge triplets
/// and verifying them against reference documents.
/// </summary>
/// <remarks>
/// <para>
/// Parses the model output into knowledge triplets (e.g., "Paris – capital of – France")
/// and checks each triplet against the reference corpus. Triplets that cannot be grounded
/// in any reference document are flagged as potential hallucinations. This approach is more
/// precise than sentence-level grounding because it isolates individual factual claims.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine breaking every sentence into simple facts like
/// "X is related to Y in way Z". Then we check each fact against the source documents.
/// If a fact doesn't appear in any source, the AI probably made it up.
/// </para>
/// <para>
/// <b>References:</b>
/// - RefChecker: Reference-based fine-grained hallucination via knowledge triplets (Amazon, 2024, arxiv:2405.14486)
/// - HHEM 2.1: Production-grade hallucination evaluation outperforming GPT-4 (Vectara, 2024)
/// - Triplet extraction for hallucination detection survey (2025, arxiv:2503.08100)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class KnowledgeTripletHallucinationDetector<T> : TextSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _threshold;
    private readonly string[] _referenceDocuments;
    private readonly int _embeddingDim;

    // Common relation indicators for triplet extraction
    private static readonly string[] RelationIndicators =
    {
        " is ", " are ", " was ", " were ", " has ", " have ", " had ",
        " contains ", " includes ", " belongs to ", " located in ",
        " founded in ", " built in ", " created by ", " invented by ",
        " capital of ", " part of ", " known as ", " called ",
        " produces ", " manufactures ", " leads ", " won ",
        " born in ", " died in ", " married to ", " works at ",
        " consists of ", " made of ", " caused by ", " results in "
    };

    /// <inheritdoc />
    public override string ModuleName => "KnowledgeTripletHallucinationDetector";

    /// <summary>
    /// Initializes a new knowledge triplet hallucination detector.
    /// </summary>
    /// <param name="referenceDocuments">Source documents to verify triplets against.</param>
    /// <param name="threshold">Grounding similarity threshold (0-1). Default: 0.4.</param>
    /// <param name="embeddingDim">Embedding dimension. Default: 64.</param>
    public KnowledgeTripletHallucinationDetector(
        string[]? referenceDocuments = null,
        double threshold = 0.4,
        int embeddingDim = 64)
    {
        _threshold = threshold;
        _referenceDocuments = referenceDocuments ?? Array.Empty<string>();
        _embeddingDim = embeddingDim;
    }

    /// <summary>
    /// Returns a new detector with updated reference documents.
    /// </summary>
    public KnowledgeTripletHallucinationDetector<T> WithReferences(string[] documents)
    {
        return new KnowledgeTripletHallucinationDetector<T>(documents, _threshold, _embeddingDim);
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        if (string.IsNullOrWhiteSpace(text) || _referenceDocuments.Length == 0)
        {
            return findings;
        }

        // Extract knowledge triplets from the text
        var triplets = ExtractTriplets(text);
        if (triplets.Count == 0)
        {
            return findings;
        }

        // Verify each triplet against reference documents
        int ungroundedCount = 0;
        var ungroundedExamples = new List<string>();

        foreach (var triplet in triplets)
        {
            double groundingScore = VerifyTriplet(triplet);

            if (groundingScore < _threshold)
            {
                ungroundedCount++;
                if (ungroundedExamples.Count < 3)
                {
                    string tripletStr = $"({triplet.Subject}, {triplet.Relation}, {triplet.Object})";
                    if (tripletStr.Length > 80)
                    {
                        tripletStr = tripletStr.Substring(0, 77) + "...";
                    }
                    ungroundedExamples.Add(tripletStr);
                }
            }
        }

        if (ungroundedCount > 0)
        {
            double hallucinationRate = (double)ungroundedCount / triplets.Count;

            if (hallucinationRate > 0.15) // More than 15% ungrounded triplets
            {
                findings.Add(new SafetyFinding
                {
                    Category = SafetyCategory.Hallucination,
                    Severity = hallucinationRate > 0.5 ? SafetySeverity.High : SafetySeverity.Medium,
                    Confidence = Math.Min(1.0, hallucinationRate),
                    Description = $"Knowledge triplet analysis: {ungroundedCount}/{triplets.Count} triplets " +
                                  $"not grounded in references ({hallucinationRate:P0}). " +
                                  $"Ungrounded: {string.Join("; ", ungroundedExamples)}",
                    RecommendedAction = SafetyAction.Warn,
                    SourceModule = ModuleName
                });
            }
        }

        return findings;
    }

    private List<KnowledgeTriplet> ExtractTriplets(string text)
    {
        var triplets = new List<KnowledgeTriplet>();
        string[] sentences = SplitIntoSentences(text);

        foreach (var sentence in sentences)
        {
            if (sentence.Length < 10) continue;

            string lower = sentence.ToLowerInvariant();

            // Try each relation indicator to split sentence into (subject, relation, object)
            foreach (var relation in RelationIndicators)
            {
                int relIdx = lower.IndexOf(relation, StringComparison.Ordinal);
                if (relIdx <= 0) continue;

                string subject = sentence.Substring(0, relIdx).Trim();
                string obj = sentence.Substring(relIdx + relation.Length).Trim();

                // Clean up: remove trailing punctuation from object
                obj = obj.TrimEnd('.', '!', '?', ',', ';');

                // Only accept if both subject and object have reasonable length
                if (subject.Length >= 2 && subject.Length <= 100 &&
                    obj.Length >= 2 && obj.Length <= 100)
                {
                    triplets.Add(new KnowledgeTriplet
                    {
                        Subject = subject,
                        Relation = relation.Trim(),
                        Object = obj,
                        OriginalSentence = sentence
                    });
                    break; // One triplet per sentence (use first match)
                }
            }
        }

        return triplets;
    }

    private double VerifyTriplet(KnowledgeTriplet triplet)
    {
        double maxScore = 0;

        // Build a triplet query string for matching
        string tripletQuery = $"{triplet.Subject} {triplet.Relation} {triplet.Object}".ToLowerInvariant();
        var queryNgrams = ExtractWordNgrams(tripletQuery, 2);
        var queryEmb = ComputeEmbedding(tripletQuery);

        foreach (var doc in _referenceDocuments)
        {
            string docLower = doc.ToLowerInvariant();

            // 1. Check if subject and object both appear in the document
            bool subjectPresent = docLower.Contains(triplet.Subject.ToLowerInvariant());
            bool objectPresent = docLower.Contains(triplet.Object.ToLowerInvariant());
            double entityScore = 0;
            if (subjectPresent && objectPresent) entityScore = 0.6;
            else if (subjectPresent || objectPresent) entityScore = 0.2;

            // 2. N-gram overlap of triplet query against document
            var docNgrams = ExtractWordNgrams(docLower, 2);
            int overlap = 0;
            foreach (var ng in queryNgrams)
            {
                if (docNgrams.Contains(ng)) overlap++;
            }
            double ngramScore = queryNgrams.Count > 0 ? (double)overlap / queryNgrams.Count : 0;

            // 3. Embedding similarity
            var docEmb = ComputeEmbedding(docLower);
            double embScore = ComputeCosineSimilarity(queryEmb, docEmb);

            // Combined: 40% entity co-occurrence + 30% n-gram + 30% embedding
            double combined = 0.4 * entityScore + 0.3 * ngramScore + 0.3 * embScore;

            if (combined > maxScore) maxScore = combined;
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

    private static HashSet<string> ExtractWordNgrams(string text, int n)
    {
        var ngrams = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
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
                if (sentence.Length > 5)
                {
                    sentences.Add(sentence);
                }
                start = i + 1;
            }
        }

        if (start < text.Length)
        {
            string remaining = text.Substring(start).Trim();
            if (remaining.Length > 5)
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

    private struct KnowledgeTriplet
    {
        public string Subject;
        public string Relation;
        public string Object;
        public string OriginalSentence;
    }
}
