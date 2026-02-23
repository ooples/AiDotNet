using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Detects hallucinations using textual entailment (NLI) principles: checking whether
/// reference documents entail (support) each claim in the model output.
/// </summary>
/// <remarks>
/// <para>
/// For each sentence in the model output, computes an entailment score against each reference
/// document. A sentence is "entailed" if it is logically supported by the reference. Sentences
/// that are contradicted or neutral (not supported) are flagged as potential hallucinations.
/// The approach uses lexical overlap, negation detection, and entity alignment as lightweight
/// proxies for full NLI model inference.
/// </para>
/// <para>
/// <b>For Beginners:</b> Given source documents and an AI response, this module checks whether
/// each statement in the response logically follows from the sources. Statements that contradict
/// or go beyond the sources are flagged as hallucinations.
/// </para>
/// <para>
/// <b>References:</b>
/// - TRUE: Re-evaluating factual consistency via NLI (Google, 2023)
/// - SummaC: NLI-based consistency benchmark for summarization (2022)
/// - MiniCheck: Efficient NLI fact-checking grounding (2024, arxiv:2404.10774)
/// - AlignScore: Unified alignment for factual consistency (ACL 2023)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EntailmentHallucinationDetector<T> : TextSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _entailmentThreshold;
    private readonly string[] _referenceDocuments;
    private readonly int _embeddingDim;

    private static readonly string[] NegationWords =
    {
        "not", "never", "no", "none", "neither", "nor", "nothing",
        "nowhere", "hardly", "barely", "scarcely", "isn't", "wasn't",
        "weren't", "won't", "don't", "doesn't", "didn't", "can't",
        "couldn't", "shouldn't", "wouldn't"
    };

    /// <inheritdoc />
    public override string ModuleName => "EntailmentHallucinationDetector";

    /// <summary>
    /// Initializes a new entailment-based hallucination detector.
    /// </summary>
    /// <param name="referenceDocuments">Source documents to check entailment against.</param>
    /// <param name="entailmentThreshold">
    /// Minimum entailment score for a claim to be considered grounded (0-1). Default: 0.4.
    /// </param>
    /// <param name="embeddingDim">Embedding dimension. Default: 64.</param>
    public EntailmentHallucinationDetector(
        string[]? referenceDocuments = null,
        double entailmentThreshold = 0.4,
        int embeddingDim = 64)
    {
        _entailmentThreshold = entailmentThreshold;
        _referenceDocuments = referenceDocuments ?? Array.Empty<string>();
        _embeddingDim = embeddingDim;
    }

    /// <summary>
    /// Returns a new detector with updated reference documents.
    /// </summary>
    public EntailmentHallucinationDetector<T> WithReferences(string[] documents)
    {
        return new EntailmentHallucinationDetector<T>(documents, _entailmentThreshold, _embeddingDim);
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        if (string.IsNullOrWhiteSpace(text) || _referenceDocuments.Length == 0)
        {
            return findings;
        }

        string[] sentences = SplitIntoSentences(text);
        int totalClaims = 0;
        int contradictedClaims = 0;
        int neutralClaims = 0;
        var contradictionExamples = new List<string>();
        var neutralExamples = new List<string>();

        foreach (var sentence in sentences)
        {
            if (sentence.Length < 10) continue;
            totalClaims++;

            var (entailmentScore, isContradiction) = ComputeEntailmentScore(sentence);

            if (isContradiction)
            {
                contradictedClaims++;
                if (contradictionExamples.Count < 2)
                {
                    contradictionExamples.Add(TruncateSentence(sentence, 70));
                }
            }
            else if (entailmentScore < _entailmentThreshold)
            {
                neutralClaims++;
                if (neutralExamples.Count < 2)
                {
                    neutralExamples.Add(TruncateSentence(sentence, 70));
                }
            }
        }

        if (totalClaims == 0) return findings;

        // Report contradictions (high severity)
        if (contradictedClaims > 0)
        {
            double contradictionRate = (double)contradictedClaims / totalClaims;
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Hallucination,
                Severity = SafetySeverity.High,
                Confidence = Math.Min(1.0, contradictionRate * 2),
                Description = $"Entailment analysis: {contradictedClaims}/{totalClaims} claims contradict " +
                              $"reference documents. Examples: {string.Join("; ", contradictionExamples)}",
                RecommendedAction = SafetyAction.Block,
                SourceModule = ModuleName
            });
        }

        // Report neutral/unsupported claims (medium severity)
        if (neutralClaims > 0)
        {
            double neutralRate = (double)neutralClaims / totalClaims;
            if (neutralRate > 0.25) // More than 25% unsupported
            {
                findings.Add(new SafetyFinding
                {
                    Category = SafetyCategory.Hallucination,
                    Severity = SafetySeverity.Medium,
                    Confidence = neutralRate,
                    Description = $"Entailment analysis: {neutralClaims}/{totalClaims} claims not supported by " +
                                  $"reference documents ({neutralRate:P0}). " +
                                  $"Examples: {string.Join("; ", neutralExamples)}",
                    RecommendedAction = SafetyAction.Warn,
                    SourceModule = ModuleName
                });
            }
        }

        return findings;
    }

    private (double entailmentScore, bool isContradiction) ComputeEntailmentScore(string claim)
    {
        double maxEntailment = 0;
        bool anyContradiction = false;
        string claimLower = claim.ToLowerInvariant();
        bool claimHasNegation = HasNegation(claimLower);

        // Extract content words from claim (skip short function words)
        var claimContentWords = ExtractContentWords(claimLower);
        var claimEmb = ComputeEmbedding(claimLower);

        foreach (var doc in _referenceDocuments)
        {
            string docLower = doc.ToLowerInvariant();

            // Find the most relevant sentence in the document
            string[] docSentences = SplitIntoSentences(doc);
            double bestSentenceScore = 0;
            bool sentenceContradicts = false;

            foreach (var docSentence in docSentences)
            {
                if (docSentence.Length < 5) continue;
                string docSentLower = docSentence.ToLowerInvariant();

                // 1. Lexical overlap of content words
                var docContentWords = ExtractContentWords(docSentLower);
                int sharedWords = 0;
                foreach (var w in claimContentWords)
                {
                    if (docContentWords.Contains(w)) sharedWords++;
                }
                double lexicalScore = claimContentWords.Count > 0
                    ? (double)sharedWords / claimContentWords.Count
                    : 0;

                // Skip if no lexical overlap (unrelated sentences)
                if (lexicalScore < 0.1) continue;

                // 2. Embedding similarity
                var docSentEmb = ComputeEmbedding(docSentLower);
                double embScore = ComputeCosineSimilarity(claimEmb, docSentEmb);

                // 3. Negation check: similar content but opposite polarity = contradiction
                bool docHasNegation = HasNegation(docSentLower);
                if (lexicalScore > 0.3 && embScore > 0.4 &&
                    claimHasNegation != docHasNegation)
                {
                    sentenceContradicts = true;
                }

                // 4. Number conflict check
                if (lexicalScore > 0.3 && HasNumericConflict(claimLower, docSentLower))
                {
                    sentenceContradicts = true;
                }

                // Entailment score: 50% lexical + 50% embedding
                double sentScore = 0.5 * lexicalScore + 0.5 * embScore;
                if (sentScore > bestSentenceScore)
                {
                    bestSentenceScore = sentScore;
                }
            }

            if (sentenceContradicts)
            {
                anyContradiction = true;
            }
            if (bestSentenceScore > maxEntailment)
            {
                maxEntailment = bestSentenceScore;
            }
        }

        return (maxEntailment, anyContradiction);
    }

    private static bool HasNegation(string text)
    {
        foreach (var neg in NegationWords)
        {
            // Check for whole-word match
            int idx = text.IndexOf(neg, StringComparison.Ordinal);
            while (idx >= 0)
            {
                bool leftBoundary = idx == 0 || !char.IsLetter(text[idx - 1]);
                bool rightBoundary = idx + neg.Length >= text.Length ||
                                     !char.IsLetter(text[idx + neg.Length]);
                if (leftBoundary && rightBoundary) return true;

                idx = text.IndexOf(neg, idx + 1, StringComparison.Ordinal);
            }
        }
        return false;
    }

    private static bool HasNumericConflict(string sent1, string sent2)
    {
        var nums1 = ExtractNumbers(sent1);
        var nums2 = ExtractNumbers(sent2);

        if (nums1.Count == 0 || nums2.Count == 0) return false;

        // Check if same context words appear with different numbers
        var words1 = new HashSet<string>(
            sent1.Split(' ').Where(w => w.Length > 3 && !double.TryParse(w, out _)));
        var words2 = new HashSet<string>(
            sent2.Split(' ').Where(w => w.Length > 3 && !double.TryParse(w, out _)));

        int shared = 0;
        foreach (var w in words1)
        {
            if (words2.Contains(w)) shared++;
        }
        if (shared < 2) return false;

        foreach (var n1 in nums1)
        {
            foreach (var n2 in nums2)
            {
                double diff = Math.Abs(n1 - n2);
                if (diff > 0.001 && diff / Math.Max(Math.Abs(n1), 1) > 0.1)
                {
                    return true;
                }
            }
        }

        return false;
    }

    private static List<double> ExtractNumbers(string text)
    {
        var numbers = new List<double>();
        int i = 0;
        while (i < text.Length)
        {
            if (char.IsDigit(text[i]) ||
                (text[i] == '-' && i + 1 < text.Length && char.IsDigit(text[i + 1])))
            {
                int start = i;
                i++;
                while (i < text.Length && (char.IsDigit(text[i]) || text[i] == '.' || text[i] == ','))
                {
                    i++;
                }
                string numStr = text.Substring(start, i - start).Replace(",", "");
                if (double.TryParse(numStr, out double num))
                {
                    numbers.Add(num);
                }
            }
            else
            {
                i++;
            }
        }
        return numbers;
    }

    private static HashSet<string> ExtractContentWords(string text)
    {
        var words = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var word in text.Split(new[] { ' ', '\t', '\n', '\r', ',', ';', ':', '(', ')', '[', ']' },
                     StringSplitOptions.RemoveEmptyEntries))
        {
            // Skip short function words
            if (word.Length > 3)
            {
                words.Add(word.TrimEnd('.', '!', '?', '\''));
            }
        }
        return words;
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

    private static string TruncateSentence(string sentence, int maxLength)
    {
        if (sentence.Length <= maxLength) return sentence;
        return sentence.Substring(0, maxLength - 3) + "...";
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
