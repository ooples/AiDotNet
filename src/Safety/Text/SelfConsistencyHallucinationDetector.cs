using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Detects hallucinations by checking internal consistency of claims within the text.
/// </summary>
/// <remarks>
/// <para>
/// Analyzes model output for internal contradictions and inconsistencies. When a model
/// hallucinates, it often produces statements that contradict each other or contain
/// logically impossible combinations. This detector identifies such patterns by comparing
/// sentence-level embeddings within the same document.
/// </para>
/// <para>
/// <b>For Beginners:</b> If an AI says "the building was built in 1990" and later says
/// "the building was constructed in 2005", those statements contradict each other. This module
/// finds such contradictions, which often indicate the AI is making things up.
/// </para>
/// <para>
/// <b>Detection approach:</b>
/// 1. Split text into sentences (claims)
/// 2. Compute embeddings for each sentence
/// 3. Compare all pairs for semantic contradiction signals
/// 4. Flag texts with high contradiction rates
/// </para>
/// <para>
/// <b>References:</b>
/// - SelfCheckGPT: Zero-resource hallucination detection (2023)
/// - ReDeEP: Hallucination detection in RAG systems (ICLR 2025)
/// - Hallucination survey: faithfulness vs factuality taxonomy (2025, arxiv:2510.06265)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SelfConsistencyHallucinationDetector<T> : TextSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _contradictionThreshold;
    private readonly int _embeddingDim;

    // Contradiction signal words
    private static readonly string[] NegationWords =
    {
        "not", "never", "no", "none", "neither", "nor", "nothing",
        "nowhere", "hardly", "barely", "scarcely", "isn't", "wasn't",
        "weren't", "won't", "don't", "doesn't", "didn't", "can't",
        "couldn't", "shouldn't", "wouldn't", "isn't"
    };

    private static readonly string[] ContradictionPairs =
    {
        "always|never", "all|none", "true|false", "correct|incorrect",
        "possible|impossible", "legal|illegal", "safe|unsafe",
        "increase|decrease", "before|after", "above|below",
        "more|less", "higher|lower", "larger|smaller"
    };

    /// <inheritdoc />
    public override string ModuleName => "SelfConsistencyHallucinationDetector";

    /// <summary>
    /// Initializes a new self-consistency hallucination detector.
    /// </summary>
    /// <param name="contradictionThreshold">
    /// Threshold for contradiction detection (0-1). Default: 0.3.
    /// </param>
    /// <param name="embeddingDim">Embedding dimension. Default: 64.</param>
    public SelfConsistencyHallucinationDetector(
        double contradictionThreshold = 0.3,
        int embeddingDim = 64)
    {
        _contradictionThreshold = contradictionThreshold;
        _embeddingDim = embeddingDim;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        if (string.IsNullOrWhiteSpace(text))
        {
            return findings;
        }

        string[] sentences = SplitIntoSentences(text);
        if (sentences.Length < 2)
        {
            return findings; // Need at least 2 sentences for consistency check
        }

        // Compute embeddings for all sentences
        var embeddings = new Vector<T>[sentences.Length];
        for (int i = 0; i < sentences.Length; i++)
        {
            embeddings[i] = ComputeEmbedding(sentences[i].ToLowerInvariant());
        }

        // Check all pairs for contradictions
        int contradictions = 0;
        int totalPairs = 0;
        var contradictionExamples = new List<string>();

        for (int i = 0; i < sentences.Length; i++)
        {
            for (int j = i + 1; j < sentences.Length; j++)
            {
                totalPairs++;
                double contradictionScore = ComputeContradictionScore(
                    sentences[i], sentences[j], embeddings[i], embeddings[j]);

                if (contradictionScore >= _contradictionThreshold)
                {
                    contradictions++;
                    if (contradictionExamples.Count < 2)
                    {
                        string s1 = sentences[i].Length > 40 ? sentences[i].Substring(0, 37) + "..." : sentences[i];
                        string s2 = sentences[j].Length > 40 ? sentences[j].Substring(0, 37) + "..." : sentences[j];
                        contradictionExamples.Add($"'{s1}' vs '{s2}'");
                    }
                }
            }
        }

        if (contradictions > 0 && totalPairs > 0)
        {
            double contradictionRate = (double)contradictions / totalPairs;

            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Hallucination,
                Severity = contradictions >= 3 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = Math.Min(1.0, contradictionRate * 3), // Scale up for visibility
                Description = $"Internal consistency check: {contradictions} potential contradiction(s) " +
                              $"found in {sentences.Length} sentences. " +
                              (contradictionExamples.Count > 0
                                  ? $"Examples: {string.Join("; ", contradictionExamples)}"
                                  : ""),
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private double ComputeContradictionScore(
        string sent1, string sent2, Vector<T> emb1, Vector<T> emb2)
    {
        string lower1 = sent1.ToLowerInvariant();
        string lower2 = sent2.ToLowerInvariant();

        // 1. Check for negation-based contradictions
        // If sentences are semantically similar but one has negation the other doesn't
        double embSimilarity = ComputeCosineSimilarity(emb1, emb2);

        bool sent1HasNegation = HasNegation(lower1);
        bool sent2HasNegation = HasNegation(lower2);

        double negationScore = 0;
        if (embSimilarity > 0.5 && (sent1HasNegation != sent2HasNegation))
        {
            // High similarity with different negation = potential contradiction
            negationScore = embSimilarity * 0.8;
        }

        // 2. Check for antonym pair presence
        double antonymScore = 0;
        foreach (var pair in ContradictionPairs)
        {
            string[] parts = pair.Split('|');
            if (parts.Length != 2) continue;

            bool s1HasFirst = lower1.Contains(parts[0]);
            bool s1HasSecond = lower1.Contains(parts[1]);
            bool s2HasFirst = lower2.Contains(parts[0]);
            bool s2HasSecond = lower2.Contains(parts[1]);

            if ((s1HasFirst && s2HasSecond) || (s1HasSecond && s2HasFirst))
            {
                // Sentences contain opposite terms on the same topic
                if (embSimilarity > 0.3) // Must be about the same topic
                {
                    antonymScore = Math.Max(antonymScore, 0.6);
                }
            }
        }

        // 3. Numeric contradictions (same entity, different numbers)
        double numericScore = DetectNumericContradiction(lower1, lower2);

        // Combined score
        return Math.Max(Math.Max(negationScore, antonymScore), numericScore);
    }

    private static bool HasNegation(string text)
    {
        foreach (var neg in NegationWords)
        {
            if (text.Contains(neg))
            {
                return true;
            }
        }
        return false;
    }

    private static double DetectNumericContradiction(string sent1, string sent2)
    {
        // Extract numbers from both sentences
        var nums1 = ExtractNumbers(sent1);
        var nums2 = ExtractNumbers(sent2);

        if (nums1.Count == 0 || nums2.Count == 0) return 0;

        // Check for shared context words with different numbers
        string[] words1 = sent1.Split(' ');
        string[] words2 = sent2.Split(' ');

        var contextWords1 = new HashSet<string>(words1.Where(w => w.Length > 3 && !IsNumber(w)));
        var contextWords2 = new HashSet<string>(words2.Where(w => w.Length > 3 && !IsNumber(w)));

        int sharedContext = 0;
        foreach (var w in contextWords1)
        {
            if (contextWords2.Contains(w)) sharedContext++;
        }

        if (sharedContext < 2) return 0; // Not enough shared context

        // Different numbers with shared context = potential contradiction
        bool hasConflictingNumbers = false;
        foreach (var n1 in nums1)
        {
            foreach (var n2 in nums2)
            {
                if (Math.Abs(n1 - n2) > 0.001 && Math.Abs(n1 - n2) / Math.Max(Math.Abs(n1), 1) > 0.1)
                {
                    hasConflictingNumbers = true;
                    break;
                }
            }
            if (hasConflictingNumbers) break;
        }

        return hasConflictingNumbers ? 0.5 : 0;
    }

    private static List<double> ExtractNumbers(string text)
    {
        var numbers = new List<double>();
        int i = 0;
        while (i < text.Length)
        {
            if (char.IsDigit(text[i]) || (text[i] == '-' && i + 1 < text.Length && char.IsDigit(text[i + 1])))
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

    private static bool IsNumber(string text)
    {
        return double.TryParse(text.Replace(",", ""), out _);
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
}
