using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Detects potential training data memorization by estimating text perplexity.
/// Low perplexity (highly predictable) text may indicate verbatim memorized content.
/// </summary>
/// <remarks>
/// <para>
/// Memorized training data typically has unusually low perplexity — the model is very
/// "confident" about every next token because it has seen the exact sequence before.
/// This detector computes a lightweight character-level perplexity proxy using n-gram
/// statistics and flags outputs that are suspiciously predictable compared to natural text.
/// </para>
/// <para>
/// <b>For Beginners:</b> If an AI can predict every next word perfectly, it's probably
/// reciting something it memorized during training. Normal text has some unpredictability.
/// This module measures how predictable the text is — too predictable means it might be
/// a memorized passage from training data.
/// </para>
/// <para>
/// <b>Detection approach:</b>
/// 1. Build character-level n-gram frequency table from the text itself
/// 2. Compute conditional entropy at each position using n-gram context
/// 3. Average entropy = proxy for perplexity
/// 4. Very low entropy indicates memorized/formulaic text
/// 5. Additional checks: token repetition patterns, entropy variance
/// </para>
/// <para>
/// <b>References:</b>
/// - Detecting pre-training data via perplexity comparison (Carlini et al., 2023)
/// - Min-K%/Min-K%++ membership inference (2024, arxiv:2404.02936)
/// - BookMIA: Book-level membership inference (2024, arxiv:2401.15588)
/// - Scalable extraction of training data from LLMs (2023, arxiv:2311.17035)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PerplexityMemorizationDetector<T> : TextSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _lowPerplexityThreshold;
    private readonly double _highRepetitionThreshold;
    private readonly int _ngramOrder;

    /// <inheritdoc />
    public override string ModuleName => "PerplexityMemorizationDetector";

    /// <summary>
    /// Initializes a new perplexity-based memorization detector.
    /// </summary>
    /// <param name="lowPerplexityThreshold">
    /// Entropy threshold below which text is flagged as memorized (bits per char). Default: 2.0.
    /// Natural English typically has 3-5 bits/char; memorized text often &lt; 2.
    /// </param>
    /// <param name="highRepetitionThreshold">
    /// Token repetition ratio above which text is flagged. Default: 0.4.
    /// </param>
    /// <param name="ngramOrder">Character n-gram order for entropy estimation. Default: 4.</param>
    public PerplexityMemorizationDetector(
        double lowPerplexityThreshold = 2.0,
        double highRepetitionThreshold = 0.4,
        int ngramOrder = 4)
    {
        _lowPerplexityThreshold = lowPerplexityThreshold;
        _highRepetitionThreshold = highRepetitionThreshold;
        _ngramOrder = ngramOrder;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        if (string.IsNullOrWhiteSpace(text) || text.Length < 50)
        {
            return findings; // Need sufficient text for statistical analysis
        }

        string normalized = text.ToLowerInvariant();

        // 1. Compute character-level conditional entropy (perplexity proxy)
        double entropy = ComputeCharacterEntropy(normalized);

        // 2. Compute word-level repetition statistics
        var repetitionStats = ComputeRepetitionStats(normalized);

        // 3. Compute entropy variance across sliding windows
        double entropyVariance = ComputeEntropyVariance(normalized);

        // 4. Evaluate memorization signals

        // Low overall entropy → text is very predictable
        if (entropy < _lowPerplexityThreshold && entropy > 0)
        {
            double confidence = 1.0 - (entropy / _lowPerplexityThreshold);
            confidence = Math.Max(0, Math.Min(1.0, confidence));

            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.TrainingDataLeakage,
                Severity = entropy < 1.0 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = confidence,
                Description = $"Low character entropy detected: {entropy:F2} bits/char " +
                              $"(threshold: {_lowPerplexityThreshold:F1}). " +
                              $"Natural text typically has 3-5 bits/char. " +
                              $"Low entropy suggests highly predictable (potentially memorized) content.",
                RecommendedAction = entropy < 1.0 ? SafetyAction.Block : SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        // High word repetition → formulaic or memorized patterns
        if (repetitionStats.RepetitionRatio > _highRepetitionThreshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.TrainingDataLeakage,
                Severity = SafetySeverity.Medium,
                Confidence = Math.Min(1.0, repetitionStats.RepetitionRatio),
                Description = $"High word repetition: {repetitionStats.RepetitionRatio:P1} of words are repeats " +
                              $"(most repeated: '{repetitionStats.MostRepeatedWord}' x{repetitionStats.MaxRepetitionCount}). " +
                              $"High repetition can indicate formulaic memorized content.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        // Very low entropy variance → uniformly predictable (memorized passage)
        // Natural text has varying predictability; memorized text is uniformly low
        if (entropy < 3.0 && entropyVariance < 0.3 && text.Length > 200)
        {
            double uniformConfidence = Math.Max(0, 1.0 - entropyVariance / 0.3);
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.TrainingDataLeakage,
                Severity = SafetySeverity.Medium,
                Confidence = uniformConfidence * 0.7,
                Description = $"Uniformly low entropy: variance={entropyVariance:F3} with mean={entropy:F2} bits/char. " +
                              $"Natural text has varying predictability; uniform low entropy suggests " +
                              $"a memorized passage from training data.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private double ComputeCharacterEntropy(string text)
    {
        // Build n-gram frequency table
        var ngramCounts = new Dictionary<string, Dictionary<char, int>>();
        var contextCounts = new Dictionary<string, int>();

        for (int i = _ngramOrder; i < text.Length; i++)
        {
            string context = text.Substring(i - _ngramOrder, _ngramOrder);
            char next = text[i];

            if (!ngramCounts.TryGetValue(context, out var nextCounts))
            {
                nextCounts = new Dictionary<char, int>();
                ngramCounts[context] = nextCounts;
            }

            nextCounts.TryGetValue(next, out int count);
            nextCounts[next] = count + 1;

            contextCounts.TryGetValue(context, out int ctxCount);
            contextCounts[context] = ctxCount + 1;
        }

        // Compute conditional entropy H(X_n | X_{n-k}...X_{n-1})
        double totalEntropy = 0;
        int totalPositions = 0;

        for (int i = _ngramOrder; i < text.Length; i++)
        {
            string context = text.Substring(i - _ngramOrder, _ngramOrder);

            if (!ngramCounts.TryGetValue(context, out var nextCounts)) continue;

            int total = contextCounts[context];
            char actual = text[i];

            nextCounts.TryGetValue(actual, out int charCount);
            if (charCount > 0 && total > 0)
            {
                double prob = (double)charCount / total;
                totalEntropy += -Math.Log(prob, 2);
                totalPositions++;
            }
        }

        return totalPositions > 0 ? totalEntropy / totalPositions : 5.0; // Default high entropy
    }

    private double ComputeEntropyVariance(string text)
    {
        int windowSize = 100;
        int stride = 50;

        if (text.Length < windowSize + _ngramOrder)
        {
            return 1.0; // Not enough data; assume high variance (non-memorized)
        }

        var windowEntropies = new List<double>();

        for (int start = 0; start + windowSize <= text.Length; start += stride)
        {
            string window = text.Substring(start, windowSize);
            double windowEntropy = ComputeCharacterEntropy(window);
            windowEntropies.Add(windowEntropy);
        }

        if (windowEntropies.Count < 2) return 1.0;

        // Compute variance
        double mean = 0;
        foreach (var e in windowEntropies) mean += e;
        mean /= windowEntropies.Count;

        double variance = 0;
        foreach (var e in windowEntropies)
        {
            double diff = e - mean;
            variance += diff * diff;
        }
        variance /= windowEntropies.Count;

        return variance;
    }

    private static RepetitionInfo ComputeRepetitionStats(string text)
    {
        string[] words = text.Split(new[] { ' ', '\t', '\n', '\r', ',', ';', ':', '(', ')', '[', ']' },
            StringSplitOptions.RemoveEmptyEntries);

        if (words.Length == 0)
        {
            return new RepetitionInfo { RepetitionRatio = 0, MaxRepetitionCount = 0, MostRepeatedWord = "" };
        }

        var wordCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var word in words)
        {
            string cleaned = word.TrimEnd('.', '!', '?', '\'', '"');
            if (cleaned.Length < 2) continue;

            wordCounts.TryGetValue(cleaned, out int count);
            wordCounts[cleaned] = count + 1;
        }

        int totalWords = 0;
        int repeatedWords = 0;
        int maxCount = 0;
        string mostRepeated = "";

        foreach (var kvp in wordCounts)
        {
            totalWords += kvp.Value;
            if (kvp.Value > 1)
            {
                repeatedWords += kvp.Value - 1; // Count excess occurrences
            }
            if (kvp.Value > maxCount)
            {
                maxCount = kvp.Value;
                mostRepeated = kvp.Key;
            }
        }

        return new RepetitionInfo
        {
            RepetitionRatio = totalWords > 0 ? (double)repeatedWords / totalWords : 0,
            MaxRepetitionCount = maxCount,
            MostRepeatedWord = mostRepeated
        };
    }

    private struct RepetitionInfo
    {
        public double RepetitionRatio;
        public int MaxRepetitionCount;
        public string MostRepeatedWord;
    }
}
