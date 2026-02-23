using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Detects gradient-based adversarial jailbreak attacks by analyzing token-level anomalies
/// in the input text.
/// </summary>
/// <remarks>
/// <para>
/// Gradient-based attacks (GCG, AutoDAN, etc.) optimize adversarial suffixes to trigger
/// unsafe behavior. These suffixes typically exhibit statistical anomalies detectable without
/// model gradients: unusual character distributions, high entropy token sequences, abnormal
/// bigram frequencies, and nonsensical subword patterns. This detector identifies such
/// adversarial artifacts.
/// </para>
/// <para>
/// <b>For Beginners:</b> Some attackers use mathematical optimization to craft special text
/// that tricks AI safety filters. This text often looks like random gibberish appended to a
/// normal prompt. This module detects such gibberish suffixes by checking whether parts of
/// the text have unusual character patterns.
/// </para>
/// <para>
/// <b>Detection signals:</b>
/// 1. Character distribution anomaly — adversarial tokens have non-English character frequencies
/// 2. Bigram entropy — random-looking token sequences have unusually high bigram entropy
/// 3. Subword coherence — GCG tokens are often nonsensical fragments
/// 4. Suffix length anomaly — adversarial suffixes are unusually long appended sequences
/// </para>
/// <para>
/// <b>References:</b>
/// - GradSafe: Detecting unsafe inputs via safety-critical gradient analysis (2024, arxiv:2402.13494)
/// - GCG: Universal and transferable adversarial attacks on aligned LMs (2023, arxiv:2307.15043)
/// - SmoothLLM: Defending LLMs against jailbreaking via randomized smoothing (2024, arxiv:2310.03684)
/// - Perplexity-based jailbreak detection (Jain et al., 2023)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GradientJailbreakDetector<T> : TextSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _anomalyThreshold;
    private readonly int _minSuffixLength;

    // Expected English character frequencies (approximate, lowercase + space)
    private static readonly Dictionary<char, double> EnglishFrequencies = new()
    {
        { ' ', 0.180 }, { 'e', 0.111 }, { 't', 0.079 }, { 'a', 0.072 },
        { 'o', 0.066 }, { 'i', 0.063 }, { 'n', 0.061 }, { 's', 0.056 },
        { 'h', 0.050 }, { 'r', 0.050 }, { 'd', 0.037 }, { 'l', 0.035 },
        { 'c', 0.024 }, { 'u', 0.024 }, { 'm', 0.021 }, { 'w', 0.020 },
        { 'f', 0.019 }, { 'g', 0.017 }, { 'y', 0.017 }, { 'p', 0.015 },
        { 'b', 0.013 }, { 'v', 0.009 }, { 'k', 0.007 }, { 'j', 0.002 },
        { 'x', 0.001 }, { 'q', 0.001 }, { 'z', 0.001 }
    };

    /// <inheritdoc />
    public override string ModuleName => "GradientJailbreakDetector";

    /// <summary>
    /// Initializes a new gradient-based jailbreak detector.
    /// </summary>
    /// <param name="anomalyThreshold">
    /// Anomaly score threshold (0-1). Default: 0.5.
    /// </param>
    /// <param name="minSuffixLength">
    /// Minimum suffix length (chars) to analyze for adversarial patterns. Default: 20.
    /// </param>
    public GradientJailbreakDetector(
        double anomalyThreshold = 0.5,
        int minSuffixLength = 20)
    {
        _anomalyThreshold = anomalyThreshold;
        _minSuffixLength = minSuffixLength;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        if (string.IsNullOrWhiteSpace(text) || text.Length < _minSuffixLength)
        {
            return findings;
        }

        // Analyze the full text for adversarial token patterns
        double charAnomalyScore = ComputeCharDistributionAnomaly(text);
        double bigramEntropyScore = ComputeBigramEntropyAnomaly(text);
        double coherenceScore = ComputeSubwordCoherenceAnomaly(text);

        // Also analyze the latter portion (adversarial suffixes are typically appended)
        int suffixStart = text.Length * 2 / 3; // Last third of text
        string suffix = text.Substring(suffixStart);
        double suffixAnomalyScore = 0;

        if (suffix.Length >= _minSuffixLength)
        {
            double suffCharAnomaly = ComputeCharDistributionAnomaly(suffix);
            double suffBigramAnomaly = ComputeBigramEntropyAnomaly(suffix);
            double suffCoherenceAnomaly = ComputeSubwordCoherenceAnomaly(suffix);

            suffixAnomalyScore = 0.35 * suffCharAnomaly + 0.35 * suffBigramAnomaly + 0.30 * suffCoherenceAnomaly;
        }

        // Full text score
        double fullTextScore = 0.30 * charAnomalyScore + 0.35 * bigramEntropyScore + 0.35 * coherenceScore;

        // Use max of full-text and suffix analysis
        double finalScore = Math.Max(fullTextScore, suffixAnomalyScore);

        if (finalScore >= _anomalyThreshold)
        {
            bool suffixIsPrimary = suffixAnomalyScore > fullTextScore;

            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.JailbreakAttempt,
                Severity = finalScore > 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = Math.Min(1.0, finalScore),
                Description = $"Gradient-based adversarial pattern detected (score: {finalScore:F3}). " +
                              $"Character anomaly: {charAnomalyScore:F3}, " +
                              $"bigram entropy: {bigramEntropyScore:F3}, " +
                              $"subword coherence: {coherenceScore:F3}" +
                              (suffixIsPrimary
                                  ? $". Anomaly concentrated in suffix (suffix score: {suffixAnomalyScore:F3})."
                                  : "."),
                RecommendedAction = finalScore > 0.8 ? SafetyAction.Block : SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <summary>
    /// Measures how much the character frequency distribution deviates from English.
    /// Uses Jensen-Shannon divergence between observed and expected distributions.
    /// </summary>
    private static double ComputeCharDistributionAnomaly(string text)
    {
        string lower = text.ToLowerInvariant();
        var charCounts = new Dictionary<char, int>();
        int total = 0;

        foreach (char c in lower)
        {
            if (char.IsLetterOrDigit(c) || c == ' ')
            {
                charCounts.TryGetValue(c, out int count);
                charCounts[c] = count + 1;
                total++;
            }
        }

        if (total < 10) return 0;

        // Compute JSD between observed and expected English distribution
        double jsd = 0;
        var allChars = new HashSet<char>(EnglishFrequencies.Keys);
        foreach (var c in charCounts.Keys) allChars.Add(c);

        foreach (char c in allChars)
        {
            charCounts.TryGetValue(c, out int count);
            double p = (double)count / total; // observed
            EnglishFrequencies.TryGetValue(c, out double q); // expected

            // Smooth both distributions
            p = p * 0.99 + 0.01 / allChars.Count;
            q = q > 0 ? q * 0.99 + 0.01 / allChars.Count : 0.01 / allChars.Count;

            double m = (p + q) / 2;
            if (p > 0 && m > 0) jsd += p * Math.Log(p / m, 2);
            if (q > 0 && m > 0) jsd += q * Math.Log(q / m, 2);
        }
        jsd /= 2; // JSD = (KL(P||M) + KL(Q||M)) / 2

        // Normalize: JSD ∈ [0, 1] for base-2 log
        // Map to [0, 1] anomaly score; typical English JSD < 0.05
        return Math.Min(1.0, jsd * 5);
    }

    /// <summary>
    /// Measures character bigram entropy. Adversarial tokens have higher entropy
    /// (more random bigram transitions) than natural text.
    /// </summary>
    private static double ComputeBigramEntropyAnomaly(string text)
    {
        string lower = text.ToLowerInvariant();
        var bigramCounts = new Dictionary<string, int>();
        int total = 0;

        for (int i = 0; i < lower.Length - 1; i++)
        {
            char c1 = lower[i];
            char c2 = lower[i + 1];
            if (!char.IsLetterOrDigit(c1) && c1 != ' ') continue;
            if (!char.IsLetterOrDigit(c2) && c2 != ' ') continue;

            string bigram = new string(new[] { c1, c2 });
            bigramCounts.TryGetValue(bigram, out int count);
            bigramCounts[bigram] = count + 1;
            total++;
        }

        if (total < 10) return 0;

        // Compute entropy of bigram distribution
        double entropy = 0;
        foreach (var count in bigramCounts.Values)
        {
            double p = (double)count / total;
            if (p > 0) entropy -= p * Math.Log(p, 2);
        }

        // Typical English bigram entropy is ~6-8 bits
        // GCG-style adversarial text has entropy ~9-11 bits
        // Map: <7 = 0 (normal), 7-11 = linear, >11 = 1 (adversarial)
        double anomaly = (entropy - 7.0) / 4.0;
        return Math.Max(0, Math.Min(1.0, anomaly));
    }

    /// <summary>
    /// Measures subword coherence. Natural text has mostly recognizable words;
    /// adversarial tokens often produce nonsensical character sequences.
    /// </summary>
    private static double ComputeSubwordCoherenceAnomaly(string text)
    {
        // Split into "words" and check what fraction are recognizable
        string[] tokens = text.Split(new[] { ' ', '\t', '\n', '\r' },
            StringSplitOptions.RemoveEmptyEntries);

        if (tokens.Length == 0) return 0;

        int nonsensicalCount = 0;

        foreach (var token in tokens)
        {
            if (IsNonsensicalToken(token))
            {
                nonsensicalCount++;
            }
        }

        double nonsensicalRatio = (double)nonsensicalCount / tokens.Length;

        // Typical text has < 5% nonsensical tokens; adversarial can have > 30%
        double anomaly = (nonsensicalRatio - 0.05) / 0.25;
        return Math.Max(0, Math.Min(1.0, anomaly));
    }

    /// <summary>
    /// Heuristic check for nonsensical tokens typical of gradient-optimized adversarial text.
    /// </summary>
    private static bool IsNonsensicalToken(string token)
    {
        if (token.Length < 2) return false;

        string cleaned = token.TrimEnd('.', ',', '!', '?', ';', ':');
        if (cleaned.Length < 2) return false;

        // Check 1: High ratio of non-alphabetic characters in what should be a word
        int nonAlpha = 0;
        foreach (char c in cleaned)
        {
            if (!char.IsLetter(c)) nonAlpha++;
        }
        if (cleaned.Length > 3 && (double)nonAlpha / cleaned.Length > 0.5) return true;

        // Check 2: Consonant cluster length > 4 (very rare in English)
        int maxConsonantCluster = 0;
        int currentCluster = 0;
        string vowels = "aeiouAEIOU";
        foreach (char c in cleaned)
        {
            if (char.IsLetter(c) && vowels.IndexOf(c) < 0)
            {
                currentCluster++;
                if (currentCluster > maxConsonantCluster) maxConsonantCluster = currentCluster;
            }
            else
            {
                currentCluster = 0;
            }
        }
        if (maxConsonantCluster > 4) return true;

        // Check 3: No vowels in a word > 3 characters
        if (cleaned.Length > 3)
        {
            bool hasVowel = false;
            foreach (char c in cleaned)
            {
                if (vowels.IndexOf(c) >= 0) { hasVowel = true; break; }
            }
            if (!hasVowel) return true;
        }

        // Check 4: Excessive character repetition (e.g., "aaaa" or "!!!!")
        int maxRepeat = 1;
        int currentRepeat = 1;
        for (int i = 1; i < cleaned.Length; i++)
        {
            if (cleaned[i] == cleaned[i - 1])
            {
                currentRepeat++;
                if (currentRepeat > maxRepeat) maxRepeat = currentRepeat;
            }
            else
            {
                currentRepeat = 1;
            }
        }
        if (maxRepeat >= 4) return true;

        // Check 5: Mixed case within a single token (e.g., "AbCdEf")
        if (cleaned.Length > 3)
        {
            int caseChanges = 0;
            for (int i = 1; i < cleaned.Length; i++)
            {
                if (char.IsLetter(cleaned[i]) && char.IsLetter(cleaned[i - 1]) &&
                    char.IsUpper(cleaned[i]) != char.IsUpper(cleaned[i - 1]))
                {
                    caseChanges++;
                }
            }
            // More than 3 case changes in a word is suspicious (camelCase is 1-2)
            if (caseChanges > 3) return true;
        }

        return false;
    }
}
