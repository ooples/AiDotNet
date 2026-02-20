using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Detects potential copyright violations by measuring n-gram overlap with known copyrighted works.
/// </summary>
/// <remarks>
/// <para>
/// Computes word-level n-gram overlap between the model output and a corpus of known copyrighted
/// texts. High overlap (long verbatim sequences) indicates potential memorization or copyright
/// infringement. The detector uses sliding window analysis to identify the longest matching
/// subsequences.
/// </para>
/// <para>
/// <b>For Beginners:</b> If an AI reproduces large chunks of a book, article, or other
/// copyrighted work word-for-word, that's a copyright problem. This module checks how much
/// of the output matches known copyrighted texts by comparing sequences of words.
/// </para>
/// <para>
/// <b>Detection approach:</b>
/// 1. Extract word n-grams (n=4,5,6) from the output
/// 2. Check against indexed n-grams from known copyrighted works
/// 3. Compute overlap ratio — high overlap indicates potential infringement
/// 4. Identify longest matching subsequence as evidence
/// </para>
/// <para>
/// <b>References:</b>
/// - DE-COP: Detecting copyrighted content via paraphrased permutations (2024, arxiv:2402.09910)
/// - Copyright pre-training data filtering (2025, arxiv:2512.02047)
/// - Machine unlearning to remove memorized copyrighted content (2024, arxiv:2412.18621)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NgramCopyrightDetector<T> : TextSafetyModuleBase<T>
{
    private readonly double _threshold;
    private readonly int _minNgramLength;
    private readonly HashSet<string>[] _copyrightedNgrams; // Indexed by n-gram length
    private const int NgramOverlapAdjustment = 3; // N consecutive 4-gram matches span N+3 words (3-word overlap)

    /// <inheritdoc />
    public override string ModuleName => "NgramCopyrightDetector";

    /// <summary>
    /// Initializes a new n-gram copyright detector.
    /// </summary>
    /// <param name="copyrightedTexts">
    /// Array of copyrighted text content to check against. Each entry is the full text of a work.
    /// </param>
    /// <param name="sourceNames">
    /// Names of the copyrighted works (parallel to copyrightedTexts).
    /// </param>
    /// <param name="threshold">
    /// N-gram overlap threshold (0-1). Default: 0.3. Higher values mean more text must match.
    /// </param>
    /// <param name="minNgramLength">Minimum n-gram word length for matching. Default: 5.</param>
    public NgramCopyrightDetector(
        string[]? copyrightedTexts = null,
        string[]? sourceNames = null,
        double threshold = 0.3,
        int minNgramLength = 5)
    {
        _threshold = threshold;
        _minNgramLength = minNgramLength;
        // Build n-gram indices from copyrighted texts
        var texts = copyrightedTexts ?? Array.Empty<string>();
        _copyrightedNgrams = new HashSet<string>[3]; // For n=4,5,6
        for (int i = 0; i < 3; i++)
        {
            _copyrightedNgrams[i] = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        }

        foreach (var text in texts)
        {
            string[] words = TokenizeWords(text);
            for (int n = 4; n <= 6; n++)
            {
                for (int i = 0; i <= words.Length - n; i++)
                {
                    string ngram = string.Join(" ", words, i, n);
                    _copyrightedNgrams[n - 4].Add(ngram);
                }
            }
        }
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();

        if (string.IsNullOrWhiteSpace(text) || _copyrightedNgrams[0].Count == 0)
        {
            return findings;
        }

        string[] words = TokenizeWords(text);
        if (words.Length < _minNgramLength)
        {
            return findings;
        }

        // Check overlap for each n-gram length
        int totalMatches = 0;
        int totalNgrams = 0;
        int longestMatch = 0;
        string longestMatchText = "";

        for (int n = 4; n <= 6; n++)
        {
            int ngramIdx = n - 4;
            int matches = 0;
            int ngrams = 0;

            for (int i = 0; i <= words.Length - n; i++)
            {
                string ngram = string.Join(" ", words, i, n);
                ngrams++;
                if (_copyrightedNgrams[ngramIdx].Contains(ngram))
                {
                    matches++;
                    if (n > longestMatch)
                    {
                        longestMatch = n;
                        longestMatchText = ngram;
                    }
                }
            }

            totalMatches += matches;
            totalNgrams += ngrams;
        }

        if (totalNgrams > 0)
        {
            double overlapScore = (double)totalMatches / totalNgrams;

            if (overlapScore >= _threshold)
            {
                findings.Add(new SafetyFinding
                {
                    Category = SafetyCategory.CopyrightViolation,
                    Severity = overlapScore > 0.6 ? SafetySeverity.High : SafetySeverity.Medium,
                    Confidence = Math.Min(1.0, overlapScore),
                    Description = $"Potential copyright infringement detected: {overlapScore:P1} n-gram overlap " +
                                  $"with copyrighted material. Longest match ({longestMatch}-gram): " +
                                  $"'{longestMatchText}'.",
                    RecommendedAction = overlapScore > 0.5 ? SafetyAction.Block : SafetyAction.Warn,
                    SourceModule = ModuleName
                });
            }
        }

        // Also check for very long verbatim matches using sliding window
        int maxConsecutive = FindLongestConsecutiveMatch(words);
        if (maxConsecutive >= 8) // 8+ consecutive matching words is very suspicious
        {
            double verbatimConfidence = Math.Min(1.0, maxConsecutive / 15.0);
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.CopyrightViolation,
                Severity = maxConsecutive >= 15 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = verbatimConfidence,
                Description = $"Verbatim match detected: {maxConsecutive} consecutive words match copyrighted text.",
                RecommendedAction = maxConsecutive >= 15 ? SafetyAction.Block : SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private int FindLongestConsecutiveMatch(string[] words)
    {
        int maxConsecutive = 0;
        int current = 0;

        for (int i = 0; i <= words.Length - 4; i++)
        {
            string ngram4 = string.Join(" ", words, i, 4);
            if (_copyrightedNgrams[0].Contains(ngram4))
            {
                current++;
                if (current > maxConsecutive)
                {
                    maxConsecutive = current;
                }
            }
            else
            {
                current = 0;
            }
        }

        // Convert from count of consecutive 4-gram matches to total word count.
        // Each 4-gram spans 4 words and consecutive 4-grams overlap by 3 words,
        // so N consecutive matches represent N + 3 total words.
        return maxConsecutive > 0 ? maxConsecutive + NgramOverlapAdjustment : 0;
    }

    private static string[] TokenizeWords(string text)
    {
        return text.Split(new[] { ' ', '\t', '\n', '\r', ',', ';', ':', '(', ')', '[', ']', '{', '}', '"' },
            StringSplitOptions.RemoveEmptyEntries);
    }
}
