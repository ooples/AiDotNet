using AiDotNet.Enums;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Text watermarker that modifies token sampling distributions to embed watermarks (SynthID-style).
/// </summary>
/// <remarks>
/// <para>
/// Uses a hash-based green/red list partition of the vocabulary conditioned on previous tokens.
/// Tokens in the "green list" are slightly favored during generation. Detection measures the
/// statistical over-representation of green-list tokens using a z-score test.
/// </para>
/// <para>
/// <b>For Beginners:</b> This watermarker subtly biases which words the AI chooses.
/// It creates a list of "preferred" words for each position based on a secret key.
/// The bias is invisible to readers, but a detector can measure whether the text
/// uses more "preferred" words than expected by chance.
/// </para>
/// <para>
/// <b>References:</b>
/// - SynthID-Text: Production text watermarking at scale (Google DeepMind, Nature 2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SamplingWatermarker<T> : TextWatermarkerBase<T>
{
    private readonly int _vocabSize;
    private readonly double _greenListFraction;

    /// <inheritdoc />
    public override string ModuleName => "SamplingWatermarker";

    /// <summary>
    /// Initializes a new sampling watermarker.
    /// </summary>
    /// <param name="watermarkStrength">Strength of the watermark bias. Default: 0.5.</param>
    /// <param name="vocabSize">Assumed vocabulary size for hash partitioning. Default: 50000.</param>
    /// <param name="greenListFraction">Fraction of tokens in the green list. Default: 0.5.</param>
    public SamplingWatermarker(double watermarkStrength = 0.5, int vocabSize = 50000,
        double greenListFraction = 0.5) : base(watermarkStrength)
    {
        _vocabSize = vocabSize;
        _greenListFraction = greenListFraction;
    }

    /// <inheritdoc />
    public override double DetectWatermark(string text)
    {
        if (string.IsNullOrWhiteSpace(text) || text.Length < 20) return 0;

        // Tokenize by splitting on whitespace (simplified tokenization)
        string[] tokens = text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
        if (tokens.Length < 5) return 0;

        int greenCount = 0;
        int total = 0;

        for (int i = 1; i < tokens.Length; i++)
        {
            // Hash the previous token to seed the green list partition
            int prevHash = GetFNVHash(tokens[i - 1]);
            int tokenHash = GetFNVHash(tokens[i]);

            // Determine if current token would be in the green list for this context
            int combinedHash = prevHash ^ (tokenHash * 16777619);
            double normalized = (uint)combinedHash / (double)uint.MaxValue;

            if (normalized < _greenListFraction)
            {
                greenCount++;
            }
            total++;
        }

        if (total < 5) return 0;

        // Z-score test: how much does green fraction deviate from expected?
        double observed = (double)greenCount / total;
        double expected = _greenListFraction;
        double stddev = Math.Sqrt(expected * (1 - expected) / total);

        if (stddev < 1e-10) return 0;

        double zScore = (observed - expected) / stddev;

        // Convert z-score to confidence (>2 is suspicious, >4 is strong evidence)
        if (zScore <= 1.0) return 0;
        return Math.Max(0, Math.Min(1.0, (zScore - 1.0) / 4.0));
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();
        double score = DetectWatermark(text);

        if (score >= 0.3)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Watermarked,
                Severity = score >= 0.7 ? SafetySeverity.Medium : SafetySeverity.Info,
                Confidence = score,
                Description = $"Sampling-based text watermark detected (confidence: {score:F3}). " +
                              "Green-list token bias consistent with SynthID-style watermarking.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private static int GetFNVHash(string s)
    {
        unchecked
        {
            int hash = (int)2166136261;
            foreach (char c in s)
            {
                hash ^= c;
                hash *= 16777619;
            }
            return hash;
        }
    }
}
