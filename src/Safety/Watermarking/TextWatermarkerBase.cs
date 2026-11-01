using AiDotNet.Safety.Text;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Abstract base class for keyed text watermarkers.
/// </summary>
/// <remarks>
/// <para>
/// Holds the watermark key and turns a detection test into findings. A derived watermarker implements
/// <see cref="Detect"/>, a statistical test whose z-score is approximately standard normal on text that does not
/// carry its key's watermark; the base maps z to a confidence with <see cref="WatermarkStatistics.Confidence"/> and
/// reports a finding from <see cref="ReportThreshold"/>.
/// </para>
/// <para>
/// A watermarker constructed without a key uses <see cref="WatermarkKey.Public"/>. That keeps it deterministic, so a
/// detector built the same way recognises its marks, but anyone can forge a public-key mark: an application marking
/// its own text passes a secret key.
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all text watermarkers.
/// Each watermarker type extends this and adds its own way of embedding and finding a keyed signature in text.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class TextWatermarkerBase<T> : TextSafetyModuleBase<T>, ITextWatermarker<T>, IKeyedWatermarker
{
    /// <summary>
    /// The watermark strength factor. Each watermarker documents how it uses it.
    /// </summary>
    protected readonly double WatermarkStrength;

    /// <summary>
    /// Initializes the text watermarker base.
    /// </summary>
    /// <param name="watermarkStrength">Watermark embedding strength. Default: 0.5.</param>
    /// <param name="key">The secret key; <c>null</c> uses the published <see cref="WatermarkKey.Public"/> key.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="watermarkStrength"/> is not a finite number.</exception>
    /// <exception cref="ArgumentException"><paramref name="key"/> is empty or whitespace.</exception>
    protected TextWatermarkerBase(double watermarkStrength = 0.5, string? key = null)
    {
        if (double.IsNaN(watermarkStrength) || double.IsInfinity(watermarkStrength))
            throw new ArgumentOutOfRangeException(nameof(watermarkStrength), "Watermark strength must be a finite number.");
        WatermarkStrength = watermarkStrength;
        Key = WatermarkKey.FromOptional(key);
    }

    /// <summary>The key the watermark is derived from.</summary>
    protected WatermarkKey Key { get; }

    /// <inheritdoc />
    public bool UsesPublicKey => Key.IsPublic;

    /// <summary>
    /// The detection confidence from which <see cref="EvaluateText"/> reports a finding (never at confidence 0).
    /// Default: 0.3, a z-score of 3.9 (one-sided false-positive rate 5e-5).
    /// </summary>
    protected virtual double ReportThreshold => 0.3;

    /// <summary>A short description of the watermark, used in findings (for example "Green-list token watermark").</summary>
    protected abstract string WatermarkDescription { get; }

    /// <inheritdoc />
    public abstract WatermarkResult Detect(string text);

    /// <inheritdoc />
    public virtual double DetectWatermark(string text) => Detect(text).Confidence;

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();
        var result = Detect(text);
        // A confidence of 0 is no evidence at all, so even a zero threshold does not report it.
        if (result.Confidence > 0 && result.Confidence >= ReportThreshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Watermarked,
                Severity = SafetySeverity.Info,
                Confidence = result.Confidence,
                Description = $"{WatermarkDescription} detected (confidence: {result.Confidence:F3}, z = {result.ZScore:F2}, " +
                              $"p = {result.PValue:E1}, {result.ScoredUnits} units scored).",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <summary>A text result for a z-score over <paramref name="scoredUnits"/> units.</summary>
    protected WatermarkResult TextResult(double z, int scoredUnits) =>
        WatermarkStatistics.Result(z, scoredUnits, "text", WatermarkDescription, ReportThreshold);
}
