using AiDotNet.Agentic.Models.Local;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Composite text watermark detector: runs the green-list, synonym and contraction detectors of one key and reports
/// the strongest evidence, corrected for having run three tests.
/// </summary>
/// <remarks>
/// <para>
/// Each detector's z-score is approximately standard normal on text without its watermark. Taking the best of three
/// would triple the false-positive rate, so the smallest p-value is multiplied by three (a Bonferroni correction) and
/// turned back into a z-score before the usual confidence mapping. A false-positive rate stated for one detector
/// therefore holds for the composite too.
/// </para>
/// <para>
/// The green-list detector can only score text when the generator's tokenizer is supplied; without it the composite
/// still checks the two post-hoc watermarks.
/// </para>
/// <para>
/// <b>For Beginners:</b> This detector checks for all of this library's text watermarks made with your key at once,
/// without becoming more trigger-happy than any single check.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Statistical)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Watermark for Large Language Models",
    "https://arxiv.org/abs/2301.10226",
    Year = 2023,
    Authors = "John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein")]
public partial class WatermarkDetector<T> : TextWatermarkerBase<T>
{
    private readonly ITextWatermarker<T>[] _detectors;

    /// <inheritdoc />
    public override string ModuleName => "WatermarkDetector";

    /// <inheritdoc />
    protected override string WatermarkDescription => "Text watermark (composite detector)";

    /// <summary>
    /// Initializes a new composite watermark detector.
    /// </summary>
    /// <param name="key">The secret key shared by the three detectors; <c>null</c> uses the published key.</param>
    /// <param name="tokenizer">The generator's tokenizer, needed to score the green-list watermark in text.</param>
    public WatermarkDetector(string? key = null, IGenerationTokenizer? tokenizer = null) : base(0.5, key)
    {
        _detectors = new ITextWatermarker<T>[]
        {
            new SamplingWatermarker<T>(key: key, tokenizer: tokenizer),
            new LexicalWatermarker<T>(key: key),
            new SyntacticWatermarker<T>(key: key)
        };
    }

    /// <inheritdoc />
    public override WatermarkResult Detect(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return TextResult(0, 0);
        double bestZ = double.NegativeInfinity;
        int units = 0;
        foreach (var detector in _detectors)
        {
            var result = detector.Detect(text);
            units += result.ScoredUnits;
            if (result.ScoredUnits > 0 && result.ZScore > bestZ) bestZ = result.ZScore;
        }

        if (double.IsNegativeInfinity(bestZ)) return TextResult(0, 0);
        return TextResult(WatermarkStatistics.BonferroniZ(bestZ, _detectors.Length), units);
    }
}
