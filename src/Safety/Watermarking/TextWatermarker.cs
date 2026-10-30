using AiDotNet.Agentic.Models.Local;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// The safety pipeline's text watermarker: the green-list watermark of <see cref="SamplingWatermarker{T}"/>
/// (Kirchenbauer et al. 2023) with a z-score reporting threshold.
/// </summary>
/// <remarks>
/// <para>
/// Embed by wrapping the generating model with <see cref="SamplingWatermarker{T}.WatermarkModel"/> (or calling
/// <see cref="SamplingWatermarker{T}.BiasLogits"/> in your decode loop); detect with
/// <see cref="TextWatermarkerBase{T}.EvaluateText"/>, which re-tokenizes the text with the generator's tokenizer and
/// reports a finding when the green-token z-score reaches <c>detectionThreshold</c>.
/// </para>
/// <para>
/// The watermark is derived from the key, never from a random value: two instances built with the same key and
/// settings embed and detect the same mark. Without a key the published <see cref="WatermarkKey.Public"/> key is
/// used, whose marks anyone can forge; mark your own text with a secret key.
/// </para>
/// <para>
/// <b>For Beginners:</b> Text watermarking works by subtly biasing which words the AI chooses.
/// When generating text, the system slightly prefers words from a secret list that changes at every position. Later,
/// the system can check whether a piece of text came from it by counting how many listed words appear.
/// </para>
/// <para>
/// <b>References:</b>
/// - Kirchenbauer et al., A Watermark for Large Language Models (ICML 2023, arxiv:2301.10226)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Statistical)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Watermark for Large Language Models",
    "https://arxiv.org/abs/2301.10226",
    Year = 2023,
    Authors = "John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein")]
public class TextWatermarker<T> : SamplingWatermarker<T>
{
    private readonly double _reportThreshold;

    /// <inheritdoc />
    public override string ModuleName => "TextWatermarker";

    /// <summary>
    /// Initializes a new text watermarker.
    /// </summary>
    /// <param name="key">
    /// The secret key the green lists are derived from. <c>null</c> uses the published <see cref="WatermarkKey.Public"/>
    /// key, whose marks anyone can forge; use a long random key of your own to mark your text.
    /// </param>
    /// <param name="greenListFraction">The green-list fraction gamma, in (0, 1). Default: 0.25.</param>
    /// <param name="detectionThreshold">
    /// The z-score from which a finding is reported. Default: 4.0 (one-sided false-positive rate 3.2e-5), the
    /// threshold Kirchenbauer et al. use. Values at or below 3 report any evidence (confidence above 0).
    /// </param>
    /// <param name="contextWidth">The number of preceding tokens hashed into the green-list seed. Default: 1.</param>
    /// <param name="tokenizer">The generator's tokenizer, used to score text; <c>null</c> scores only token ids.</param>
    /// <param name="watermarkStrength">Logit bias strength: delta = 4 x strength. Default: 0.5 (delta = 2.0).</param>
    public TextWatermarker(
        string? key = null,
        double greenListFraction = 0.25,
        double detectionThreshold = 4.0,
        int contextWidth = 1,
        IGenerationTokenizer? tokenizer = null,
        double watermarkStrength = 0.5)
        : base(watermarkStrength, null, greenListFraction, key, tokenizer, contextWidth)
    {
        if (double.IsNaN(detectionThreshold) || double.IsInfinity(detectionThreshold))
            throw new ArgumentOutOfRangeException(nameof(detectionThreshold), "The detection threshold must be a finite z-score.");
        _reportThreshold = WatermarkStatistics.Confidence(detectionThreshold);
    }

    /// <inheritdoc />
    protected override double ReportThreshold => _reportThreshold;
}
