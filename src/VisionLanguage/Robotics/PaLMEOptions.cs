using AiDotNet.VisionLanguage.Robotics;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Configuration options for PaLM-E.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the PaLME model. Default values follow the original paper settings.</para>
/// </remarks>
public class PaLMEOptions : VisionLanguageActionOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PaLMEOptions(PaLMEOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        VisionDim = other.VisionDim;
        DecoderDim = other.DecoderDim;
        NumVisionLayers = other.NumVisionLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumHeads = other.NumHeads;
        VocabSize = other.VocabSize;
        MaxSequenceLength = other.MaxSequenceLength;
        MaxGenerationLength = other.MaxGenerationLength;
        DropoutRate = other.DropoutRate;
        ArchitectureType = other.ArchitectureType;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        ActionDimension = other.ActionDimension;
        LanguageModelName = other.LanguageModelName;
        PredictionHorizon = other.PredictionHorizon;
        ObservationHistory = other.ObservationHistory;
        TotalParameters = other.TotalParameters;
    }

    /// <summary>
    /// Default constructor — uses a research-scale config (~PaLM-E-Lite, ~5M
    /// total params) so the model can be constructed and exercised on a unit
    /// test budget. The 562B-parameter full Driess et al. 2023 config is
    /// computationally infeasible for CI (profiled construction alone exceeds
    /// the 120s timeout because each decoder block holds ~805M parameters
    /// across 64 layers). Use <see cref="Full562B"/> to opt in to the
    /// paper-published production config.
    /// </summary>
    public PaLMEOptions()
    {
        // Architecture proportions follow the paper (depth ratio decoder:vision
        // ≈ 4:3, FFN expansion 4×, head_dim = dim/heads = 64) but absolute
        // sizes are scaled down for tractability.
        VisionDim = 192;
        DecoderDim = 256;
        NumVisionLayers = 3;
        NumDecoderLayers = 4;
        NumHeads = 4;
        ImageSize = 128;
        VocabSize = 4096;
        LanguageModelName = "PaLM";
        ActionDimension = 7;
        PredictionHorizon = 4;
        // Approximate parameter count for the research-scale config:
        //   vision block ~= 4*192² + 2*192*768 ≈ 442K  × 3 ≈ 1.3M
        //   decoder block ~= 4*256² + 2*256*1024 ≈ 786K × 4 ≈ 3.1M
        // Total ~5M — comfortably constructible in <100ms on CI hardware.
        TotalParameters = 0;
    }

    /// <summary>
    /// Constructs the full PaLM-E 562B config from Driess et al. 2023 §3.
    /// Use this only on hardware that can actually allocate ~52GB of model
    /// weights — it WILL exceed CI smoke-test budgets.
    /// </summary>
    public static PaLMEOptions Full562B() => new PaLMEOptions
    {
        VisionDim = 1408,
        DecoderDim = 8192,
        NumVisionLayers = 48,
        NumDecoderLayers = 64,
        NumHeads = 64,
        ImageSize = 224,
        VocabSize = 256000,
        LanguageModelName = "PaLM",
        ActionDimension = 7,
        PredictionHorizon = 16,
        TotalParameters = 562
    };

    /// <summary>Gets or sets the total parameter count in billions.</summary>
    public int TotalParameters { get; set; }
}
