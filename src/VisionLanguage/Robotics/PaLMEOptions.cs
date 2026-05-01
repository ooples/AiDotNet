using AiDotNet.Tensors.LinearAlgebra;
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
        WeightOffloadOptions = other.WeightOffloadOptions;
    }

    public PaLMEOptions()
    {
        VisionDim = 1408;
        DecoderDim = 8192;
        NumVisionLayers = 48;
        NumDecoderLayers = 64;
        NumHeads = 64;
        ImageSize = 224;
        VocabSize = 256000;
        LanguageModelName = "PaLM";
        ActionDimension = 7;
        PredictionHorizon = 16;
    }

    /// <summary>Gets or sets the total parameter count in billions.</summary>
    public int TotalParameters { get; set; } = 562;

    /// <summary>
    /// Optional weight-lifetime / GPU-offload configuration applied during
    /// native-mode construction. PaLM-E is 562B parameters at fp64 ⇒ ~140 GB
    /// resident, so the constructor opts into streaming offload by default;
    /// supply your own instance to override the resident-bytes ceiling, the
    /// scheme, or the backing-store path. Set to <c>null</c> to skip the
    /// automatic call to <c>ConfigureWeightLifetime</c>.
    /// </summary>
    public GpuOffloadOptions? WeightOffloadOptions { get; set; }
}
