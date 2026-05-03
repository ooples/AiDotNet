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
    /// native-mode construction. PaLM-E is 562B parameters: raw weights are
    /// ~4.5 TB at fp64 (562e9 × 8 B), ~2.25 TB at fp32, ~1.13 TB at fp16,
    /// excluding activations / optimizer state. So callers running the
    /// model in native mode at full size should set this property — the
    /// PaLM-E constructor wires it through to the per-network weight
    /// registry to enable streaming offload. Leaving this at <c>null</c>
    /// keeps the model fully resident in RAM, which is fine for
    /// research-scale parameter counts but will OOM at 562B.
    /// </summary>
    /// <remarks>
    /// This is the supported public entry point for weight-offload
    /// configuration on a PaLM-E instance. The lower-level
    /// <c>ConfigureWeightLifetime</c> hook on <c>NeuralNetworkBase</c> is
    /// internal because it mutates a process-wide singleton
    /// (<c>WeightRegistry</c>) and would surprise consumers who don't
    /// know about the cross-instance side effect. Setting
    /// <see cref="WeightOffloadOptions"/> here is the contract — the
    /// PaLM-E constructor handles the registry plumbing.
    /// </remarks>
    public GpuOffloadOptions? WeightOffloadOptions { get; set; }
}
