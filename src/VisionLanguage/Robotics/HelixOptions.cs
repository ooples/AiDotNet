using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Robotics;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Configuration options for Helix.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Helix model. Default values follow the original paper settings.</para>
/// </remarks>
public class HelixOptions : VisionLanguageActionOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public HelixOptions(HelixOptions other)
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
        NumJoints = other.NumJoints;
        System2LatentDim = other.System2LatentDim;
        System1HiddenDim = other.System1HiddenDim;
        System1NumLayers = other.System1NumLayers;
        System1NumHeads = other.System1NumHeads;
        System1ToSystem2Ratio = other.System1ToSystem2Ratio;
        WeightOffloadOptions = other.WeightOffloadOptions;
    }

    public HelixOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "LLaMA";
        ActionDimension = 35;
        PredictionHorizon = 16;
    }

    /// <summary>Number of joint DOFs the upper-body controller exposes. Paper §3.4: 35 (torso 3 + arms 7×2 + hands 8×2 + neck 2).</summary>
    public int NumJoints { get; set; } = 35;

    /// <summary>Dimensionality of the latent vector System 2 emits to condition System 1. Paper §3.2: 512.</summary>
    public int System2LatentDim { get; set; } = 512;

    /// <summary>Hidden dimension of the System 1 fast visuomotor transformer. Paper §3.3: 384 (yields ~80M params with 8 layers).</summary>
    public int System1HiddenDim { get; set; } = 384;

    /// <summary>Number of transformer blocks in System 1. Paper §3.3: 8.</summary>
    public int System1NumLayers { get; set; } = 8;

    /// <summary>Number of attention heads per System 1 transformer block. Paper §3.3: 6.</summary>
    public int System1NumHeads { get; set; } = 6;

    /// <summary>
    /// How many S1 ticks one S2 invocation remains valid before the runner re-invokes S2.
    /// Default 22 — paper §4.1's S1:S2 = 200 Hz : ~9 Hz rate ratio.
    /// </summary>
    public int System1ToSystem2Ratio { get; set; } = 22;

    /// <summary>
    /// Optional weight-offload / streaming configuration. When non-null, the Helix
    /// constructor calls <c>ConfigureWeightLifetime</c> so the ~6.7B paper-scale
    /// weights are streamed (disk-backed or pinned-host) instead of held fully
    /// resident — the same contract as <c>PaLMEOptions.WeightOffloadOptions</c>.
    /// Null (default) keeps weights resident, matching the original paper-faithful
    /// in-memory behaviour for callers with enough RAM.
    /// </summary>
    public GpuOffloadOptions? WeightOffloadOptions { get; set; }
}
