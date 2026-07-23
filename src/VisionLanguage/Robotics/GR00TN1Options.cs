using AiDotNet.VisionLanguage.Robotics;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Configuration options for GR00T N1.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GR00TN1 model. Default values follow the original paper settings.</para>
/// </remarks>
public class GR00TN1Options : VisionLanguageActionOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GR00TN1Options(GR00TN1Options other)
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
        FlowMatchingSteps = other.FlowMatchingSteps;
    }

    public GR00TN1Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
        LanguageModelName = "Eagle-2";
        ActionDimension = 52;
        PredictionHorizon = 16;
    }

    /// <summary>Number of humanoid joints. Paper §3.4: 52 whole-body (arms 7×2 + hands 12×2 + torso 3 + legs 6×2 + neck 2).</summary>
    public int NumJoints { get; set; } = 52;

    /// <summary>Dimensionality of the latent vector System 2 (Eagle-2 VLM) emits to condition the flow-matching head. Paper §3.1: 1536.</summary>
    public int System2LatentDim { get; set; } = 1536;

    /// <summary>Hidden dimension of the System-1 DiT velocity transformer. Paper §3.2: 1024 (yields ~280M params with 12 layers).</summary>
    public int System1HiddenDim { get; set; } = 1024;

    /// <summary>Number of DiT transformer blocks in System 1. Paper §3.2: 12.</summary>
    public int System1NumLayers { get; set; } = 12;

    /// <summary>Number of attention heads per System-1 DiT block. Paper §3.2: 16.</summary>
    public int System1NumHeads { get; set; } = 16;

    /// <summary>How many S1 ticks an S2 latent remains valid in streaming mode. Paper §4.1: S1 @ 50 Hz, S2 @ ~10 Hz → 5:1.</summary>
    public int System1ToSystem2Ratio { get; set; } = 5;

    /// <summary>Number of Euler integration steps the flow-matching action head runs at inference. Paper §4.1 default: 16.</summary>
    public int FlowMatchingSteps { get; set; } = 16;
}
