using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for DeepSeek-VL2 (MoE, dynamic tiling, multi-head latent attention).</summary>
public class DeepSeekVL2Options : InstructionTunedVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DeepSeekVL2Options(DeepSeekVL2Options other)
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
        InstructionArchitectureType = other.InstructionArchitectureType;
        ProjectionDim = other.ProjectionDim;
        LanguageModelName = other.LanguageModelName;
        MaxVisualTokens = other.MaxVisualTokens;
        SystemPrompt = other.SystemPrompt;
        EnableDynamicTiling = other.EnableDynamicTiling;
        NumExperts = other.NumExperts;
        NumActiveExperts = other.NumActiveExperts;
    }

    public DeepSeekVL2Options() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 60; NumHeads = 32; ImageSize = 384; LanguageModelName = "DeepSeek-MoE"; MaxVisualTokens = 576; }
    /// <summary>Gets or sets whether dynamic tiling is enabled.</summary>
    public bool EnableDynamicTiling { get; set; } = true;
    /// <summary>Gets or sets the number of MoE experts.</summary>
    public int NumExperts { get; set; } = 64;
    /// <summary>Gets or sets the number of active MoE experts per token.</summary>
    public int NumActiveExperts { get; set; } = 6;
}
