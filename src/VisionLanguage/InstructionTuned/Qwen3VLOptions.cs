using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Qwen3-VL (latest series with 2B/4B/8B/32B variants).</summary>
public class Qwen3VLOptions : InstructionTunedVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Qwen3VLOptions(Qwen3VLOptions other)
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
        ResamplerDim = other.ResamplerDim;
        NumResamplerLayers = other.NumResamplerLayers;
        NumResamplerHeads = other.NumResamplerHeads;
    }

    public Qwen3VLOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.CrossAttentionResampler; VisionDim = 1152; DecoderDim = 3584; NumVisionLayers = 32; NumDecoderLayers = 28; NumHeads = 16; ImageSize = 448; LanguageModelName = "Qwen3"; MaxVisualTokens = 1024; }
    /// <summary>Gets or sets the resampler dimension.</summary>
    public int ResamplerDim { get; set; } = 1152;
    /// <summary>Gets or sets the number of resampler layers.</summary>
    public int NumResamplerLayers { get; set; } = 6;
    /// <summary>Gets or sets the number of resampler heads.</summary>
    public int NumResamplerHeads { get; set; } = 16;
}
