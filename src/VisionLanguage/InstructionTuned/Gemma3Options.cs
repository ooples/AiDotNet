using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Gemma 3 (3B-72B, Native Dynamic Resolution ViT, 128k context, 29 languages).</summary>
public class Gemma3Options : InstructionTunedVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Gemma3Options(Gemma3Options other)
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
        EnableNativeDynamicResolution = other.EnableNativeDynamicResolution;
    }

    public Gemma3Options() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1152; DecoderDim = 3584; ProjectionDim = 3584; NumVisionLayers = 27; NumDecoderLayers = 36; NumHeads = 16; ImageSize = 896; LanguageModelName = "Gemma-3"; MaxVisualTokens = 4096; }
    /// <summary>Gets or sets whether native dynamic resolution is enabled.</summary>
    public bool EnableNativeDynamicResolution { get; set; } = true;
}
