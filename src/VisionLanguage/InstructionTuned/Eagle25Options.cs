using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Eagle 2.5 (NVIDIA long-context multimodal for video + high-res images).</summary>
public class Eagle25Options : InstructionTunedVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Eagle25Options(Eagle25Options other)
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
        MaxVideoFrames = other.MaxVideoFrames;
        EnableLongContext = other.EnableLongContext;
    }

    public Eagle25Options() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 3584; ProjectionDim = 3584; NumVisionLayers = 24; NumDecoderLayers = 28; NumHeads = 28; ImageSize = 448; LanguageModelName = "Qwen2"; MaxVisualTokens = 2048; }
    /// <summary>Gets or sets the maximum number of video frames for long-context video understanding.</summary>
    public int MaxVideoFrames { get; set; } = 512;
    /// <summary>Gets or sets whether long-context mode is enabled.</summary>
    public bool EnableLongContext { get; set; } = true;
}
