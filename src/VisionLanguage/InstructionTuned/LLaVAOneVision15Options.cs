using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for LLaVA-OneVision-1.5 (fully open, outperforms Qwen2.5-VL on 18/27 benchmarks).</summary>
public class LLaVAOneVision15Options : InstructionTunedVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public LLaVAOneVision15Options(LLaVAOneVision15Options other)
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
        EnableVideo = other.EnableVideo;
        MaxVideoFrames = other.MaxVideoFrames;
    }

    public LLaVAOneVision15Options() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1152; DecoderDim = 3584; ProjectionDim = 3584; NumVisionLayers = 27; NumDecoderLayers = 28; NumHeads = 28; ImageSize = 384; LanguageModelName = "Qwen2.5"; MaxVisualTokens = 729; }
    /// <summary>Gets or sets whether video understanding is enabled.</summary>
    public bool EnableVideo { get; set; } = true;
    /// <summary>Gets or sets the maximum number of video frames.</summary>
    public int MaxVideoFrames { get; set; } = 64;
}
