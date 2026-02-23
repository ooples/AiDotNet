using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for SmolVLM (tiny efficient VLMs: 256M/500M/2.2B from HuggingFace).</summary>
public class SmolVLMOptions : InstructionTunedVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SmolVLMOptions(SmolVLMOptions other)
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
        ModelVariant = other.ModelVariant;
    }

    public SmolVLMOptions() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 384; DecoderDim = 576; ProjectionDim = 576; NumVisionLayers = 12; NumDecoderLayers = 16; NumHeads = 9; ImageSize = 384; LanguageModelName = "SmolLM"; MaxVisualTokens = 256; }
    /// <summary>Gets or sets the model size variant (e.g., "256M", "500M", "2.2B").</summary>
    public string ModelVariant { get; set; } = "2.2B";
}
