using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>Options for Cambrian-1 (Spatial Vision Aggregator with 35+ vision encoder combinations).</summary>
public class Cambrian1Options : InstructionTunedVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Cambrian1Options(Cambrian1Options other)
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
        NumVisionEncoders = other.NumVisionEncoders;
        EnableSpatialVisionAggregator = other.EnableSpatialVisionAggregator;
    }

    public Cambrian1Options() { InstructionArchitectureType = InstructionTunedArchitectureType.MLPProjection; VisionDim = 1024; DecoderDim = 4096; ProjectionDim = 4096; NumVisionLayers = 24; NumDecoderLayers = 32; NumHeads = 32; ImageSize = 336; LanguageModelName = "LLaMA-3"; MaxVisualTokens = 576; }
    /// <summary>Gets or sets the number of vision encoders used in the Spatial Vision Aggregator.</summary>
    public int NumVisionEncoders { get; set; } = 4;
    /// <summary>Gets or sets whether the Spatial Vision Aggregator is enabled.</summary>
    public bool EnableSpatialVisionAggregator { get; set; } = true;
}
