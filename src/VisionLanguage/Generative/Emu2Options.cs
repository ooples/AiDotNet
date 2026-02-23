using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for Emu2 (scaled to 37B; enhanced understanding + generation).</summary>
public class Emu2Options : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Emu2Options(Emu2Options other)
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
        RegressionDim = other.RegressionDim;
        NumRegressionLayers = other.NumRegressionLayers;
    }

    public Emu2Options() { ArchitectureType = GenerativeArchitectureType.UnifiedGeneration; VisionDim = 1408; DecoderDim = 5120; NumVisionLayers = 39; NumDecoderLayers = 60; NumHeads = 40; }
    /// <summary>Gets or sets the visual regression head dimension.</summary>
    public int RegressionDim { get; set; } = 1408;
    /// <summary>Gets or sets the number of regression head layers.</summary>
    public int NumRegressionLayers { get; set; } = 2;
}
