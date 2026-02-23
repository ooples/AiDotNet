using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>Options for Emu (unified VQA, captioning, generation via EVA-CLIP + LLM + diffusion).</summary>
public class EmuOptions : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EmuOptions(EmuOptions other)
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

    public EmuOptions() { ArchitectureType = GenerativeArchitectureType.UnifiedGeneration; VisionDim = 1408; DecoderDim = 4096; NumVisionLayers = 39; NumDecoderLayers = 32; NumHeads = 32; }
    /// <summary>Gets or sets the visual regression head dimension (for image generation).</summary>
    public int RegressionDim { get; set; } = 1408;
    /// <summary>Gets or sets the number of regression head layers.</summary>
    public int NumRegressionLayers { get; set; } = 2;
}
