using AiDotNet.Models.Options;
using AiDotNet.Onnx;
using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>
/// Base configuration options for generative vision-language models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Generative model. Default values follow the original paper settings.</para>
/// </remarks>
public class GenerativeVLMOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public GenerativeVLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GenerativeVLMOptions(GenerativeVLMOptions other)
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
    }

    /// <summary>Gets or sets the input image size.</summary>
    public int ImageSize { get; set; } = 224;

    /// <summary>Gets or sets the vision encoder feature dimension.</summary>
    public int VisionDim { get; set; } = 768;

    /// <summary>Gets or sets the text decoder hidden dimension.</summary>
    public int DecoderDim { get; set; } = 768;

    /// <summary>Gets or sets the number of vision encoder layers.</summary>
    public int NumVisionLayers { get; set; } = 12;

    /// <summary>Gets or sets the number of text decoder layers.</summary>
    public int NumDecoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumHeads { get; set; } = 12;

    /// <summary>Gets or sets the vocabulary size for tokenization.</summary>
    public int VocabSize { get; set; } = 32000;

    /// <summary>Gets or sets the maximum input text sequence length.</summary>
    public int MaxSequenceLength { get; set; } = 512;

    /// <summary>Gets or sets the maximum generation output length in tokens.</summary>
    public int MaxGenerationLength { get; set; } = 128;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the generative architecture type.</summary>
    public GenerativeArchitectureType ArchitectureType { get; set; } = GenerativeArchitectureType.EncoderDecoder;

    /// <summary>Gets or sets the per-channel image normalization mean.</summary>
    public double[] ImageMean { get; set; } = [0.48145466, 0.4578275, 0.40821073];

    /// <summary>Gets or sets the per-channel image normalization std.</summary>
    public double[] ImageStd { get; set; } = [0.26862954, 0.26130258, 0.27577711];

    /// <summary>Gets or sets the ONNX model path.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the weight decay.</summary>
    public double WeightDecay { get; set; } = 0.01;
}
