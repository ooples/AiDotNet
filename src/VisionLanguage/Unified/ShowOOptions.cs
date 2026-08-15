namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Configuration options for Show-o: single transformer for unified understanding and generation.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow the released 256px Show-o checkpoint: Phi-1.5 at 2048 hidden dimensions,
/// 24 layers and 32 heads, an 8192-entry MAGVIT-v2 codebook, 256 image tokens, and the
/// published AdamW 1e-4/0.01 training recipe.
/// </para>
/// <para><b>For Beginners:</b> These options configure the released Show-o architecture and can be changed for other checkpoints.</para>
/// </remarks>
public class ShowOOptions : UnifiedVisionOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ShowOOptions(ShowOOptions other)
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        OnnxOptions = new AiDotNet.Onnx.OnnxModelOptions(other.OnnxOptions);
        ImageTokenCount = other.ImageTokenCount;
        DiffusionSteps = other.DiffusionSteps;
    }

    public ShowOOptions()
    {
        VisionDim = 2048;
        DecoderDim = 2048;
        NumVisionLayers = 0;
        NumDecoderLayers = 24;
        NumHeads = 32;
        ImageSize = 256;
        VocabSize = 58498;
        MaxSequenceLength = 128;
        DropoutRate = 0.0;
        LearningRate = 1e-4;
        WeightDecay = 0.01;
        LanguageModelName = "Phi-1.5";
        NumVisualTokens = 8192;
        OutputImageSize = 256;
    }

    /// <summary>Gets or sets the number of MAGVIT-v2 image-token positions.</summary>
    public int ImageTokenCount { get; set; } = 256;

    /// <summary>Gets or sets the number of discrete mask-prediction diffusion steps.</summary>
    public int DiffusionSteps { get; set; } = 16;
}
