namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Configuration options for Janus-Pro: scaled data and model with optimized training strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the JanusPro model. Default values follow the original paper settings.</para>
/// </remarks>
public class JanusProOptions : UnifiedVisionOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public JanusProOptions(JanusProOptions other)
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
        LanguageModelName = other.LanguageModelName;
        SupportsGeneration = other.SupportsGeneration;
        OutputImageSize = other.OutputImageSize;
        NumVisualTokens = other.NumVisualTokens;
        EnableDecoupledEncoding = other.EnableDecoupledEncoding;
        NumGenerationTokens = other.NumGenerationTokens;
        CodebookEmbeddingDim = other.CodebookEmbeddingDim;
        CfgScale = other.CfgScale;
    }

    public JanusProOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 384;
        VocabSize = 32000;
        LanguageModelName = "DeepSeek-LLM";
        NumVisualTokens = 16384;
    }

    /// <summary>Whether to keep understanding and generation vision paths fully decoupled (Janus paper §3.1).</summary>
    public bool EnableDecoupledEncoding { get; set; } = true;

    // Settable properties below validate at the API boundary per the
    // audit-2026-05 review — invalid values (zero, negative, NaN,
    // Infinity) would otherwise propagate into generation paths and
    // crash or produce undefined-behaviour outputs deeper in the
    // pipeline where the diagnostic is much harder to pin to a
    // misconfigured option. Backing-field pattern matches the rest
    // of the codebase (see SpikingNeuralNetworkOptions / ESN
    // options for the same pattern).

    private int _numGenerationTokens = 576;

    /// <summary>
    /// Number of VQ tokens emitted by the generation path. Janus-Pro uses 576 (a 24×24 grid) for the
    /// default 384×384 output (paper §3.3 — patch size 16, output 384 → 24×24). Must be positive.
    /// </summary>
    public int NumGenerationTokens
    {
        get => _numGenerationTokens;
        set =>
            _numGenerationTokens =
                value > 0
                    ? value
                    : throw new ArgumentOutOfRangeException(
                        nameof(NumGenerationTokens),
                        value,
                        "Must be > 0."
                    );
    }

    private int _codebookEmbeddingDim = 8;

    /// <summary>Dimensionality of each VQ codebook entry's continuous embedding. Janus-Pro paper Table 1: 8. Must be positive.</summary>
    public int CodebookEmbeddingDim
    {
        get => _codebookEmbeddingDim;
        set =>
            _codebookEmbeddingDim =
                value > 0
                    ? value
                    : throw new ArgumentOutOfRangeException(
                        nameof(CodebookEmbeddingDim),
                        value,
                        "Must be > 0."
                    );
    }

    private double _cfgScale = 7.0;

    /// <summary>
    /// Classifier-free guidance scale used during generation (Ho &amp; Salimans 2022). Janus-Pro paper uses
    /// 5–7 depending on prompt; 7.0 is the default for high-fidelity outputs. Must be finite and positive
    /// (negative or zero CFG inverts the guidance direction; NaN/Inf would propagate to weighted-sum
    /// formulas downstream).
    /// </summary>
    public double CfgScale
    {
        get => _cfgScale;
        set
        {
            if (double.IsNaN(value) || double.IsInfinity(value) || value <= 0)
                throw new ArgumentOutOfRangeException(
                    nameof(CfgScale),
                    value,
                    "Must be a finite positive number. Paper uses 5-7."
                );
            _cfgScale = value;
        }
    }
}
