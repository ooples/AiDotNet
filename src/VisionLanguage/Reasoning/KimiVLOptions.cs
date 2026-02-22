namespace AiDotNet.VisionLanguage.Reasoning;

/// <summary>
/// Configuration options for Kimi-VL: MoE VLM with MoonViT and long-context processing.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the KimiVL model. Default values follow the original paper settings.</para>
/// </remarks>
public class KimiVLOptions : ReasoningVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public KimiVLOptions(KimiVLOptions other)
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
        ReasoningApproach = other.ReasoningApproach;
        MaxReasoningTokens = other.MaxReasoningTokens;
        EnableThinkingSteps = other.EnableThinkingSteps;
        TotalParameters = other.TotalParameters;
        ActiveParameters = other.ActiveParameters;
        EnableLongContext = other.EnableLongContext;
    }

    public KimiVLOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 28;
        NumHeads = 32;
        ImageSize = 384;
        VocabSize = 128256;
        LanguageModelName = "MoonshotMoE";
        ReasoningApproach = "MoE-Reasoning";
    }

    /// <summary>Gets or sets the total parameter count in billions.</summary>
    public int TotalParameters { get; set; } = 16;

    /// <summary>Gets or sets the active parameter count in billions (MoE routing).</summary>
    public int ActiveParameters { get; set; } = 2;

    /// <summary>Gets or sets whether to enable 128K long-context mode.</summary>
    public bool EnableLongContext { get; set; } = true;
}
