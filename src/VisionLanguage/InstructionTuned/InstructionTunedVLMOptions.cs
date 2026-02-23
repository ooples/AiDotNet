using AiDotNet.VisionLanguage.Encoders;
using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>
/// Base configuration options for instruction-tuned vision-language models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the InstructionTuned model. Default values follow the original paper settings.</para>
/// </remarks>
public class InstructionTunedVLMOptions : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public InstructionTunedVLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public InstructionTunedVLMOptions(InstructionTunedVLMOptions other)
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
    }

    /// <summary>Gets or sets the instruction-tuned architecture type.</summary>
    public InstructionTunedArchitectureType InstructionArchitectureType { get; set; } = InstructionTunedArchitectureType.MLPProjection;

    /// <summary>Gets or sets the MLP projection hidden dimension (for MLP connector models).</summary>
    public int ProjectionDim { get; set; } = 4096;

    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "LLaMA";

    /// <summary>Gets or sets the maximum number of visual tokens per image.</summary>
    public int MaxVisualTokens { get; set; } = 576;

    /// <summary>Gets or sets the system prompt for chat mode.</summary>
    public string SystemPrompt { get; set; } = "You are a helpful assistant.";
}
