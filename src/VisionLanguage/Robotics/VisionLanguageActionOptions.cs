using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Base configuration options for Vision-Language-Action (VLA) models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VisionLanguageAction model. Default values follow the original paper settings.</para>
/// </remarks>
public class VisionLanguageActionOptions : GenerativeVLMOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public VisionLanguageActionOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VisionLanguageActionOptions(VisionLanguageActionOptions other)
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
        ActionDimension = other.ActionDimension;
        LanguageModelName = other.LanguageModelName;
        PredictionHorizon = other.PredictionHorizon;
        ObservationHistory = other.ObservationHistory;
    }

    /// <summary>Gets or sets the action space dimensionality (e.g., number of joint DOFs).</summary>
    public int ActionDimension { get; set; } = 7;

    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "LLaMA";

    /// <summary>Gets or sets the maximum action prediction horizon (number of future steps).</summary>
    public int PredictionHorizon { get; set; } = 16;

    /// <summary>Gets or sets the observation history length (number of past frames).</summary>
    public int ObservationHistory { get; set; } = 2;
}
