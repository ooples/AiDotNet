using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// Base configuration options for Vision-Language-Action (VLA) models.
/// </summary>
public class VisionLanguageActionOptions : GenerativeVLMOptions
{
    /// <summary>Gets or sets the action space dimensionality (e.g., number of joint DOFs).</summary>
    public int ActionDimension { get; set; } = 7;

    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "LLaMA";

    /// <summary>Gets or sets the maximum action prediction horizon (number of future steps).</summary>
    public int PredictionHorizon { get; set; } = 16;

    /// <summary>Gets or sets the observation history length (number of past frames).</summary>
    public int ObservationHistory { get; set; } = 2;
}
