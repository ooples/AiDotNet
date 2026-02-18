using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Base configuration options for video-language models.
/// </summary>
public class VideoLanguageOptions : GenerativeVLMOptions
{
    /// <summary>Gets or sets the maximum number of video frames the model can process.</summary>
    public int MaxFrames { get; set; } = 32;

    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "LLaMA";

    /// <summary>Gets or sets the MLP projection hidden dimension.</summary>
    public int ProjectionDim { get; set; } = 4096;

    /// <summary>Gets or sets the system prompt for chat mode.</summary>
    public string SystemPrompt { get; set; } = "You are a helpful video understanding assistant.";
}
