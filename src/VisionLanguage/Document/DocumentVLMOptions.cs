using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Base configuration options for document understanding vision-language models.
/// </summary>
public class DocumentVLMOptions : GenerativeVLMOptions
{
    /// <summary>Gets or sets whether this model operates OCR-free (no external OCR required).</summary>
    public bool IsOcrFree { get; set; } = true;

    /// <summary>Gets or sets the maximum document page count supported.</summary>
    public int MaxPages { get; set; } = 1;

    /// <summary>Gets or sets the maximum output text length in tokens.</summary>
    public int MaxOutputTokens { get; set; } = 4096;
}
