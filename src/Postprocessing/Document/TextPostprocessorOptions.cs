namespace AiDotNet.Postprocessing.Document;

/// <summary>
/// Options for text postprocessing.
/// </summary>
public class TextPostprocessorOptions
{
    /// <summary>
    /// Whether to remove control characters. Default: true.
    /// </summary>
    public bool RemoveControlCharacters { get; set; } = true;

    /// <summary>
    /// Whether to normalize special characters. Default: true.
    /// </summary>
    public bool NormalizeCharacters { get; set; } = true;

    /// <summary>
    /// Whether to normalize whitespace. Default: true.
    /// </summary>
    public bool NormalizeWhitespace { get; set; } = true;

    /// <summary>
    /// Whether to fix common OCR errors. Default: true.
    /// </summary>
    public bool FixCommonOcrErrors { get; set; } = true;

    /// <summary>
    /// Whether to apply spell correction. Default: false.
    /// </summary>
    public bool ApplySpellCorrection { get; set; }

    /// <summary>
    /// Whether to merge incorrectly broken lines. Default: true.
    /// </summary>
    public bool MergeBrokenLines { get; set; } = true;

    /// <summary>
    /// Whether to remove duplicate spaces. Default: true.
    /// </summary>
    public bool RemoveDuplicateSpaces { get; set; } = true;
}
