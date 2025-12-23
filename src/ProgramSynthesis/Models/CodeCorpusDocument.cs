using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// A code document that may be searched or used for clone detection.
/// </summary>
public sealed class CodeCorpusDocument
{
    public string DocumentId { get; set; } = string.Empty;

    public string? FilePath { get; set; }

    public ProgramLanguage Language { get; set; } = ProgramLanguage.Generic;

    public string Content { get; set; } = string.Empty;
}
