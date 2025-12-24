namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Provenance metadata for a result hit when an indexed corpus is used.
/// </summary>
public sealed class CodeProvenance
{
    public string? IndexId { get; set; }

    public string? RepoId { get; set; }

    public string? CommitOrRef { get; set; }

    public string? SourcePath { get; set; }
}
