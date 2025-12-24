namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Defines a corpus either by embedding the documents in the request, or by referencing an indexed corpus in Serving.
/// </summary>
public sealed class CodeCorpusReference
{
    public List<CodeCorpusDocument> Documents { get; set; } = new();

    public string? CorpusId { get; set; }

    public string? IndexId { get; set; }
}
