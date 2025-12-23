using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Requests;

/// <summary>
/// Request for searching code using a query against a corpus.
/// </summary>
public sealed class CodeSearchRequest : CodeTaskRequestBase
{
    public override CodeTask Task => CodeTask.Search;

    public string Query { get; set; } = string.Empty;

    /// <summary>
    /// The search corpus definition (request-scoped docs and/or an indexed corpus reference).
    /// </summary>
    public CodeCorpusReference Corpus { get; set; } = new();

    /// <summary>
    /// Optional filters expressed as simple strings (Serving may interpret/enforce these).
    /// </summary>
    public List<string> Filters { get; set; } = new();
}
