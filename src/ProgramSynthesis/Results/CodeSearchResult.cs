using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Results;

public sealed class CodeSearchResult : CodeTaskResultBase
{
    public override CodeTask Task => CodeTask.Search;

    public List<string> FiltersApplied { get; set; } = new();

    public List<CodeSearchHit> Results { get; set; } = new();
}
