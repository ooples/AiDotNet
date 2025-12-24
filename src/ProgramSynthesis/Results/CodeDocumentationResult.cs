using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Results;

public sealed class CodeDocumentationResult : CodeTaskResultBase
{
    public override CodeTask Task => CodeTask.Documentation;

    public string Documentation { get; set; } = string.Empty;

    public string UpdatedCode { get; set; } = string.Empty;
}
