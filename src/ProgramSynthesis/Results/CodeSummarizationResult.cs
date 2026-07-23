using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Results;

public sealed class CodeSummarizationResult : CodeTaskResultBase
{
    public override CodeTask Task => CodeTask.Summarization;

    public string Summary { get; set; } = string.Empty;
}
