using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Requests;

/// <summary>
/// Request for summarizing code into natural language.
/// </summary>
public sealed class CodeSummarizationRequest : CodeTaskRequestBase
{
    public override CodeTask Task => CodeTask.Summarization;

    public string Code { get; set; } = string.Empty;
}
