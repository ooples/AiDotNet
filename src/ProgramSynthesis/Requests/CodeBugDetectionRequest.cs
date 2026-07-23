using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Requests;

/// <summary>
/// Request for identifying potential bugs and issues in code.
/// </summary>
public sealed class CodeBugDetectionRequest : CodeTaskRequestBase
{
    public override CodeTask Task => CodeTask.BugDetection;

    public string Code { get; set; } = string.Empty;
}
