using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Results;

public sealed class CodeBugDetectionResult : CodeTaskResultBase
{
    public override CodeTask Task => CodeTask.BugDetection;

    public List<CodeIssue> Issues { get; set; } = new();
}
