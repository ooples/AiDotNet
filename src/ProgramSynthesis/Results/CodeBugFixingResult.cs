using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Results;

public sealed class CodeBugFixingResult : CodeTaskResultBase
{
    public override CodeTask Task => CodeTask.BugFixing;

    public string FixedCode { get; set; } = string.Empty;

    public CodeTransformDiff Diff { get; set; } = new();

    public List<CodeIssue> FixedIssues { get; set; } = new();
}
