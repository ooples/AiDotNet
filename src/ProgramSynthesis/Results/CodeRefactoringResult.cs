using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Results;

public sealed class CodeRefactoringResult : CodeTaskResultBase
{
    public override CodeTask Task => CodeTask.Refactoring;

    public string RefactoredCode { get; set; } = string.Empty;

    public CodeTransformDiff Diff { get; set; } = new();
}
