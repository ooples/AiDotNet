using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Results;

public sealed class CodeCompletionResult : CodeTaskResultBase
{
    public override CodeTask Task => CodeTask.Completion;

    public List<CodeCompletionCandidate> Candidates { get; set; } = new();
}
