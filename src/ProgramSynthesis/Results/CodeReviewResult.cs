using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Results;

public sealed class CodeReviewResult : CodeTaskResultBase
{
    public override CodeTask Task => CodeTask.CodeReview;

    public List<CodeIssue> Issues { get; set; } = new();

    public List<CodeFixSuggestion> FixSuggestions { get; set; } = new();

    public List<string> PrioritizedPlan { get; set; } = new();
}
