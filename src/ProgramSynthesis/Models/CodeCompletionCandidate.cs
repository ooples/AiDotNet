namespace AiDotNet.ProgramSynthesis.Models;

public sealed class CodeCompletionCandidate
{
    public string CompletionText { get; set; } = string.Empty;

    public double Score { get; set; }
}
