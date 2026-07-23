namespace AiDotNet.ProgramSynthesis.Models;

public sealed class CodeComplexityMetrics
{
    public int LineCount { get; set; }

    public int CharacterCount { get; set; }

    public int EstimatedCyclomaticComplexity { get; set; }
}
