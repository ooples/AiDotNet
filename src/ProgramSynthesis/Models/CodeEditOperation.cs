namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// A machine-applicable edit operation for code transforms.
/// </summary>
public sealed class CodeEditOperation
{
    public CodeEditOperationType OperationType { get; set; }

    public CodeSpan Span { get; set; } = new();

    public string? Text { get; set; }
}
