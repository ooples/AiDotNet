using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Requests;

/// <summary>
/// Request for generating or improving code documentation.
/// </summary>
public sealed class CodeDocumentationRequest : CodeTaskRequestBase
{
    public override CodeTask Task => CodeTask.Documentation;

    public string Code { get; set; } = string.Empty;
}
