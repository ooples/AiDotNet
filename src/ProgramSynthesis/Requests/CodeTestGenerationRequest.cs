using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Requests;

/// <summary>
/// Request for generating tests for code.
/// </summary>
public sealed class CodeTestGenerationRequest : CodeTaskRequestBase
{
    /// <inheritdoc />
    public override CodeTask Task => CodeTask.TestGeneration;

    public string Code { get; set; } = string.Empty;
}
