using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Requests;

/// <summary>
/// Request for structured code understanding output.
/// </summary>
public sealed class CodeUnderstandingRequest : CodeTaskRequestBase
{
    public override CodeTask Task => CodeTask.Understanding;

    public string Code { get; set; } = string.Empty;

    /// <summary>
    /// Optional filename used for location/path display.
    /// </summary>
    public string? FilePath { get; set; }
}
