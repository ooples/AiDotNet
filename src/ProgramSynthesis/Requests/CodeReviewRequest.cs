using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Requests;

/// <summary>
/// Request for structured code review output.
/// </summary>
public sealed class CodeReviewRequest : CodeTaskRequestBase
{
    public override CodeTask Task => CodeTask.CodeReview;

    public string Code { get; set; } = string.Empty;

    /// <summary>
    /// Optional filename used for location/path display.
    /// </summary>
    public string? FilePath { get; set; }
}
