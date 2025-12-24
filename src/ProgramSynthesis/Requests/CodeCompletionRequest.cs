using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Requests;

/// <summary>
/// Request for code completion.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> You give partial code, and the system suggests how to finish it.</para>
/// </remarks>
public sealed class CodeCompletionRequest : CodeTaskRequestBase
{
    public override CodeTask Task => CodeTask.Completion;

    public string Code { get; set; } = string.Empty;

    /// <summary>
    /// Optional cursor position (absolute offset) inside <see cref="Code"/>.
    /// When null, the cursor is assumed to be at the end of <see cref="Code"/>.
    /// </summary>
    public int? CursorOffset { get; set; }

    /// <summary>
    /// Maximum number of completion candidates to return.
    /// </summary>
    public int MaxCandidates { get; set; } = 3;
}
