using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Requests;

/// <summary>
/// Base type for all code task execution requests.
/// </summary>
/// <remarks>
/// <para>
/// This request model is used as the canonical input shape for all <see cref="CodeTask"/> operations.
/// Concrete request types (e.g., completion, search, code review) add task-specific fields.
/// </para>
/// <para><b>For Beginners:</b> This is the common "envelope" for any code task request.
///
/// Think of this as a form you submit to ask the system to do something with code.
/// The specific task (like Search or CodeReview) determines which extra fields you need to fill in.
/// </para>
/// </remarks>
public abstract class CodeTaskRequestBase
{
    /// <summary>
    /// Gets the requested task.
    /// </summary>
    public abstract CodeTask Task { get; }

    /// <summary>
    /// Gets or sets the primary language context for the request.
    /// </summary>
    public ProgramLanguage Language { get; set; } = ProgramLanguage.Generic;

    /// <summary>
    /// Gets or sets the SQL dialect to use when <see cref="Language"/> is <see cref="ProgramLanguage.SQL"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If not specified, Serving uses its configured default dialect (typically SQLite).
    /// </para>
    /// </remarks>
    public SqlDialect? SqlDialect { get; set; }

    /// <summary>
    /// Gets or sets an optional request identifier for correlation and tracing.
    /// </summary>
    public string? RequestId { get; set; }

    /// <summary>
    /// Gets or sets an optional wall-clock time budget (in milliseconds) for the request.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is a best-effort time budget for request processing. When running through AiDotNet.Serving, the effective
    /// time budget is clamped by the tier limits configured on the server.
    /// </para>
    /// <para><b>For Beginners:</b> This is a "timeout" for the request. If the work takes too long, the server can stop
    /// processing and return a failure result instead of hanging forever.
    /// </para>
    /// </remarks>
    public int? MaxWallClockMilliseconds { get; set; }

    protected CodeTaskRequestBase()
    {
    }
}
