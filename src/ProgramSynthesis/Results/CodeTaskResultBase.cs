using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Results;

/// <summary>
/// Base type for structured results returned from code tasks.
/// </summary>
/// <remarks>
/// <para>
/// All task results include the task identity, a success/error envelope, and telemetry that can be
/// tier-redacted by Serving.
/// </para>
/// <para><b>For Beginners:</b> This is the common "envelope" for any code task result.
/// It tells you whether the task succeeded, and it includes useful metadata about what happened.
/// </para>
/// </remarks>
public abstract class CodeTaskResultBase
{
    public abstract CodeTask Task { get; }

    public ProgramLanguage Language { get; set; } = ProgramLanguage.Generic;

    public string? RequestId { get; set; }

    public bool Success { get; set; }

    public string? Error { get; set; }

    public CodeTaskTelemetry Telemetry { get; set; } = new();

    protected CodeTaskResultBase()
    {
    }
}
