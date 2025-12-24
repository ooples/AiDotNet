namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Telemetry captured during a code task execution.
/// </summary>
/// <remarks>
/// <para>
/// Serving may redact this information by tier.
/// </para>
/// </remarks>
public sealed class CodeTaskTelemetry
{
    /// <summary>
    /// Gets or sets the total processing time in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }

    /// <summary>
    /// Gets or sets optional execution telemetry if the task ran code in a sandbox.
    /// </summary>
    public CodeExecutionTelemetry? Execution { get; set; }
}
