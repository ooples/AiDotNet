namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Minimal execution telemetry for sandboxed runs.
/// </summary>
public sealed class CodeExecutionTelemetry
{
    public int? ExitCode { get; set; }

    public bool TimedOut { get; set; }

    public long? StdoutBytes { get; set; }

    public long? StderrBytes { get; set; }
}
