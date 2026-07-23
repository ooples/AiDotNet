using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Execution;

public sealed class ProgramExecuteResponse
{
    public required bool Success { get; init; }

    public required ProgramLanguage Language { get; init; }

    public bool CompilationAttempted { get; init; }

    public bool? CompilationSucceeded { get; init; }

    public List<CompilationDiagnostic> CompilationDiagnostics { get; init; } = new();

    public required int ExitCode { get; init; }

    public string StdOut { get; init; } = string.Empty;

    public string StdErr { get; init; } = string.Empty;

    public bool StdOutTruncated { get; init; }

    public bool StdErrTruncated { get; init; }

    public string? Error { get; init; }

    public ProgramExecuteErrorCode? ErrorCode { get; init; }
}
