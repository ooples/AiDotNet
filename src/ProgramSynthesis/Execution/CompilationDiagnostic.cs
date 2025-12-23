namespace AiDotNet.ProgramSynthesis.Execution;

public sealed class CompilationDiagnostic
{
    public CompilationDiagnosticSeverity Severity { get; init; } = CompilationDiagnosticSeverity.Error;

    public string Message { get; init; } = string.Empty;

    public string? Code { get; init; }

    public string? FilePath { get; init; }

    public int? Line { get; init; }

    public int? Column { get; init; }

    public string? Tool { get; init; }
}

