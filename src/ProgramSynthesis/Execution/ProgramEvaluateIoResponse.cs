using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Execution;

public sealed class ProgramEvaluateIoResponse
{
    public required bool Success { get; init; }

    public required ProgramLanguage Language { get; init; }

    public int TotalTests { get; init; }

    public int PassedTests { get; init; }

    public double PassRate { get; init; }

    public List<ProgramEvaluateIoTestResult> TestResults { get; init; } = new();

    public string? Error { get; init; }
}

