using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Execution;

public sealed class ProgramEvaluateIoTestResult
{
    public ProgramInputOutputExample TestCase { get; init; } = new();

    public bool Passed { get; init; }

    public string? FailureReason { get; init; }

    public ProgramExecuteResponse Execution { get; init; } = new ProgramExecuteResponse
    {
        Success = false,
        Language = ProgramLanguage.Generic,
        ExitCode = -1
    };
}

