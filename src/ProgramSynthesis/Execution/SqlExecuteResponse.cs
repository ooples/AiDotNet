using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Execution;

public sealed class SqlExecuteResponse
{
    public required bool Success { get; init; }

    public SqlDialect? Dialect { get; init; }

    public List<string> Columns { get; init; } = new();

    public List<Dictionary<string, SqlValue>> Rows { get; init; } = new();

    public string? Error { get; init; }

    public SqlExecuteErrorCode? ErrorCode { get; init; }
}
