using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Execution;

public sealed class SqlExecuteRequest
{
    public SqlDialect? Dialect { get; set; }

    public string Query { get; set; } = string.Empty;

    /// <summary>
    /// Optional request-scoped schema (DDL) for ephemeral execution contexts.
    /// </summary>
    public string? SchemaSql { get; set; }

    /// <summary>
    /// Optional request-scoped seed data (DML) for ephemeral execution contexts.
    /// </summary>
    public string? SeedSql { get; set; }

    /// <summary>
    /// Optional Serving-registered database context identifier.
    /// </summary>
    public string? DbId { get; set; }

    /// <summary>
    /// Optional Serving-registered dataset context identifier.
    /// </summary>
    public string? DatasetId { get; set; }
}

