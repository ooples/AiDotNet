using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Configuration for executing <see cref="ProgramLanguage.SQL"/> safely in AiDotNet.Serving.
/// </summary>
public sealed class ServingSqlSandboxOptions
{
    public SqlDialect DefaultDialect { get; set; } = SqlDialect.SQLite;

    public int CommandTimeoutSeconds { get; set; } = 5;

    public int MaxResultRows { get; set; } = 1000;

    /// <summary>
    /// Connection string for a Postgres server that sandboxed requests run against. When unset and
    /// <see cref="EnableDockerFallback"/> is true, every request gets its own throwaway container instead.
    /// </summary>
    /// <remarks>
    /// Caller-authored SQL (query, schema and seed scripts) executes with this credential. The per-request
    /// schema and <c>search_path</c> are namespacing, not a security boundary: SQL can still name other
    /// schemas and, through the multi-statement schema/seed scripts, issue transaction control. Point this at
    /// a dedicated, disposable database whose role owns nothing else and cannot reach other databases.
    /// </remarks>
    public string? PostgresConnectionString { get; set; }

    /// <summary>
    /// Connection string for a MySQL server that sandboxed requests run against. When unset and
    /// <see cref="EnableDockerFallback"/> is true, every request gets its own throwaway container instead.
    /// </summary>
    /// <remarks>
    /// Caller-authored SQL executes with this credential, and the per-request database is namespacing, not a
    /// security boundary (SQL can name other databases). Use a dedicated server or an account whose grants
    /// are limited to creating and using its own scratch databases.
    /// </remarks>
    public string? MySqlConnectionString { get; set; }

    public bool EnableDockerFallback { get; set; } = true;

    /// <summary>
    /// Optional named database contexts that can be referenced via <c>DbId</c> in <see cref="AiDotNet.ProgramSynthesis.Execution.SqlExecuteRequest"/>.
    /// </summary>
    /// <remarks>
    /// The same trust rule as <see cref="PostgresConnectionString"/> applies: caller SQL runs with each
    /// context's credential, so every registered connection string must be a dedicated, least-privilege
    /// sandbox database.
    /// </remarks>
    public List<ServingSqlDbContextRegistration> DbContexts { get; set; } = new();

    /// <summary>
    /// Optional named datasets (schema + seed) that can be referenced via <c>DatasetId</c> in <see cref="AiDotNet.ProgramSynthesis.Execution.SqlExecuteRequest"/>.
    /// </summary>
    public List<ServingSqlDatasetRegistration> Datasets { get; set; } = new();
}
