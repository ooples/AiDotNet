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

    public string? PostgresConnectionString { get; set; }

    public string? MySqlConnectionString { get; set; }

    public bool EnableDockerFallback { get; set; } = true;

    /// <summary>
    /// Optional named database contexts that can be referenced via <c>DbId</c> in <see cref="AiDotNet.ProgramSynthesis.Execution.SqlExecuteRequest"/>.
    /// </summary>
    public List<ServingSqlDbContextRegistration> DbContexts { get; set; } = new();

    /// <summary>
    /// Optional named datasets (schema + seed) that can be referenced via <c>DatasetId</c> in <see cref="AiDotNet.ProgramSynthesis.Execution.SqlExecuteRequest"/>.
    /// </summary>
    public List<ServingSqlDatasetRegistration> Datasets { get; set; } = new();
}
