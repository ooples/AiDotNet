namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Database configuration for AiDotNet.Serving persistence.
/// </summary>
/// <remarks>
/// <para>
/// This database stores Serving state such as API keys, credential metadata, index metadata, and key material.
/// It is not the same as <see cref="AiDotNet.ProgramSynthesis.Enums.SqlDialect"/> which is used for executing SQL code.
/// </para>
/// </remarks>
public sealed class ServingDatabaseOptions
{
    public ServingDatabaseProvider Provider { get; set; } = ServingDatabaseProvider.Sqlite;

    /// <summary>
    /// Explicit connection string. If omitted for SQLite, a file-based SQLite DB is used.
    /// </summary>
    public string? ConnectionString { get; set; }

    /// <summary>
    /// SQLite file name used when <see cref="Provider"/> is <see cref="ServingDatabaseProvider.Sqlite"/>
    /// and <see cref="ConnectionString"/> is not provided.
    /// </summary>
    public string SqliteFileName { get; set; } = "aidotnet.serving.db";

    /// <summary>
    /// If true, applies EF Core migrations at startup.
    /// </summary>
    public bool ApplyMigrationsAtStartup { get; set; } = true;
}

