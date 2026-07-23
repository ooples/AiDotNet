namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Supported database providers for AiDotNet.Serving persistence.
/// </summary>
public enum ServingDatabaseProvider
{
    /// <summary>
    /// SQLite (file-based, single-node).
    /// </summary>
    Sqlite,

    /// <summary>
    /// PostgreSQL.
    /// </summary>
    PostgreSql,

    /// <summary>
    /// Microsoft SQL Server.
    /// </summary>
    SqlServer,

    /// <summary>
    /// MySQL.
    /// </summary>
    MySql
}

