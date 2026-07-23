namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Configuration options for AiDotNet.Serving persistence (API keys, artifact keys, Data Protection keys).
/// </summary>
public class ServingPersistenceOptions
{
    /// <summary>
    /// Gets or sets the database provider.
    /// </summary>
    public ServingDatabaseProvider Provider { get; set; } = ServingDatabaseProvider.Sqlite;

    /// <summary>
    /// Gets or sets the database connection string.
    /// </summary>
    /// <remarks>
    /// If empty, a safe default is used for SQLite (a local file under the app's data directory).
    /// </remarks>
    public string ConnectionString { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the MySQL server version used by EF Core when <see cref="Provider"/> is <see cref="ServingDatabaseProvider.MySql"/>.
    /// </summary>
    /// <remarks>
    /// This is an explicit version to avoid connecting to the database at application startup (as <c>ServerVersion.AutoDetect</c> would do).
    /// Use a <c>major.minor.patch</c> version string such as <c>8.0.21</c>.
    /// </remarks>
    public string MySqlServerVersion { get; set; } = "8.0.0";

    /// <summary>
    /// Gets or sets whether to apply EF Core migrations at startup.
    /// </summary>
    public bool MigrateOnStartup { get; set; } = true;
}
