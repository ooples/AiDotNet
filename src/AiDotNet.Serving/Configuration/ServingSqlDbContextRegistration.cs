using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Represents a named SQL database context that can be referenced by ID in Serving requests.
/// </summary>
public sealed class ServingSqlDbContextRegistration
{
    /// <summary>
    /// Gets or sets the identifier used by requests (SqlExecuteRequest.DbId).
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the SQL dialect for this context.
    /// </summary>
    public SqlDialect Dialect { get; set; } = SqlDialect.SQLite;

    /// <summary>
    /// Gets or sets the connection string for the database server/context.
    /// </summary>
    public string ConnectionString { get; set; } = string.Empty;
}

