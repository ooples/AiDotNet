using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Represents a named SQL dataset (schema + seed data) that can be referenced by ID in Serving requests.
/// </summary>
public sealed class ServingSqlDatasetRegistration
{
    /// <summary>
    /// Gets or sets the identifier used by requests (SqlExecuteRequest.DatasetId).
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the SQL dialect for this dataset.
    /// </summary>
    public SqlDialect Dialect { get; set; } = SqlDialect.SQLite;

    /// <summary>
    /// Gets or sets an optional database context identifier to use when executing this dataset (SqlExecuteRequest.DbId).
    /// </summary>
    public string? DbId { get; set; }

    /// <summary>
    /// Gets or sets the dataset schema (DDL) applied before executing the query.
    /// </summary>
    public string? SchemaSql { get; set; }

    /// <summary>
    /// Gets or sets the dataset seed data (DML) applied before executing the query.
    /// </summary>
    public string? SeedSql { get; set; }
}

