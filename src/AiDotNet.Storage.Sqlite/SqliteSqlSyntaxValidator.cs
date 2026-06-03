using AiDotNet.ProgramSynthesis.Interfaces;
using Microsoft.Data.Sqlite;

namespace AiDotNet.ProgramSynthesis.Validation;

/// <summary>
/// Precise <see cref="ISqlSyntaxValidator"/> that validates SQL by preparing it
/// against an in-memory SQLite engine.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> this checks whether a string is valid SQL by handing it to a
/// real (in-memory) SQLite database and asking it to prepare the statement. If SQLite
/// accepts it, the SQL is valid; if SQLite rejects it, it is invalid.
/// <para>
/// Register it once at startup so the program synthesizer uses precise SQL validation:
/// <code>
/// AiDotNet.ProgramSynthesis.Engines.SqlSyntaxValidation.Validator =
///     new AiDotNet.ProgramSynthesis.Validation.SqliteSqlSyntaxValidator();
/// </code>
/// This validator lives in the opt-in <c>AiDotNet.Storage.Sqlite</c> package so the
/// native SQLite dependency stays out of the core package (audit-2026-05 finding #14).
/// </para>
/// </remarks>
public sealed class SqliteSqlSyntaxValidator : ISqlSyntaxValidator
{
    /// <inheritdoc />
    /// <remarks>
    /// Returns <see langword="false"/> for genuinely invalid SQL. Native-library load
    /// failures (<see cref="TypeInitializationException"/>, <see cref="DllNotFoundException"/>,
    /// <see cref="FileNotFoundException"/>, <see cref="BadImageFormatException"/>) are allowed
    /// to propagate so the caller can fall back to generic validation rather than treating a
    /// load failure as invalid SQL.
    /// </remarks>
    public bool IsValidSql(string sql)
    {
        try
        {
            using var connection = new SqliteConnection("Data Source=:memory:");
            connection.Open();

            using var command = connection.CreateCommand();
            command.CommandText = sql ?? string.Empty;
            command.Prepare();

            return true;
        }
        catch (SqliteException)
        {
            return false;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
        catch (ArgumentException)
        {
            return false;
        }
    }
}
