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
        catch (SqliteException ex)
        {
            // Prepare compiles against THIS connection's schema — a fresh, EMPTY
            // in-memory database — so a syntactically valid statement that references
            // any table/column/function still fails at prepare time with a
            // schema-resolution error ("no such table: users", …). Those failures mean
            // the PARSE succeeded: the statement is valid SQL syntax and only the
            // referenced objects are absent from the empty scratch database. This is a
            // SYNTAX validator, so treat them as valid; everything else (e.g.
            // 'near "FORM": syntax error', 'incomplete input') is a genuine syntax
            // failure and stays invalid.
            return IsSchemaResolutionError(ex);
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

    /// <summary>
    /// True when the prepare-time failure is the empty scratch schema failing to
    /// resolve a referenced object rather than the SQL failing to parse. SQLite
    /// reports both syntax and schema-resolution failures with the same primary
    /// result code (<c>SQLITE_ERROR</c> = 1), so the message text is the only
    /// discriminator available at prepare time.
    /// </summary>
    private static bool IsSchemaResolutionError(SqliteException ex)
    {
        string message = ex.Message ?? string.Empty;
        return ContainsOrdinalIgnoreCase(message, "no such table")
            || ContainsOrdinalIgnoreCase(message, "no such column")
            || ContainsOrdinalIgnoreCase(message, "no such view")
            || ContainsOrdinalIgnoreCase(message, "no such index")
            || ContainsOrdinalIgnoreCase(message, "no such function")
            || ContainsOrdinalIgnoreCase(message, "no such collation sequence")
            || ContainsOrdinalIgnoreCase(message, "no such module");
    }

    // string.Contains(string, StringComparison) is unavailable on net471.
    private static bool ContainsOrdinalIgnoreCase(string haystack, string needle)
        => haystack.IndexOf(needle, StringComparison.OrdinalIgnoreCase) >= 0;
}
