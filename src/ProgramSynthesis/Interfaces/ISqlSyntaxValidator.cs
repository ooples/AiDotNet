namespace AiDotNet.ProgramSynthesis.Interfaces;

/// <summary>
/// Validates whether a string is syntactically valid SQL, used by the program
/// synthesizer to reject malformed candidate SQL programs.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> when the synthesizer generates a SQL program, it needs to
/// know whether the SQL even parses before scoring it. A precise check runs the SQL
/// through a real SQL engine; if no such engine is available, the synthesizer falls
/// back to a generic structural check (balanced brackets, no null bytes, etc.).
/// <para>
/// The precise SQLite-backed implementation (<c>SqliteSqlSyntaxValidator</c>) ships
/// in the opt-in <c>AiDotNet.Storage.Sqlite</c> package (audit-2026-05 finding #14),
/// keeping the native SQLite dependency out of the core package. Register it via
/// <see cref="Engines.SqlSyntaxValidation.Validator"/>.
/// </para>
/// </remarks>
public interface ISqlSyntaxValidator
{
    /// <summary>
    /// Returns <see langword="true"/> if <paramref name="sql"/> is syntactically valid SQL.
    /// </summary>
    /// <remarks>
    /// <paramref name="sql"/> is nullable: <see langword="null"/>, empty, or whitespace is treated
    /// as vacuously valid (there is no malformed program to reject), matching the validators'
    /// behavior and tests. Implementations should return <see langword="false"/> for genuinely
    /// invalid SQL. If the underlying engine cannot be loaded (e.g. a missing native library), the
    /// implementation should let that exception propagate so the caller can fall back to generic
    /// validation rather than mis-reporting a load failure as invalid SQL.
    /// </remarks>
    bool IsValidSql(string? sql);
}
