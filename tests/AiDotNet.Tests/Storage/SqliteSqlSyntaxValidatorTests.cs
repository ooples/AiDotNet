using AiDotNet.ProgramSynthesis.Validation;
using Xunit;

namespace AiDotNet.Tests.Storage;

/// <summary>
/// Tests for the opt-in precise SQL syntax validator from the
/// <c>AiDotNet.Storage.Sqlite</c> metapackage (audit-2026-05 phase 2c, #1427).
/// The validator prepares statements against a fresh in-memory SQLite engine, so the
/// critical contract is that SYNTAX validation is decoupled from SCHEMA resolution:
/// a syntactically valid statement referencing tables/columns/functions absent from
/// the empty scratch database must still validate (SQLite reports both failure
/// classes as SQLITE_ERROR, distinguishable only by message at prepare time).
/// </summary>
public class SqliteSqlSyntaxValidatorTests
{
    private readonly SqliteSqlSyntaxValidator _validator = new();

    [Theory]
    [InlineData("SELECT 1")]
    [InlineData("SELECT 1 + 2 AS total")]
    [InlineData("VALUES (1, 'a'), (2, 'b')")]
    public void IsValidSql_SchemaFreeStatements_AreValid(string sql)
    {
        Assert.True(_validator.IsValidSql(sql));
    }

    /// <summary>
    /// The regression this validator shipped with: Prepare() runs against an EMPTY
    /// in-memory database, so these used to fail with "no such table" and the
    /// validator rejected perfectly valid SQL. Schema-resolution failures mean the
    /// parse succeeded — they must count as valid syntax.
    /// </summary>
    [Theory]
    [InlineData("SELECT * FROM users")]
    [InlineData("SELECT id, name FROM customers WHERE age > 21 ORDER BY name")]
    [InlineData("SELECT u.id FROM users u JOIN orders o ON o.user_id = u.id")]
    [InlineData("INSERT INTO logs (message) VALUES ('hello')")]
    [InlineData("UPDATE accounts SET balance = balance - 10 WHERE id = 1")]
    [InlineData("DELETE FROM sessions WHERE expires_at < 0")]
    public void IsValidSql_ValidSqlReferencingAbsentSchema_IsValid(string sql)
    {
        Assert.True(_validator.IsValidSql(sql),
            $"Syntactically valid SQL must not be rejected because the scratch database lacks the referenced objects: {sql}");
    }

    [Theory]
    [InlineData("SELECT * FORM users")]                  // typo'd keyword → parse error
    [InlineData("SELEC 1")]                              // typo'd keyword
    [InlineData("SELECT FROM WHERE")]                    // malformed clause structure
    [InlineData("INSERT INTO (a, b) VALUES (1)")]        // missing table name
    [InlineData("UPDATE SET x = 1")]                     // missing table name
    [InlineData("complete gibberish !!!")]
    public void IsValidSql_GenuineSyntaxErrors_AreInvalid(string sql)
    {
        Assert.False(_validator.IsValidSql(sql), $"Genuinely invalid SQL must be rejected: {sql}");
    }

    /// <summary>
    /// Empty/null input is treated as vacuously valid — SQLite prepares an empty
    /// statement list without error, matching the generic structural fallback
    /// (<c>ValidateGenericSource</c>), which also accepts empty input. The two
    /// validators must agree so registering the precise validator never flips the
    /// synthesizer's accept/reject decision on degenerate input.
    /// </summary>
    [Theory]
    [InlineData("")]
    [InlineData(null)]
    public void IsValidSql_EmptyOrNullInput_MatchesGenericFallbackConvention(string? sql)
    {
        Assert.True(_validator.IsValidSql(sql!));
    }
}
