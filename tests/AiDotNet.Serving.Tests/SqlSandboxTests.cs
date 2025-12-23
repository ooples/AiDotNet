using System.Net;
using System.Net.Http.Json;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using Xunit;

namespace AiDotNet.Serving.Tests;

public class SqlSandboxTests : IClassFixture<SqlSandboxTestFactory>
{
    private readonly HttpClient _client;

    public SqlSandboxTests(SqlSandboxTestFactory factory)
    {
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task ExecuteSql_WithSQLiteAndRequestScopedSchema_ReturnsRows()
    {
        var request = new SqlExecuteRequest
        {
            Dialect = SqlDialect.SQLite,
            SchemaSql = "CREATE TABLE t (id INTEGER, name TEXT);",
            SeedSql = "INSERT INTO t (id, name) VALUES (1, 'a');",
            Query = "SELECT id, name FROM t ORDER BY id"
        };

        var response = await _client.PostAsJsonAsync("/api/program-synthesis/sql/execute", request);
        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<SqlExecuteResponse>();
        Assert.NotNull(result);
        Assert.True(result.Success);
        Assert.Equal(SqlDialect.SQLite, result.Dialect);
        Assert.Contains("id", result.Columns);
        Assert.Contains("name", result.Columns);
        Assert.Single(result.Rows);

        var row = result.Rows[0];
        Assert.True(row.ContainsKey("id"));
        Assert.True(row.ContainsKey("name"));
        Assert.Equal(SqlValueKind.Integer, row["id"].Kind);
        Assert.Equal(1, row["id"].IntegerValue);
        Assert.Equal(SqlValueKind.Text, row["name"].Kind);
        Assert.Equal("a", row["name"].TextValue);
    }

    [Fact]
    public async Task ExecuteSql_WithPostgresWithoutConfiguration_ReturnsBadRequest()
    {
        var request = new SqlExecuteRequest
        {
            Dialect = SqlDialect.Postgres,
            Query = "SELECT 1"
        };

        var response = await _client.PostAsJsonAsync("/api/program-synthesis/sql/execute", request);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

        var result = await response.Content.ReadFromJsonAsync<SqlExecuteResponse>();
        Assert.NotNull(result);
        Assert.False(result.Success);
        Assert.Equal(SqlDialect.Postgres, result.Dialect);
        Assert.Equal(SqlExecuteErrorCode.DialectNotConfigured, result.ErrorCode);
    }
}
