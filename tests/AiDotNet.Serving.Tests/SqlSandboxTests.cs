using System.Net;
using System.Text;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Serialization;
using Xunit;

namespace AiDotNet.Serving.Tests;

public class SqlSandboxTests : IClassFixture<SqlSandboxTestFactory>
{
    private readonly HttpClient _client;
    private static readonly JsonSerializerSettings JsonSettings = new()
    {
        ContractResolver = new CamelCasePropertyNamesContractResolver(),
        Converters = { new StringEnumConverter(new CamelCaseNamingStrategy(), allowIntegerValues: false) }
    };

    public SqlSandboxTests(SqlSandboxTestFactory factory)
    {
        _client = factory.CreateClient();
    }

    private async Task<HttpResponseMessage> PostAsJsonAsync<T>(string requestUri, T value)
    {
        var json = JsonConvert.SerializeObject(value, JsonSettings);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        return await _client.PostAsync(requestUri, content);
    }

    private static async Task<T?> ReadFromJsonAsync<T>(HttpContent content)
    {
        var json = await content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<T>(json, JsonSettings);
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

        var response = await PostAsJsonAsync("/api/program-synthesis/sql/execute", request);
        response.EnsureSuccessStatusCode();

        var result = await ReadFromJsonAsync<SqlExecuteResponse>(response.Content);
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

        var response = await PostAsJsonAsync("/api/program-synthesis/sql/execute", request);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

        var result = await ReadFromJsonAsync<SqlExecuteResponse>(response.Content);
        Assert.NotNull(result);
        Assert.False(result.Success);
        Assert.Equal(SqlDialect.Postgres, result.Dialect);
        Assert.Equal(SqlExecuteErrorCode.DialectNotConfigured, result.ErrorCode);
    }
}
