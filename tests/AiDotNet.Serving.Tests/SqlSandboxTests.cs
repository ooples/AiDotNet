using System.Net;
using System.Text;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Serialization;
using Xunit;
using System.Threading.Tasks;

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

    [Fact(Timeout = 60000)]
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

    [Fact(Timeout = 60000)]
    public async Task ExecuteSql_WithSQLiteAttachOfHostDatabase_CannotReadHostData()
    {
        // A pre-existing database file on the server (stand-in for Serving's own persistence DB).
        var hostDb = NewTempPath();
        try
        {
            await using (var setup = new Microsoft.Data.Sqlite.SqliteConnection($"Data Source={hostDb};Pooling=False"))
            {
                await setup.OpenAsync();
                await using var cmd = setup.CreateCommand();
                cmd.CommandText = "CREATE TABLE secrets (v TEXT); INSERT INTO secrets VALUES ('host-secret-value');";
                await cmd.ExecuteNonQueryAsync();
            }

            var request = new SqlExecuteRequest
            {
                Dialect = SqlDialect.SQLite,
                SchemaSql = $"ATTACH DATABASE '{hostDb}' AS host;",
                Query = "SELECT v FROM host.secrets"
            };

            var response = await PostAsJsonAsync("/api/program-synthesis/sql/execute", request);
            var body = await response.Content.ReadAsStringAsync();

            Assert.False(response.IsSuccessStatusCode, body);
            Assert.DoesNotContain("host-secret-value", body, StringComparison.Ordinal);
            var result = JsonConvert.DeserializeObject<SqlExecuteResponse>(body, JsonSettings);
            Assert.NotNull(result);
            Assert.False(result.Success);
        }
        finally
        {
            TryDelete(hostDb);
        }
    }

    [Theory(Timeout = 60000)]
    [InlineData("query-attach")]
    [InlineData("script-attach-write")]
    [InlineData("vacuum-into")]
    public async Task ExecuteSql_WithSQLite_CannotCreateFilesOnHost(string vector)
    {
        var target = NewTempPath();
        try
        {
            var request = vector switch
            {
                "query-attach" => new SqlExecuteRequest
                {
                    Dialect = SqlDialect.SQLite,
                    Query = $"ATTACH DATABASE '{target}' AS outside"
                },
                "script-attach-write" => new SqlExecuteRequest
                {
                    Dialect = SqlDialect.SQLite,
                    SchemaSql = $"ATTACH DATABASE '{target}' AS outside; CREATE TABLE outside.dropped (payload TEXT);",
                    SeedSql = "INSERT INTO outside.dropped VALUES ('attacker-controlled');",
                    Query = "SELECT 1"
                },
                _ => new SqlExecuteRequest
                {
                    Dialect = SqlDialect.SQLite,
                    SchemaSql = "CREATE TABLE t (payload TEXT); INSERT INTO t VALUES ('attacker-controlled');",
                    Query = $"VACUUM INTO '{target}'"
                }
            };

            var response = await PostAsJsonAsync("/api/program-synthesis/sql/execute", request);
            var result = await ReadFromJsonAsync<SqlExecuteResponse>(response.Content);

            Assert.False(File.Exists(target), $"{vector}: sandboxed SQL created a file on the host at {target}.");
            Assert.NotNull(result);
            Assert.False(result.Success);
        }
        finally
        {
            TryDelete(target);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task ExecuteSql_WithSQLiteMultiStatementScripts_StillWorks()
    {
        var request = new SqlExecuteRequest
        {
            Dialect = SqlDialect.SQLite,
            SchemaSql = "CREATE TABLE a (id INTEGER); CREATE TABLE b (id INTEGER, a_id INTEGER);",
            SeedSql = "INSERT INTO a VALUES (1); INSERT INTO a VALUES (2); INSERT INTO b VALUES (10, 2);",
            Query = "WITH j AS (SELECT a.id AS aid, b.id AS bid FROM a JOIN b ON b.a_id = a.id) SELECT aid, bid FROM j"
        };

        var response = await PostAsJsonAsync("/api/program-synthesis/sql/execute", request);
        response.EnsureSuccessStatusCode();

        var result = await ReadFromJsonAsync<SqlExecuteResponse>(response.Content);
        Assert.NotNull(result);
        Assert.True(result.Success);
        var row = Assert.Single(result.Rows);
        Assert.Equal(2, row["aid"].IntegerValue);
        Assert.Equal(10, row["bid"].IntegerValue);
    }

    private static string NewTempPath() =>
        Path.Combine(Path.GetTempPath(), $"aidotnet-sql-escape-{Guid.NewGuid():N}.db");

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (IOException)
        {
            // Best-effort cleanup of a temp file.
        }
    }

    [Fact(Timeout = 60000)]
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
