using System.Data.Common;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Sandboxing.Docker;
using AiDotNet.Serving.Security;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Options;
using MySqlConnector;
using Npgsql;
using AiDotNet.Validation;

namespace AiDotNet.Serving.Sandboxing.Sql;

public sealed class SqlSandboxExecutor : ISqlSandboxExecutor
{
    private readonly ServingSqlSandboxOptions _options;
    private readonly IDockerRunner _dockerRunner;
    private readonly ILogger<SqlSandboxExecutor> _logger;
    private readonly IReadOnlyDictionary<string, ServingSqlDbContextRegistration> _dbContextsById;
    private readonly IReadOnlyDictionary<string, ServingSqlDatasetRegistration> _datasetsById;

    public SqlSandboxExecutor(
        IOptions<ServingSqlSandboxOptions> options,
        IDockerRunner dockerRunner,
        ILogger<SqlSandboxExecutor> logger)
    {
        Guard.NotNull(options);
        _options = options.Value;
        Guard.NotNull(dockerRunner);
        _dockerRunner = dockerRunner;
        Guard.NotNull(logger);
        _logger = logger;

        _dbContextsById = CreateLookup(_options.DbContexts, nameof(_options.DbContexts), x => x.Id);
        _datasetsById = CreateLookup(_options.Datasets, nameof(_options.Datasets), x => x.Id);

        foreach (var ctx in _dbContextsById.Values)
        {
            if (string.IsNullOrWhiteSpace(ctx.ConnectionString))
            {
                throw new InvalidOperationException("ServingSqlSandbox:DbContexts contains an entry with an empty ConnectionString.");
            }
        }
    }

    public async Task<SqlExecuteResponse> ExecuteAsync(
        SqlExecuteRequest request,
        ServingRequestContext requestContext,
        CancellationToken cancellationToken)
    {
        if (request is null) throw new ArgumentNullException(nameof(request));
        if (requestContext is null) throw new ArgumentNullException(nameof(requestContext));

        var defaultDialect = _options.DefaultDialect;
        ServingSqlDatasetRegistration? dataset = null;
        if (!string.IsNullOrWhiteSpace(request.DatasetId))
        {
            var datasetId = request.DatasetId.Trim();
            if (!_datasetsById.TryGetValue(datasetId, out dataset))
            {
                return new SqlExecuteResponse
                {
                    Success = false,
                    Dialect = request.Dialect ?? defaultDialect,
                    Error = "Unknown DatasetId.",
                    ErrorCode = SqlExecuteErrorCode.UnknownDatasetId
                };
            }

            if (request.Dialect.HasValue && request.Dialect.Value != dataset.Dialect)
            {
                return new SqlExecuteResponse
                {
                    Success = false,
                    Dialect = request.Dialect.Value,
                    Error = "Dialect does not match the requested DatasetId.",
                    ErrorCode = SqlExecuteErrorCode.DialectMismatch
                };
            }
        }

        var dbId = !string.IsNullOrWhiteSpace(request.DbId)
            ? request.DbId.Trim()
            : dataset?.DbId?.Trim();

        ServingSqlDbContextRegistration? dbContext = null;
        if (!string.IsNullOrWhiteSpace(dbId))
        {
            if (!_dbContextsById.TryGetValue(dbId, out dbContext))
            {
                return new SqlExecuteResponse
                {
                    Success = false,
                    Dialect = request.Dialect ?? dataset?.Dialect ?? defaultDialect,
                    Error = "Unknown DbId.",
                    ErrorCode = SqlExecuteErrorCode.UnknownDbId
                };
            }

            if (request.Dialect.HasValue && request.Dialect.Value != dbContext.Dialect)
            {
                return new SqlExecuteResponse
                {
                    Success = false,
                    Dialect = request.Dialect.Value,
                    Error = "Dialect does not match the requested DbId.",
                    ErrorCode = SqlExecuteErrorCode.DialectMismatch
                };
            }

            if (dataset is not null && dataset.Dialect != dbContext.Dialect)
            {
                return new SqlExecuteResponse
                {
                    Success = false,
                    Dialect = dataset.Dialect,
                    Error = "Dialect does not match the requested DatasetId.",
                    ErrorCode = SqlExecuteErrorCode.DialectMismatch
                };
            }
        }

        var dialect = request.Dialect ?? dataset?.Dialect ?? dbContext?.Dialect ?? defaultDialect;

        var schemaSql = CombineSqlScripts(dataset?.SchemaSql, request.SchemaSql);
        var seedSql = CombineSqlScripts(dataset?.SeedSql, request.SeedSql);

        if (string.IsNullOrWhiteSpace(request.Query))
        {
            return new SqlExecuteResponse
            {
                Success = false,
                Dialect = dialect,
                Error = "Query is required.",
                ErrorCode = SqlExecuteErrorCode.QueryRequired
            };
        }

        try
        {
            return dialect switch
            {
                SqlDialect.SQLite => await ExecuteSqliteAsync(request.Query, schemaSql, seedSql, cancellationToken).ConfigureAwait(false),
                SqlDialect.Postgres => await ExecutePostgresAsync(request.Query, schemaSql, seedSql, dbContext?.ConnectionString, cancellationToken).ConfigureAwait(false),
                SqlDialect.MySql => await ExecuteMySqlAsync(request.Query, schemaSql, seedSql, dbContext?.ConnectionString, cancellationToken).ConfigureAwait(false),
                _ => new SqlExecuteResponse
                {
                    Success = false,
                    Dialect = dialect,
                    Error = "Unsupported SQL dialect.",
                    ErrorCode = SqlExecuteErrorCode.UnsupportedDialect
                }
            };
        }
        catch (OperationCanceledException)
        {
            return new SqlExecuteResponse
            {
                Success = false,
                Dialect = dialect,
                Error = "SQL execution timed out or was canceled.",
                ErrorCode = SqlExecuteErrorCode.TimeoutOrCanceled
            };
        }
        catch (InvalidOperationException ex) when (
            ex.Message.Contains("is not configured and Docker fallback is disabled.", StringComparison.Ordinal))
        {
            return new SqlExecuteResponse
            {
                Success = false,
                Dialect = dialect,
                Error = $"SQL dialect {dialect} is not configured on this Serving instance.",
                ErrorCode = SqlExecuteErrorCode.DialectNotConfigured
            };
        }
        catch (InvalidOperationException ex) when (string.Equals(ex.Message, "Only a single SQL statement is allowed per request.", StringComparison.Ordinal))
        {
            return new SqlExecuteResponse
            {
                Success = false,
                Dialect = dialect,
                Error = ex.Message,
                ErrorCode = SqlExecuteErrorCode.MultiStatementNotAllowed
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "SQL execution failed (Dialect={Dialect}).", dialect);
            return new SqlExecuteResponse
            {
                Success = false,
                Dialect = dialect,
                Error = "SQL execution failed.",
                ErrorCode = SqlExecuteErrorCode.ExecutionFailed
            };
        }
    }

    private async Task<SqlExecuteResponse> ExecuteSqliteAsync(
        string query,
        string? schemaSql,
        string? seedSql,
        CancellationToken cancellationToken)
    {
        var connectionString = "Data Source=:memory:";

        await using var connection = new SqliteConnection(connectionString);
        await connection.OpenAsync(cancellationToken).ConfigureAwait(false);

        await ExecuteOptionalScriptAsync(connection, schemaSql, cancellationToken).ConfigureAwait(false);
        await ExecuteOptionalScriptAsync(connection, seedSql, cancellationToken).ConfigureAwait(false);

        return await ExecuteQueryAsync(
            dialect: SqlDialect.SQLite,
            createCommand: sql => CreateSqliteCommand(connection, sql),
            query: query,
            cancellationToken: cancellationToken).ConfigureAwait(false);
    }

    private async Task<SqlExecuteResponse> ExecutePostgresAsync(
        string query,
        string? schemaSql,
        string? seedSql,
        string? connectionStringOverride,
        CancellationToken cancellationToken)
    {
        var (connectionString, cleanup) = await GetPostgresConnectionStringAsync(connectionStringOverride, cancellationToken).ConfigureAwait(false);

        try
        {
            await using var connection = new NpgsqlConnection(connectionString);
            await connection.OpenAsync(cancellationToken).ConfigureAwait(false);

            var schemaName = $"aidotnet_{Guid.NewGuid():N}";

            await using var transaction = await connection.BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

            await using (var createSchema = CreateNpgsqlCommand(connection, $"CREATE SCHEMA \"{schemaName}\";"))
            {
                await createSchema.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }

            await using (var setPath = CreateNpgsqlCommand(connection, $"SET search_path TO \"{schemaName}\";"))
            {
                await setPath.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }

            await ExecuteOptionalScriptAsync(connection, schemaSql, cancellationToken).ConfigureAwait(false);
            await ExecuteOptionalScriptAsync(connection, seedSql, cancellationToken).ConfigureAwait(false);

            var result = await ExecuteQueryAsync(
                dialect: SqlDialect.Postgres,
                createCommand: sql => CreateNpgsqlCommand(connection, sql),
                query: query,
                cancellationToken: cancellationToken).ConfigureAwait(false);

            await using (var dropSchema = CreateNpgsqlCommand(connection, $"DROP SCHEMA IF EXISTS \"{schemaName}\" CASCADE;"))
            {
                await dropSchema.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }

            await transaction.CommitAsync(cancellationToken).ConfigureAwait(false);

            return result;
        }
        finally
        {
            if (cleanup is not null)
            {
                await cleanup().ConfigureAwait(false);
            }
        }
    }

    private async Task<SqlExecuteResponse> ExecuteMySqlAsync(
        string query,
        string? schemaSql,
        string? seedSql,
        string? connectionStringOverride,
        CancellationToken cancellationToken)
    {
        var (connectionString, cleanup) = await GetMySqlConnectionStringAsync(connectionStringOverride, cancellationToken).ConfigureAwait(false);

        try
        {
            await using var connection = new MySqlConnection(connectionString);
            await connection.OpenAsync(cancellationToken).ConfigureAwait(false);

            var databaseName = $"aidotnet_{Guid.NewGuid():N}";

            await using (var createDb = CreateMySqlCommand(connection, $"CREATE DATABASE `{databaseName}`;"))
            {
                await createDb.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }

            connection.ChangeDatabase(databaseName);

            await ExecuteOptionalScriptAsync(connection, schemaSql, cancellationToken).ConfigureAwait(false);
            await ExecuteOptionalScriptAsync(connection, seedSql, cancellationToken).ConfigureAwait(false);

            var result = await ExecuteQueryAsync(
                dialect: SqlDialect.MySql,
                createCommand: sql => CreateMySqlCommand(connection, sql),
                query: query,
                cancellationToken: cancellationToken).ConfigureAwait(false);

            await using (var dropDb = CreateMySqlCommand(connection, $"DROP DATABASE IF EXISTS `{databaseName}`;"))
            {
                await dropDb.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }

            return result;
        }
        finally
        {
            if (cleanup is not null)
            {
                await cleanup().ConfigureAwait(false);
            }
        }
    }

    private async Task<(string ConnectionString, Func<Task>? Cleanup)> GetPostgresConnectionStringAsync(
        string? connectionStringOverride,
        CancellationToken cancellationToken)
    {
        if (!string.IsNullOrWhiteSpace(connectionStringOverride))
        {
            return (connectionStringOverride!, null);
        }

        if (!string.IsNullOrWhiteSpace(_options.PostgresConnectionString))
        {
            return (_options.PostgresConnectionString!, null);
        }

        if (!_options.EnableDockerFallback)
        {
            throw new InvalidOperationException("PostgresConnectionString is not configured and Docker fallback is disabled.");
        }

        return await StartPostgresDockerAsync(cancellationToken).ConfigureAwait(false);
    }

    private async Task<(string ConnectionString, Func<Task>? Cleanup)> GetMySqlConnectionStringAsync(
        string? connectionStringOverride,
        CancellationToken cancellationToken)
    {
        if (!string.IsNullOrWhiteSpace(connectionStringOverride))
        {
            return (connectionStringOverride!, null);
        }

        if (!string.IsNullOrWhiteSpace(_options.MySqlConnectionString))
        {
            return (_options.MySqlConnectionString!, null);
        }

        if (!_options.EnableDockerFallback)
        {
            throw new InvalidOperationException("MySqlConnectionString is not configured and Docker fallback is disabled.");
        }

        return await StartMySqlDockerAsync(cancellationToken).ConfigureAwait(false);
    }

    private async Task<(string ConnectionString, Func<Task>? Cleanup)> StartPostgresDockerAsync(CancellationToken cancellationToken)
    {
        var name = $"aidotnet-pg-{Guid.NewGuid():N}";
        var password = DockerPasswordGenerator.Generate();
        var envFile = CreateTempEnvFile("aidotnet-pg-env", new Dictionary<string, string>
        {
            ["POSTGRES_PASSWORD"] = password
        });

        try
        {
            await _dockerRunner.RunAsync(
                $"run -d --rm --name {name} --env-file \"{envFile}\" -p 0:5432 postgres:16-alpine",
                stdIn: null,
                timeout: TimeSpan.FromMinutes(2),
                maxStdOutChars: 16_000,
                maxStdErrChars: 16_000,
                cancellationToken: cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            TryDeleteFile(envFile);
            throw;
        }

        try
        {
            var port = await GetPublishedPortAsync(name, "5432/tcp", cancellationToken).ConfigureAwait(false);

            var csb = new NpgsqlConnectionStringBuilder
            {
                Host = "localhost",
                Port = port,
                Username = "postgres",
                Password = password,
                Database = "postgres",
                Timeout = 5,
                CommandTimeout = _options.CommandTimeoutSeconds
            };

            await WaitForPostgresReadyAsync(csb.ConnectionString, cancellationToken).ConfigureAwait(false);

            return (csb.ConnectionString, Cleanup);
        }
        catch
        {
            await Cleanup().ConfigureAwait(false);
            throw;
        }

        async Task Cleanup()
        {
            try
            {
                await _dockerRunner.RunAsync(
                    $"stop {name}",
                    stdIn: null,
                    timeout: TimeSpan.FromSeconds(30),
                    maxStdOutChars: 16_000,
                    maxStdErrChars: 16_000,
                    cancellationToken: CancellationToken.None).ConfigureAwait(false);
            }
            catch
            {
                // Best-effort cleanup.
            }
            finally
            {
                TryDeleteFile(envFile);
            }
        }
    }

    private async Task<(string ConnectionString, Func<Task>? Cleanup)> StartMySqlDockerAsync(CancellationToken cancellationToken)
    {
        var name = $"aidotnet-mysql-{Guid.NewGuid():N}";
        var password = DockerPasswordGenerator.Generate();
        var envFile = CreateTempEnvFile("aidotnet-mysql-env", new Dictionary<string, string>
        {
            ["MYSQL_ROOT_PASSWORD"] = password
        });

        try
        {
            await _dockerRunner.RunAsync(
                $"run -d --rm --name {name} --env-file \"{envFile}\" -p 0:3306 mysql:8",
                stdIn: null,
                timeout: TimeSpan.FromMinutes(2),
                maxStdOutChars: 16_000,
                maxStdErrChars: 16_000,
                cancellationToken: cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            TryDeleteFile(envFile);
            throw;
        }

        try
        {
            var port = await GetPublishedPortAsync(name, "3306/tcp", cancellationToken).ConfigureAwait(false);

            var csb = new MySqlConnectionStringBuilder
            {
                Server = "localhost",
                Port = (uint)port,
                UserID = "root",
                Password = password,
                ConnectionTimeout = 5,
                DefaultCommandTimeout = (uint)_options.CommandTimeoutSeconds,
                AllowUserVariables = true
            };

            await WaitForMySqlReadyAsync(csb.ConnectionString, cancellationToken).ConfigureAwait(false);

            return (csb.ConnectionString, Cleanup);
        }
        catch
        {
            await Cleanup().ConfigureAwait(false);
            throw;
        }

        async Task Cleanup()
        {
            try
            {
                await _dockerRunner.RunAsync(
                    $"stop {name}",
                    stdIn: null,
                    timeout: TimeSpan.FromSeconds(30),
                    maxStdOutChars: 16_000,
                    maxStdErrChars: 16_000,
                    cancellationToken: CancellationToken.None).ConfigureAwait(false);
            }
            catch
            {
                // Best-effort cleanup.
            }
            finally
            {
                TryDeleteFile(envFile);
            }
        }
    }

    private static string CreateTempEnvFile(string prefix, IReadOnlyDictionary<string, string> values)
    {
        var dir = Path.Combine(Path.GetTempPath(), "aidotnet-sql-sandbox");
        Directory.CreateDirectory(dir);

        var file = Path.Combine(dir, $"{prefix}-{Guid.NewGuid():N}.env");
        var lines = values.Select(kvp => $"{kvp.Key}={kvp.Value}").ToArray();
        File.WriteAllLines(file, lines, System.Text.Encoding.UTF8);
        return file;
    }

    private static void TryDeleteFile(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
            // Best-effort cleanup.
        }
    }

    private async Task<int> GetPublishedPortAsync(string containerName, string containerPort, CancellationToken cancellationToken)
    {
        var result = await _dockerRunner.RunAsync(
            $"port {containerName} {containerPort}",
            stdIn: null,
            timeout: TimeSpan.FromSeconds(30),
            maxStdOutChars: 16_000,
            maxStdErrChars: 16_000,
            cancellationToken: cancellationToken).ConfigureAwait(false);

        if (result.ExitCode != 0)
        {
            throw new InvalidOperationException($"Failed to query docker port mapping: {result.StdErr}");
        }

        var text = result.StdOut.Trim();
        if (string.IsNullOrWhiteSpace(text))
        {
            throw new InvalidOperationException("Docker did not return a published port mapping.");
        }

        // Examples:
        // 0.0.0.0:49153
        // :::49153
        var parts = text.Split(':', StringSplitOptions.RemoveEmptyEntries);
        var portText = parts[^1].Trim();

        if (!int.TryParse(portText, out var port))
        {
            throw new InvalidOperationException($"Failed to parse published port: '{text}'.");
        }

        return port;
    }

    private async Task WaitForPostgresReadyAsync(string connectionString, CancellationToken cancellationToken)
    {
        var deadline = DateTimeOffset.UtcNow.AddSeconds(30);

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();

            try
            {
                await using var conn = new NpgsqlConnection(connectionString);
                await conn.OpenAsync(cancellationToken).ConfigureAwait(false);
                return;
            }
            catch
            {
                if (DateTimeOffset.UtcNow >= deadline)
                {
                    throw;
                }

                await Task.Delay(500, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private async Task WaitForMySqlReadyAsync(string connectionString, CancellationToken cancellationToken)
    {
        var deadline = DateTimeOffset.UtcNow.AddSeconds(45);

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();

            try
            {
                await using var conn = new MySqlConnection(connectionString);
                await conn.OpenAsync(cancellationToken).ConfigureAwait(false);
                return;
            }
            catch
            {
                if (DateTimeOffset.UtcNow >= deadline)
                {
                    throw;
                }

                await Task.Delay(750, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private async Task ExecuteOptionalScriptAsync(DbConnection connection, string? sql, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(sql))
        {
            return;
        }

        await using var cmd = connection.CreateCommand();
        cmd.CommandText = sql;
        cmd.CommandTimeout = _options.CommandTimeoutSeconds;
        await cmd.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    private async Task<SqlExecuteResponse> ExecuteQueryAsync(
        SqlDialect dialect,
        Func<string, DbCommand> createCommand,
        string query,
        CancellationToken cancellationToken)
    {
        var trimmed = query.Trim();
        var safeQuery = NormalizeSingleStatement(trimmed);

        var returnsRows = ReturnsRows(safeQuery);
        var queryToRun = returnsRows && IsSelectLike(safeQuery)
            ? ApplyLimit(dialect, safeQuery, _options.MaxResultRows)
            : safeQuery;

        await using var cmd = createCommand(queryToRun);
        cmd.CommandTimeout = _options.CommandTimeoutSeconds;

        if (!returnsRows)
        {
            await cmd.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            return new SqlExecuteResponse
            {
                Success = true,
                Dialect = dialect
            };
        }

        await using var reader = await cmd.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

        var columns = new List<string>(reader.FieldCount);
        for (int i = 0; i < reader.FieldCount; i++)
        {
            columns.Add(reader.GetName(i));
        }

        var rows = new List<Dictionary<string, SqlValue>>();
        var maxRows = _options.MaxResultRows;
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            var row = new Dictionary<string, SqlValue>(StringComparer.Ordinal);
            for (int i = 0; i < reader.FieldCount; i++)
            {
                var name = columns[i];
                row[name] = ReadSqlValue(reader, i);
            }

            rows.Add(row);

            if (maxRows > 0 && rows.Count >= maxRows)
            {
                break;
            }
        }

        return new SqlExecuteResponse
        {
            Success = true,
            Dialect = dialect,
            Columns = columns,
            Rows = rows
        };
    }

    private static string? CombineSqlScripts(string? first, string? second)
    {
        if (string.IsNullOrWhiteSpace(first))
        {
            return string.IsNullOrWhiteSpace(second) ? null : second;
        }

        if (string.IsNullOrWhiteSpace(second))
        {
            return first;
        }

        return $"{first}\n\n{second}";
    }

    private static IReadOnlyDictionary<string, TItem> CreateLookup<TItem>(
        IEnumerable<TItem> items,
        string collectionName,
        Func<TItem, string> getId)
    {
        var dict = new Dictionary<string, TItem>(StringComparer.OrdinalIgnoreCase);

        foreach (var item in items ?? Array.Empty<TItem>())
        {
            var id = (getId(item) ?? string.Empty).Trim();
            if (string.IsNullOrWhiteSpace(id))
            {
                throw new InvalidOperationException($"{collectionName} contains an entry with an empty Id.");
            }

            if (!dict.TryAdd(id, item))
            {
                throw new InvalidOperationException($"{collectionName} contains a duplicate Id '{id}'.");
            }
        }

        return dict;
    }

    private static SqliteCommand CreateSqliteCommand(SqliteConnection connection, string sql) =>
        new(sql, connection);

    private static NpgsqlCommand CreateNpgsqlCommand(NpgsqlConnection connection, string sql) =>
        new(sql, connection);

    private static MySqlCommand CreateMySqlCommand(MySqlConnection connection, string sql) =>
        new(sql, connection);

    private static SqlValue ReadSqlValue(DbDataReader reader, int ordinal)
    {
        if (reader.IsDBNull(ordinal))
        {
            return new SqlValue { Kind = SqlValueKind.Null };
        }

        var value = reader.GetValue(ordinal);
        return value switch
        {
            bool b => new SqlValue { Kind = SqlValueKind.Boolean, BooleanValue = b },
            byte[] bytes => new SqlValue { Kind = SqlValueKind.Blob, BlobBase64 = Convert.ToBase64String(bytes) },
            sbyte or byte or short or ushort or int or uint or long or ulong =>
                new SqlValue { Kind = SqlValueKind.Integer, IntegerValue = Convert.ToInt64(value) },
            float or double or decimal =>
                new SqlValue { Kind = SqlValueKind.Real, RealValue = Convert.ToDouble(value) },
            _ => new SqlValue { Kind = SqlValueKind.Text, TextValue = value.ToString() }
        };
    }

    private static bool IsSelectLike(string sql)
    {
        var start = sql.TrimStart();
        return start.StartsWith("select", StringComparison.OrdinalIgnoreCase) ||
               start.StartsWith("with", StringComparison.OrdinalIgnoreCase);
    }

    private static bool ReturnsRows(string sql)
    {
        var start = sql.TrimStart();
        return start.StartsWith("select", StringComparison.OrdinalIgnoreCase) ||
               start.StartsWith("with", StringComparison.OrdinalIgnoreCase) ||
               start.StartsWith("pragma", StringComparison.OrdinalIgnoreCase) ||
               start.StartsWith("show", StringComparison.OrdinalIgnoreCase) ||
               start.StartsWith("describe", StringComparison.OrdinalIgnoreCase) ||
               start.StartsWith("explain", StringComparison.OrdinalIgnoreCase);
    }

    private static string ApplyLimit(SqlDialect dialect, string sql, int limit)
    {
        var capped = limit > 0 ? limit : 1000;
        return dialect switch
        {
            SqlDialect.MySql => $"SELECT * FROM ({sql}) AS aidotnet_q LIMIT {capped}",
            _ => $"SELECT * FROM ({sql}) AS aidotnet_q LIMIT {capped}"
        };
    }

    private static string NormalizeSingleStatement(string sql)
    {
        var trimmed = sql.Trim();

        while (trimmed.EndsWith(";", StringComparison.Ordinal))
        {
            trimmed = trimmed[..^1].TrimEnd();
        }

        if (trimmed.Contains(';'))
        {
            throw new InvalidOperationException("Only a single SQL statement is allowed per request.");
        }

        return trimmed;
    }
}
