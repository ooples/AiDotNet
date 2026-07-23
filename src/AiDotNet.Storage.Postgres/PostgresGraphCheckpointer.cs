using System.Collections.Generic;
using System.Data.Common;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Npgsql;

namespace AiDotNet.Agentic.Graph.Checkpointing;

/// <summary>
/// A PostgreSQL-backed <see cref="IGraphCheckpointer{TState}"/> that persists graph checkpoints to a table,
/// giving durable resume and time-travel across processes and machines with a shared, scalable database.
/// Ships in the opt-in <c>AiDotNet.Storage.Postgres</c> package so the core stays free of the Npgsql
/// dependency.
/// </summary>
/// <typeparam name="TState">The graph's state type (serialized to JSON for storage).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Saves your graph's progress into a Postgres database. Many processes can share
/// one server, so runs resume or rewind by thread id from anywhere. The table is created automatically.
/// </para>
/// </remarks>
public sealed class PostgresGraphCheckpointer<TState> : IGraphCheckpointer<TState>
{
    private readonly string _connectionString;

    // Schema DDL runs once per checkpointer instance, not on every open: the
    // CREATE TABLE/INDEX IF NOT EXISTS round trips and catalog locks would
    // otherwise sit on the hot path of every save and lookup.
    private readonly SemaphoreSlim _schemaGate = new(1, 1);
    private volatile bool _schemaReady;

    /// <summary>
    /// Initializes the checkpointer with a Postgres connection string.
    /// </summary>
    /// <param name="connectionString">The Npgsql connection string.</param>
    /// <exception cref="ArgumentException">Thrown when <paramref name="connectionString"/> is empty/whitespace.</exception>
    public PostgresGraphCheckpointer(string connectionString)
    {
        if (string.IsNullOrWhiteSpace(connectionString))
        {
            throw new ArgumentException("Connection string cannot be empty.", nameof(connectionString));
        }

        _connectionString = connectionString;
    }

    /// <inheritdoc/>
    public async Task SaveAsync(GraphCheckpoint<TState> checkpoint, CancellationToken cancellationToken = default)
    {
        if (checkpoint is null)
        {
            throw new ArgumentNullException(nameof(checkpoint));
        }

        using var connection = await OpenAsync(cancellationToken).ConfigureAwait(false);
        using var command = connection.CreateCommand();
        // Idempotent write: a retry after a transient failure must update the
        // same logical checkpoint, not insert a duplicate row that GetHistory
        // would then return twice.
        command.CommandText =
            "INSERT INTO graph_checkpoints (thread_id, checkpoint_id, step, next_node, state_json) " +
            "VALUES (@thread_id, @checkpoint_id, @step, @next_node, @state_json) " +
            "ON CONFLICT (thread_id, checkpoint_id) DO UPDATE SET " +
            "step = EXCLUDED.step, next_node = EXCLUDED.next_node, state_json = EXCLUDED.state_json;";
        command.Parameters.AddWithValue("thread_id", checkpoint.ThreadId);
        command.Parameters.AddWithValue("checkpoint_id", checkpoint.CheckpointId);
        command.Parameters.AddWithValue("step", checkpoint.Step);
        command.Parameters.AddWithValue("next_node", checkpoint.NextNode);
        command.Parameters.AddWithValue("state_json", JsonConvert.SerializeObject(checkpoint.State));
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async Task<GraphCheckpoint<TState>?> GetLatestAsync(string threadId, CancellationToken cancellationToken = default)
    {
        if (threadId is null)
        {
            throw new ArgumentNullException(nameof(threadId));
        }

        using var connection = await OpenAsync(cancellationToken).ConfigureAwait(false);
        using var command = connection.CreateCommand();
        command.CommandText =
            "SELECT checkpoint_id, step, next_node, state_json FROM graph_checkpoints " +
            "WHERE thread_id = @thread_id ORDER BY seq DESC LIMIT 1;";
        command.Parameters.AddWithValue("thread_id", threadId);
        using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        if (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            return ReadRow(threadId, reader);
        }

        return null;
    }

    /// <inheritdoc/>
    public async Task<GraphCheckpoint<TState>?> GetAsync(string threadId, string checkpointId, CancellationToken cancellationToken = default)
    {
        if (threadId is null)
        {
            throw new ArgumentNullException(nameof(threadId));
        }

        if (checkpointId is null)
        {
            throw new ArgumentNullException(nameof(checkpointId));
        }

        using var connection = await OpenAsync(cancellationToken).ConfigureAwait(false);
        using var command = connection.CreateCommand();
        command.CommandText =
            "SELECT checkpoint_id, step, next_node, state_json FROM graph_checkpoints " +
            "WHERE thread_id = @thread_id AND checkpoint_id = @checkpoint_id ORDER BY seq DESC LIMIT 1;";
        command.Parameters.AddWithValue("thread_id", threadId);
        command.Parameters.AddWithValue("checkpoint_id", checkpointId);
        using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        if (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            return ReadRow(threadId, reader);
        }

        return null;
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<GraphCheckpoint<TState>>> GetHistoryAsync(string threadId, CancellationToken cancellationToken = default)
    {
        if (threadId is null)
        {
            throw new ArgumentNullException(nameof(threadId));
        }

        var history = new List<GraphCheckpoint<TState>>();
        using var connection = await OpenAsync(cancellationToken).ConfigureAwait(false);
        using var command = connection.CreateCommand();
        command.CommandText =
            "SELECT checkpoint_id, step, next_node, state_json FROM graph_checkpoints " +
            "WHERE thread_id = @thread_id ORDER BY seq ASC;";
        command.Parameters.AddWithValue("thread_id", threadId);
        using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            history.Add(ReadRow(threadId, reader));
        }

        return history;
    }

    private static GraphCheckpoint<TState> ReadRow(string threadId, DbDataReader reader)
    {
        var checkpointId = reader.GetString(0);
        var step = reader.GetInt32(1);
        var nextNode = reader.GetString(2);
        var stateJson = reader.GetString(3);
        var state = JsonConvert.DeserializeObject<TState>(stateJson);
        return new GraphCheckpoint<TState>(threadId, checkpointId, step, nextNode, state);
    }

    private async Task<NpgsqlConnection> OpenAsync(CancellationToken cancellationToken)
    {
        var connection = new NpgsqlConnection(_connectionString);
        await connection.OpenAsync(cancellationToken).ConfigureAwait(false);
        await EnsureSchemaAsync(connection, cancellationToken).ConfigureAwait(false);
        return connection;
    }

    private async Task EnsureSchemaAsync(NpgsqlConnection connection, CancellationToken cancellationToken)
    {
        if (_schemaReady)
        {
            return;
        }

        await _schemaGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_schemaReady)
            {
                return;
            }

            using var command = connection.CreateCommand();
            command.CommandText =
                "CREATE TABLE IF NOT EXISTS graph_checkpoints (" +
                "seq BIGSERIAL PRIMARY KEY, " +
                "thread_id TEXT NOT NULL, " +
                "checkpoint_id TEXT NOT NULL, " +
                "step INTEGER NOT NULL, " +
                "next_node TEXT NOT NULL, " +
                "state_json TEXT NOT NULL);" +
                "CREATE INDEX IF NOT EXISTS ix_graph_checkpoints_thread ON graph_checkpoints (thread_id, seq);" +
                // Backs the idempotent ON CONFLICT upsert in SaveAsync.
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_graph_checkpoints_thread_checkpoint " +
                "ON graph_checkpoints (thread_id, checkpoint_id);";
            await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            _schemaReady = true;
        }
        finally
        {
            _schemaGate.Release();
        }
    }
}
