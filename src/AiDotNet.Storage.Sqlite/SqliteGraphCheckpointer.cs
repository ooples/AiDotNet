using System.Collections.Generic;
using System.Data.Common;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Data.Sqlite;
using Newtonsoft.Json;

namespace AiDotNet.Agentic.Graph.Checkpointing;

/// <summary>
/// A SQLite-backed <see cref="IGraphCheckpointer{TState}"/> that persists graph checkpoints in a database
/// table, giving durable resume and time-travel across process restarts and machines. Ships in the opt-in
/// <c>AiDotNet.Storage.Sqlite</c> package so the core stays free of the SQLite dependency.
/// </summary>
/// <typeparam name="TState">The graph's state type (serialized to JSON for storage).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Saves your graph's progress into a SQLite database file. Point it at a
/// connection string and runs survive restarts — resume or rewind by thread id.
/// </para>
/// </remarks>
public sealed class SqliteGraphCheckpointer<TState> : IGraphCheckpointer<TState>
{
    private readonly string _connectionString;

    /// <summary>
    /// Initializes the checkpointer with a SQLite connection string (e.g. <c>Data Source=graph.db</c>).
    /// </summary>
    /// <param name="connectionString">The SQLite connection string.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="connectionString"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="connectionString"/> is empty/whitespace.</exception>
    public SqliteGraphCheckpointer(string connectionString)
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
        command.CommandText =
            "INSERT INTO GraphCheckpoints (ThreadId, CheckpointId, Step, NextNode, StateJson) " +
            "VALUES ($threadId, $checkpointId, $step, $nextNode, $stateJson);";
        command.Parameters.AddWithValue("$threadId", checkpoint.ThreadId);
        command.Parameters.AddWithValue("$checkpointId", checkpoint.CheckpointId);
        command.Parameters.AddWithValue("$step", checkpoint.Step);
        command.Parameters.AddWithValue("$nextNode", checkpoint.NextNode);
        command.Parameters.AddWithValue("$stateJson", JsonConvert.SerializeObject(checkpoint.State));
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
            "SELECT CheckpointId, Step, NextNode, StateJson FROM GraphCheckpoints " +
            "WHERE ThreadId = $threadId ORDER BY Seq DESC LIMIT 1;";
        command.Parameters.AddWithValue("$threadId", threadId);
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
            "SELECT CheckpointId, Step, NextNode, StateJson FROM GraphCheckpoints " +
            "WHERE ThreadId = $threadId AND CheckpointId = $checkpointId ORDER BY Seq DESC LIMIT 1;";
        command.Parameters.AddWithValue("$threadId", threadId);
        command.Parameters.AddWithValue("$checkpointId", checkpointId);
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
            "SELECT CheckpointId, Step, NextNode, StateJson FROM GraphCheckpoints " +
            "WHERE ThreadId = $threadId ORDER BY Seq ASC;";
        command.Parameters.AddWithValue("$threadId", threadId);
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

    private async Task<SqliteConnection> OpenAsync(CancellationToken cancellationToken)
    {
        var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(cancellationToken).ConfigureAwait(false);

        using var command = connection.CreateCommand();
        command.CommandText =
            "CREATE TABLE IF NOT EXISTS GraphCheckpoints (" +
            "Seq INTEGER PRIMARY KEY AUTOINCREMENT, " +
            "ThreadId TEXT NOT NULL, " +
            "CheckpointId TEXT NOT NULL, " +
            "Step INTEGER NOT NULL, " +
            "NextNode TEXT NOT NULL, " +
            "StateJson TEXT NOT NULL);" +
            "CREATE INDEX IF NOT EXISTS IX_GraphCheckpoints_Thread ON GraphCheckpoints (ThreadId, Seq);";
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);

        return connection;
    }
}
