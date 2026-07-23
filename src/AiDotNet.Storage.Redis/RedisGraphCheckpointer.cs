using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using StackExchange.Redis;

namespace AiDotNet.Agentic.Graph.Checkpointing;

/// <summary>
/// A Redis-backed <see cref="IGraphCheckpointer{TState}"/> that stores each thread's checkpoints as a Redis
/// list, giving fast, distributed durable resume and time-travel. Ships in the opt-in
/// <c>AiDotNet.Storage.Redis</c> package so the core stays free of the StackExchange.Redis dependency.
/// </summary>
/// <typeparam name="TState">The graph's state type (serialized to JSON for storage).</typeparam>
/// <remarks>
/// <para>
/// Each checkpoint is appended (RPUSH) to the list at <c>aidotnet:graph:checkpoints:{threadId}</c>, so the
/// latest is the tail and history is the full range — preserving step order. Construct with a connection
/// string; the client is owned and disposed by this instance.
/// </para>
/// <para><b>For Beginners:</b> Saves your graph's progress into Redis (a fast in-memory data store many
/// servers can share). Great for high-throughput, distributed agents that need to resume from anywhere.
/// </para>
/// </remarks>
public sealed class RedisGraphCheckpointer<TState> : IGraphCheckpointer<TState>, IDisposable
{
    private const string KeyPrefix = "aidotnet:graph:checkpoints:";

    private readonly IConnectionMultiplexer _connection;
    private readonly bool _ownsConnection;

    /// <summary>
    /// Initializes the checkpointer from a Redis connection string (e.g. <c>localhost:6379</c>).
    /// </summary>
    /// <param name="connectionString">The StackExchange.Redis connection string.</param>
    /// <exception cref="ArgumentException">Thrown when <paramref name="connectionString"/> is empty/whitespace.</exception>
    public RedisGraphCheckpointer(string connectionString)
    {
        if (string.IsNullOrWhiteSpace(connectionString))
        {
            throw new ArgumentException("Connection string cannot be empty.", nameof(connectionString));
        }

        _connection = ConnectionMultiplexer.Connect(connectionString);
        _ownsConnection = true;
    }

    /// <summary>
    /// Initializes the checkpointer over an existing connection multiplexer (not owned/disposed by this instance).
    /// </summary>
    /// <param name="connection">The shared connection multiplexer.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="connection"/> is <c>null</c>.</exception>
    public RedisGraphCheckpointer(IConnectionMultiplexer connection)
    {
        _connection = connection ?? throw new ArgumentNullException(nameof(connection));
        _ownsConnection = false;
    }

    /// <inheritdoc/>
    public async Task SaveAsync(GraphCheckpoint<TState> checkpoint, CancellationToken cancellationToken = default)
    {
        if (checkpoint is null)
        {
            throw new ArgumentNullException(nameof(checkpoint));
        }

        var record = new CheckpointRecord
        {
            CheckpointId = checkpoint.CheckpointId,
            Step = checkpoint.Step,
            NextNode = checkpoint.NextNode,
            StateJson = JsonConvert.SerializeObject(checkpoint.State),
        };

        await _connection.GetDatabase()
            .ListRightPushAsync(Key(checkpoint.ThreadId), JsonConvert.SerializeObject(record))
            .ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async Task<GraphCheckpoint<TState>?> GetLatestAsync(string threadId, CancellationToken cancellationToken = default)
    {
        if (threadId is null)
        {
            throw new ArgumentNullException(nameof(threadId));
        }

        var value = await _connection.GetDatabase().ListGetByIndexAsync(Key(threadId), -1).ConfigureAwait(false);
        return value.IsNull ? null : ToCheckpoint(threadId, value);
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

        var values = await _connection.GetDatabase().ListRangeAsync(Key(threadId)).ConfigureAwait(false);
        for (var i = values.Length - 1; i >= 0; i--)
        {
            var checkpoint = ToCheckpoint(threadId, values[i]);
            if (checkpoint is not null && string.Equals(checkpoint.CheckpointId, checkpointId, StringComparison.Ordinal))
            {
                return checkpoint;
            }
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

        var values = await _connection.GetDatabase().ListRangeAsync(Key(threadId)).ConfigureAwait(false);
        var history = new List<GraphCheckpoint<TState>>(values.Length);
        foreach (var value in values)
        {
            var checkpoint = ToCheckpoint(threadId, value);
            if (checkpoint is not null)
            {
                history.Add(checkpoint);
            }
        }

        return history;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsConnection)
        {
            _connection.Dispose();
        }
    }

    private static string Key(string threadId) => KeyPrefix + threadId;

    private static GraphCheckpoint<TState>? ToCheckpoint(string threadId, RedisValue value)
    {
        if (value.IsNull)
        {
            return null;
        }

        var record = JsonConvert.DeserializeObject<CheckpointRecord>(value.ToString());
        if (record is null)
        {
            return null;
        }

        var state = JsonConvert.DeserializeObject<TState>(record.StateJson);
        return new GraphCheckpoint<TState>(threadId, record.CheckpointId, record.Step, record.NextNode, state);
    }

    private sealed class CheckpointRecord
    {
        public string CheckpointId { get; set; } = string.Empty;

        public int Step { get; set; }

        public string NextNode { get; set; } = string.Empty;

        public string StateJson { get; set; } = string.Empty;
    }
}
