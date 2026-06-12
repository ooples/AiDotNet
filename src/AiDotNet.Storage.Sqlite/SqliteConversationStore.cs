using System.Data;
using AiDotNet.Agentic.Memory;
using AiDotNet.Agentic.Models;
using Microsoft.Data.Sqlite;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage type in the core package.
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Storage.Sqlite;

/// <summary>
/// A durable, database-backed <see cref="IConversationStore"/> that persists agent conversation threads to
/// SQLite. Each message is stored as its role and text, ordered within a thread, so conversations survive
/// process restarts and can be shared across processes.
/// </summary>
/// <remarks>
/// <para>
/// This lives in the opt-in <c>AiDotNet.Storage.Sqlite</c> package so the native SQLite dependency stays out
/// of the core library. It mirrors the SQLite graph checkpointer: an auto-incrementing sequence preserves
/// message order, and the schema is created on demand. For single-process durability without a native
/// dependency, the core <c>JsonFileConversationStore</c> is an alternative.
/// </para>
/// <para><b>For Beginners:</b> The same conversation notebook as the in-memory and JSON-file stores, but
/// kept in a real SQLite database file — so it survives restarts and multiple programs can share it.
/// </para>
/// </remarks>
public sealed class SqliteConversationStore : IConversationStore
{
    private readonly string _connectionString;
    // Per connection-string, remember that the schema has been created so we
    // don't pay the CREATE-TABLE-IF-NOT-EXISTS round trip on every operation.
    // SQLite handles concurrent CREATE IF NOT EXISTS safely, but skipping the
    // repeated DDL is a free win once we've initialised.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, bool> _schemaInitialised
        = new(StringComparer.Ordinal);

    /// <summary>
    /// Initializes a new store over the given SQLite connection string. The schema is created automatically
    /// on first use.
    /// </summary>
    /// <param name="connectionString">The SQLite connection string (e.g., <c>Data Source=conversations.db</c>).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="connectionString"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="connectionString"/> is empty/whitespace.</exception>
    public SqliteConversationStore(string connectionString)
    {
        if (connectionString is null)
        {
            throw new ArgumentNullException(nameof(connectionString));
        }

        if (connectionString.Trim().Length == 0)
        {
            throw new ArgumentException("Connection string must be non-empty.", nameof(connectionString));
        }

        _connectionString = connectionString;
    }

    /// <inheritdoc/>
    public async Task AppendAsync(string threadId, IReadOnlyList<ChatMessage> messages, CancellationToken cancellationToken = default)
    {
        ValidateThreadId(threadId);
        if (messages is null)
        {
            throw new ArgumentNullException(nameof(messages));
        }

        using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(cancellationToken).ConfigureAwait(false);
        await EnsureSchemaAsync(connection, cancellationToken).ConfigureAwait(false);

        using var transaction = connection.BeginTransaction();
        foreach (var message in messages)
        {
            if (message is null)
            {
                throw new ArgumentException("Messages must not contain null elements.", nameof(messages));
            }

            using var command = connection.CreateCommand();
            command.Transaction = transaction;
            command.CommandText =
                "INSERT INTO Conversations (ThreadId, Role, Text) VALUES ($threadId, $role, $text);";
            command.Parameters.AddWithValue("$threadId", threadId);
            command.Parameters.AddWithValue("$role", message.Role.ToString());
            command.Parameters.AddWithValue("$text", message.Text);
            await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }

        await transaction.CommitAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<ChatMessage>> GetAsync(string threadId, CancellationToken cancellationToken = default)
    {
        ValidateThreadId(threadId);

        using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(cancellationToken).ConfigureAwait(false);
        await EnsureSchemaAsync(connection, cancellationToken).ConfigureAwait(false);

        using var command = connection.CreateCommand();
        command.CommandText =
            "SELECT Role, Text FROM Conversations WHERE ThreadId = $threadId ORDER BY Seq ASC;";
        command.Parameters.AddWithValue("$threadId", threadId);

        var result = new List<ChatMessage>();
        using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            var role = ParseRole(reader.GetString(0));
            var text = reader.GetString(1);
            result.Add(new ChatMessage(role, text));
        }

        return result;
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<string>> ListThreadsAsync(CancellationToken cancellationToken = default)
    {
        using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(cancellationToken).ConfigureAwait(false);
        await EnsureSchemaAsync(connection, cancellationToken).ConfigureAwait(false);

        using var command = connection.CreateCommand();
        command.CommandText = "SELECT DISTINCT ThreadId FROM Conversations;";

        var result = new List<string>();
        using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            result.Add(reader.GetString(0));
        }

        return result;
    }

    /// <inheritdoc/>
    public async Task ClearAsync(string threadId, CancellationToken cancellationToken = default)
    {
        ValidateThreadId(threadId);

        using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(cancellationToken).ConfigureAwait(false);
        await EnsureSchemaAsync(connection, cancellationToken).ConfigureAwait(false);

        using var command = connection.CreateCommand();
        command.CommandText = "DELETE FROM Conversations WHERE ThreadId = $threadId;";
        command.Parameters.AddWithValue("$threadId", threadId);
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    private async Task EnsureSchemaAsync(SqliteConnection connection, CancellationToken cancellationToken)
    {
        // Fast path: already initialised for this connection string.
        if (_schemaInitialised.ContainsKey(_connectionString))
        {
            return;
        }

        using var command = connection.CreateCommand();
        command.CommandText =
            "CREATE TABLE IF NOT EXISTS Conversations (" +
            "Seq INTEGER PRIMARY KEY AUTOINCREMENT, " +
            "ThreadId TEXT NOT NULL, " +
            "Role TEXT NOT NULL, " +
            "Text TEXT NOT NULL);" +
            "CREATE INDEX IF NOT EXISTS IX_Conversations_ThreadId ON Conversations (ThreadId, Seq);";
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        _schemaInitialised[_connectionString] = true;
    }

    private static ChatRole ParseRole(string role)
    {
        // Defensive parse — a database row with a role string that no longer
        // matches the current enum (schema evolution, manual edit) gives a
        // descriptive InvalidOperationException naming the offending value
        // instead of the raw ArgumentException from Enum.Parse.
        if (Enum.TryParse<ChatRole>(role, out var result))
        {
            return result;
        }
        throw new InvalidOperationException(
            $"Invalid role value in database: '{role}'.");
    }

    private static void ValidateThreadId(string threadId)
    {
        if (threadId is null)
        {
            throw new ArgumentNullException(nameof(threadId));
        }

        if (threadId.Trim().Length == 0)
        {
            throw new ArgumentException("Thread id must be non-empty.", nameof(threadId));
        }
    }
}
