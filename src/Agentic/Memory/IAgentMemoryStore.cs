namespace AiDotNet.Agentic.Memory;

/// <summary>
/// A long-term, cross-thread memory store: it remembers facts and can retrieve the ones most relevant to a
/// query. This is the durable counterpart to the per-thread <see cref="IConversationStore"/> and the
/// retrieval source behind <see cref="MemoryAugmentedAgent{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// Implementations differ only in <em>how</em> they rank relevance: <see cref="InMemoryAgentMemoryStore"/>
/// uses lexical overlap (zero-config, no model), while <c>EmbeddingAgentMemoryStore&lt;T&gt;</c> uses an
/// <see cref="AiDotNet.Interfaces.IEmbeddingModel{T}"/> for true semantic similarity by reusing AiDotNet's
/// RAG embedding + cosine-metric stack. Callers depend only on this interface, so semantic search is a
/// drop-in upgrade over the lexical default.
/// </para>
/// <para><b>For Beginners:</b> This is the assistant's notebook of long-term facts. You can add notes,
/// search for the notes most related to a question, list them, or remove one. Whether the search matches by
/// shared words or by meaning depends on which implementation you plug in — the way you use it is the same.
/// </para>
/// </remarks>
public interface IAgentMemoryStore
{
    /// <summary>
    /// Stores a new memory and returns its generated id.
    /// </summary>
    /// <param name="content">The text to remember. Must be non-empty.</param>
    /// <param name="metadata">Optional key/value metadata. <c>null</c> means none.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    /// <returns>The new memory's id.</returns>
    Task<string> AddAsync(
        string content,
        IReadOnlyDictionary<string, string>? metadata = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Returns the memories most relevant to a query, highest score first.
    /// </summary>
    /// <param name="query">The query text. Must be non-empty.</param>
    /// <param name="topK">The maximum number of memories to return. Must be positive.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    /// <returns>The top matches ordered by descending relevance (empty when nothing matches).</returns>
    Task<IReadOnlyList<ScoredMemory>> SearchAsync(
        string query,
        int topK = 5,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Returns all stored memories (order unspecified).
    /// </summary>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    Task<IReadOnlyList<AgentMemory>> GetAllAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Removes a memory by id. A no-op when the id is unknown.
    /// </summary>
    /// <param name="id">The id of the memory to remove. Must be non-empty.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    Task RemoveAsync(string id, CancellationToken cancellationToken = default);
}
