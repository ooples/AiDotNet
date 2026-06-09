using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;

namespace AiDotNet.Agentic.Memory;

/// <summary>
/// A semantic <see cref="IAgentMemoryStore"/> that ranks memories by meaning, not words. It embeds each
/// memory with an <see cref="IEmbeddingModel{T}"/> and scores a query by cosine similarity against those
/// embeddings — reusing AiDotNet's RAG embedding and similarity-metric stack, so a memory about "due date"
/// is recalled for a query about "deadline".
/// </summary>
/// <typeparam name="T">The numeric type of the embedding model and similarity metric.</typeparam>
/// <remarks>
/// <para>
/// Memories and their embedding vectors are held in memory; the vector for each memory is computed once on
/// <see cref="AddAsync"/>. This keeps the abstraction identical to the lexical
/// <see cref="InMemoryAgentMemoryStore"/> while delivering true semantic recall — pick whichever fits the
/// deployment without touching agent code.
/// </para>
/// <para><b>For Beginners:</b> Same notebook idea, but instead of matching words it matches <em>meaning</em>.
/// It turns every note (and your question) into a list of numbers that capture meaning, then finds the notes
/// whose numbers are closest to your question's — so synonyms and paraphrases still match.
/// </para>
/// </remarks>
public sealed class EmbeddingAgentMemoryStore<T> : IAgentMemoryStore
{
    private readonly object _gate = new();
    private readonly List<Entry> _entries = new();
    private readonly IEmbeddingModel<T> _embeddingModel;
    private readonly ISimilarityMetric<T> _similarity;

    /// <summary>
    /// Initializes a new semantic memory store.
    /// </summary>
    /// <param name="embeddingModel">The model used to embed memories and queries.</param>
    /// <param name="similarity">The similarity metric. <c>null</c> uses cosine similarity.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="embeddingModel"/> is <c>null</c>.</exception>
    public EmbeddingAgentMemoryStore(IEmbeddingModel<T> embeddingModel, ISimilarityMetric<T>? similarity = null)
    {
        Guard.NotNull(embeddingModel);
        _embeddingModel = embeddingModel;
        _similarity = similarity ?? new CosineSimilarityMetric<T>();
    }

    /// <inheritdoc/>
    public async Task<string> AddAsync(
        string content,
        IReadOnlyDictionary<string, string>? metadata = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(content);
        var id = Guid.NewGuid().ToString("N");
        var memory = new AgentMemory(id, content, metadata);
        var vector = await _embeddingModel.EmbedAsync(content).ConfigureAwait(false);

        lock (_gate)
        {
            _entries.Add(new Entry(memory, vector));
        }

        return id;
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<ScoredMemory>> SearchAsync(
        string query,
        int topK = 5,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(query);
        Guard.Positive(topK);

        List<Entry> snapshot;
        lock (_gate)
        {
            snapshot = new List<Entry>(_entries);
        }

        if (snapshot.Count == 0)
        {
            return new List<ScoredMemory>();
        }

        var queryVector = await _embeddingModel.EmbedAsync(query).ConfigureAwait(false);

        var scored = new List<ScoredMemory>(snapshot.Count);
        foreach (var entry in snapshot)
        {
            var score = Convert.ToDouble(_similarity.Calculate(queryVector, entry.Vector));
            scored.Add(new ScoredMemory(entry.Memory, score));
        }

        var ordered = _similarity.HigherIsBetter
            ? scored.OrderByDescending(s => s.Score)
            : scored.OrderBy(s => s.Score);

        return ordered.Take(topK).ToList();
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<AgentMemory>> GetAllAsync(CancellationToken cancellationToken = default)
    {
        lock (_gate)
        {
            IReadOnlyList<AgentMemory> all = _entries.Select(e => e.Memory).ToList();
            return Task.FromResult(all);
        }
    }

    /// <inheritdoc/>
    public Task RemoveAsync(string id, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(id);

        lock (_gate)
        {
            _entries.RemoveAll(e => string.Equals(e.Memory.Id, id, StringComparison.Ordinal));
        }

        return Task.CompletedTask;
    }

    private readonly struct Entry
    {
        public Entry(AgentMemory memory, Vector<T> vector)
        {
            Memory = memory;
            Vector = vector;
        }

        public AgentMemory Memory { get; }

        public Vector<T> Vector { get; }
    }
}
