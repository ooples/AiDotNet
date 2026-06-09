using System.Text;

namespace AiDotNet.Agentic.Memory;

/// <summary>
/// A process-local <see cref="IAgentMemoryStore"/> that ranks memories by lexical overlap with the query —
/// the fraction of the query's words that appear in the memory. Requires no embedding model, so it is the
/// zero-config default; for meaning-based recall, use <c>EmbeddingAgentMemoryStore&lt;T&gt;</c>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This notebook finds notes by matching words. If you ask about "deadline" it
/// finds notes containing "deadline", but it won't realize "due date" means the same thing — that needs the
/// embedding-backed store. It's fast, needs no setup, and is great for tests and simple cases.
/// </para>
/// </remarks>
public sealed class InMemoryAgentMemoryStore : IAgentMemoryStore
{
    private readonly object _gate = new();
    private readonly List<AgentMemory> _memories = new();

    /// <inheritdoc/>
    public Task<string> AddAsync(
        string content,
        IReadOnlyDictionary<string, string>? metadata = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(content);
        var id = Guid.NewGuid().ToString("N");
        var memory = new AgentMemory(id, content, metadata);

        lock (_gate)
        {
            _memories.Add(memory);
        }

        return Task.FromResult(id);
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<ScoredMemory>> SearchAsync(
        string query,
        int topK = 5,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(query);
        Guard.Positive(topK);

        var queryTerms = Tokenize(query);
        if (queryTerms.Count == 0)
        {
            return Task.FromResult<IReadOnlyList<ScoredMemory>>(new List<ScoredMemory>());
        }

        List<AgentMemory> snapshot;
        lock (_gate)
        {
            snapshot = new List<AgentMemory>(_memories);
        }

        var scored = new List<ScoredMemory>();
        foreach (var memory in snapshot)
        {
            var terms = Tokenize(memory.Content);
            if (terms.Count == 0)
            {
                continue;
            }

            var matches = 0;
            foreach (var term in queryTerms)
            {
                if (terms.Contains(term))
                {
                    matches++;
                }
            }

            if (matches > 0)
            {
                scored.Add(new ScoredMemory(memory, (double)matches / queryTerms.Count));
            }
        }

        IReadOnlyList<ScoredMemory> result = scored
            .OrderByDescending(s => s.Score)
            .Take(topK)
            .ToList();
        return Task.FromResult(result);
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<AgentMemory>> GetAllAsync(CancellationToken cancellationToken = default)
    {
        lock (_gate)
        {
            IReadOnlyList<AgentMemory> all = new List<AgentMemory>(_memories);
            return Task.FromResult(all);
        }
    }

    /// <inheritdoc/>
    public Task RemoveAsync(string id, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(id);

        lock (_gate)
        {
            _memories.RemoveAll(m => string.Equals(m.Id, id, StringComparison.Ordinal));
        }

        return Task.CompletedTask;
    }

    private static HashSet<string> Tokenize(string text)
    {
        var terms = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var current = new StringBuilder();
        foreach (var ch in text)
        {
            if (char.IsLetterOrDigit(ch))
            {
                current.Append(char.ToLowerInvariant(ch));
            }
            else if (current.Length > 0)
            {
                terms.Add(current.ToString());
                current.Clear();
            }
        }

        if (current.Length > 0)
        {
            terms.Add(current.ToString());
        }

        return terms;
    }
}
