using AiDotNet.Evolution;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Agentic.Embeddings;

/// <summary>Caches embedding vectors by content hash so a text already embedded costs nothing to embed again.</summary>
/// <remarks>
/// <para>
/// The key is the SHA-256 of the text, so two candidates that are byte-identical share one vector no matter which
/// batch they arrived in, and a candidate re-proposed later in a run never reaches the provider a second time. Only
/// the misses are forwarded: a batch of ten texts of which nine are cached becomes one request for one text, and a
/// batch with no misses becomes no request at all. That is the whole point — an evolutionary search re-proposes
/// near-identical programs constantly, and the reference implementation pays a provider call for every one of them
/// because it embeds inside the insertion path with no cache anywhere.
/// </para>
/// <para>
/// A failed batch is never cached. Upstream stores its failure sentinel on the program record permanently, so one
/// transient error keeps that record's similarity at zero for the rest of the run; here the next call simply tries
/// again. Eviction is first-in-first-out at <see cref="Capacity"/> entries, which bounds memory without the
/// bookkeeping of a recency order, and the cache is safe for concurrent use.
/// </para>
/// <para><b>For Beginners:</b> Asking a provider to embed text costs money and time, and evolutionary search asks
/// about the same text over and over. This wrapper remembers answers it has already received, keyed by a
/// fingerprint of the text, and only asks about text it has not seen. Wrap any client in it and the rest of your
/// code does not change; <see cref="Hits"/> and <see cref="InnerCalls"/> tell you how much it saved.</para>
/// </remarks>
public sealed class CachingEmbeddingClient : IEmbeddingClient
{
    /// <summary>The default number of vectors retained.</summary>
    public const int DefaultCapacity = 4_096;

    private readonly IEmbeddingClient _inner;
    private readonly Dictionary<string, EmbeddingVector> _cache = new(StringComparer.Ordinal);
    private readonly Queue<string> _insertionOrder = new();
    private readonly object _gate = new();

    private long _hits;
    private long _misses;
    private long _innerCalls;

    /// <summary>Initializes a caching decorator around another client.</summary>
    /// <param name="inner">The client that answers cache misses.</param>
    /// <param name="capacity">The number of vectors retained before the oldest is evicted; 1 to 1,000,000.</param>
    /// <exception cref="ArgumentNullException"><paramref name="inner"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="capacity"/> is outside its permitted range.</exception>
    public CachingEmbeddingClient(IEmbeddingClient inner, int capacity = DefaultCapacity)
    {
        Guard.NotNull(inner);
        if (capacity <= 0 || capacity > 1_000_000)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "Value must be between 1 and 1000000.");
        }

        _inner = inner;
        Capacity = capacity;
    }

    /// <inheritdoc/>
    public string ModelId => _inner.ModelId;

    /// <summary>Gets the client that answers cache misses.</summary>
    public IEmbeddingClient Inner => _inner;

    /// <summary>Gets the number of vectors retained before the oldest is evicted.</summary>
    public int Capacity { get; }

    /// <summary>Gets how many requested texts were served from the cache.</summary>
    public long Hits => Interlocked.Read(ref _hits);

    /// <summary>Gets how many requested texts had to be embedded.</summary>
    public long Misses => Interlocked.Read(ref _misses);

    /// <summary>Gets how many batches were forwarded to the inner client.</summary>
    public long InnerCalls => Interlocked.Read(ref _innerCalls);

    /// <summary>Gets how many vectors the cache currently holds.</summary>
    public int Count
    {
        get
        {
            lock (_gate) return _cache.Count;
        }
    }

    /// <summary>Computes the cache key for a text.</summary>
    /// <param name="text">The text that would be embedded.</param>
    /// <returns>The lowercase hexadecimal SHA-256 of the text.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="text"/> is <c>null</c>.</exception>
    public static string ComputeKey(string text)
    {
        Guard.NotNull(text);
        return EvolutionHash.Compute(text);
    }

    /// <summary>Removes every cached vector, leaving the counters untouched.</summary>
    public void Clear()
    {
        lock (_gate)
        {
            _cache.Clear();
            _insertionOrder.Clear();
        }
    }

    /// <inheritdoc/>
    public async ValueTask<EmbeddingBatch> EmbedAsync(
        IReadOnlyList<string> texts,
        CancellationToken cancellationToken = default)
    {
        EmbeddingRequestValidation.Validate(texts);

        var keys = new string[texts.Count];
        var resolved = new EmbeddingVector?[texts.Count];
        var missingKeys = new List<string>();
        var missingTexts = new List<string>();

        for (int index = 0; index < texts.Count; index++)
        {
            keys[index] = ComputeKey(texts[index]);
        }

        lock (_gate)
        {
            for (int index = 0; index < texts.Count; index++)
            {
                if (_cache.TryGetValue(keys[index], out EmbeddingVector? cached))
                {
                    resolved[index] = cached;
                    continue;
                }

                // A batch that repeats the same text is forwarded once, not twice.
                if (!missingKeys.Contains(keys[index], StringComparer.Ordinal))
                {
                    missingKeys.Add(keys[index]);
                    missingTexts.Add(texts[index]);
                }
            }
        }

        int hitCount = 0;
        foreach (EmbeddingVector? vector in resolved)
        {
            if (vector is not null) hitCount++;
        }

        Interlocked.Add(ref _hits, hitCount);
        Interlocked.Add(ref _misses, texts.Count - hitCount);

        if (missingTexts.Count == 0)
        {
            return EmbeddingBatch.Success(Materialize(resolved));
        }

        Interlocked.Increment(ref _innerCalls);
        EmbeddingBatch batch = await _inner
            .EmbedAsync(missingTexts, cancellationToken)
            .ConfigureAwait(false);

        if (batch is null) return EmbeddingBatch.Failure("the inner embedding client returned no batch");
        if (!batch.Succeeded) return batch;
        if (batch.Vectors.Count != missingTexts.Count)
        {
            return EmbeddingBatch.Failure("the inner embedding client returned the wrong number of vectors");
        }

        lock (_gate)
        {
            for (int index = 0; index < missingKeys.Count; index++)
            {
                Store(missingKeys[index], batch.Vectors[index]);
            }

            for (int index = 0; index < texts.Count; index++)
            {
                if (resolved[index] is null && _cache.TryGetValue(keys[index], out EmbeddingVector? filled))
                {
                    resolved[index] = filled;
                }
            }
        }

        return EmbeddingBatch.Success(Materialize(resolved));
    }

    private static IReadOnlyList<EmbeddingVector> Materialize(EmbeddingVector?[] resolved)
    {
        var vectors = new List<EmbeddingVector>(resolved.Length);
        foreach (EmbeddingVector? vector in resolved)
        {
            if (vector is null)
            {
                throw new InvalidOperationException(
                    "The embedding cache resolved fewer vectors than inputs, which indicates a defect in the cache.");
            }

            vectors.Add(vector);
        }

        return vectors;
    }

    private void Store(string key, EmbeddingVector vector)
    {
        if (_cache.ContainsKey(key)) return;
        while (_cache.Count >= Capacity && _insertionOrder.Count > 0)
        {
            string oldest = _insertionOrder.Dequeue();
            _cache.Remove(oldest);
        }

        if (_cache.Count >= Capacity) return;
        _cache[key] = vector;
        _insertionOrder.Enqueue(key);
    }
}
