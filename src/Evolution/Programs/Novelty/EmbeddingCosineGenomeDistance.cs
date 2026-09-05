using AiDotNet.Agentic.Embeddings;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Novelty;

/// <summary>A cosine distance over program embeddings, served from a primed cache so no comparison blocks on I/O.</summary>
/// <remarks>
/// <para>
/// <see cref="IGenomeDistance{TGenome}.Distance"/> is synchronous, and a synchronous method must never block on a
/// network call — doing so is how a search deadlocks under a synchronization context. So this metric separates the
/// two halves: <see cref="PrimeAsync"/> fetches vectors in one batched, cancellable request, and
/// <see cref="Distance"/> then reads them out of the cache and returns <c>1 - cosine</c>, clamped to 0 to 1. A pair
/// whose vectors are not both cached falls back to <see cref="Fallback"/>, which never needs a provider, so a
/// missing vector degrades precision rather than correctness.
/// </para>
/// <para>
/// Keying is by <see cref="ProgramGenome.Id"/>, the SHA-256 of the normalized source, so two candidates that differ
/// only in line endings share one vector and one request. Priming a genome already primed costs nothing. Wrapping
/// the supplied client in a <see cref="CachingEmbeddingClient"/> extends that reuse across metric instances and
/// across runs in the same process.
/// </para>
/// <para>
/// The reference implementation has no equivalent: it embeds inside the insertion path, one blocking request per
/// candidate, then compares the new vector against every member of the island in a Python loop. Here the requests
/// are batched and explicit, the comparison is arithmetic over cached vectors, and
/// <see cref="CosineComparisons"/> against <see cref="FallbackComparisons"/> reports exactly how often the paid
/// path was actually used.
/// </para>
/// <para><b>For Beginners:</b> Embeddings tell you whether two programs mean similar things, but fetching one takes
/// a network call. This class keeps the two steps apart: first you ask it to fetch vectors for a set of programs,
/// then comparing any two of them is instant. If you compare a program it never fetched, it quietly falls back to a
/// text-only comparison instead of stalling.</para>
/// </remarks>
public sealed class EmbeddingCosineGenomeDistance : IGenomeDistance<ProgramGenome>
{
    /// <summary>The identifier this metric reports.</summary>
    public const string MetricId = "program-embedding-cosine";

    private readonly IEmbeddingClient _client;
    private readonly IGenomeDistance<ProgramGenome> _fallback;
    private readonly Dictionary<string, EmbeddingVector> _vectors = new(StringComparer.Ordinal);
    private readonly object _gate = new();

    private long _cosineComparisons;
    private long _fallbackComparisons;
    private long _primeRequests;

    /// <summary>Initializes a cosine distance over embeddings.</summary>
    /// <param name="client">The embedding client used by <see cref="PrimeAsync"/>.</param>
    /// <param name="fallback">
    /// The metric used when a pair is not primed; <c>null</c> uses a <see cref="ProgramTokenSetDistance"/>.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="client"/> is <c>null</c>.</exception>
    public EmbeddingCosineGenomeDistance(
        IEmbeddingClient client,
        IGenomeDistance<ProgramGenome>? fallback = null)
    {
        Guard.NotNull(client);
        _client = client;
        _fallback = fallback ?? new ProgramTokenSetDistance();
    }

    /// <summary>Gets the metric used when a pair has no cached vectors.</summary>
    public IGenomeDistance<ProgramGenome> Fallback => _fallback;

    /// <summary>Gets the embedding client this metric primes from.</summary>
    public IEmbeddingClient Client => _client;

    /// <summary>Gets how many comparisons were answered from cached vectors.</summary>
    public long CosineComparisons => Interlocked.Read(ref _cosineComparisons);

    /// <summary>Gets how many comparisons fell back because a vector was missing.</summary>
    public long FallbackComparisons => Interlocked.Read(ref _fallbackComparisons);

    /// <summary>Gets how many batched priming requests were issued.</summary>
    public long PrimeRequests => Interlocked.Read(ref _primeRequests);

    /// <summary>Gets how many genome vectors are currently cached.</summary>
    public int PrimedCount
    {
        get
        {
            lock (_gate) return _vectors.Count;
        }
    }

    /// <inheritdoc/>
    public string Id => MetricId;

    /// <inheritdoc/>
    public string VersionHash => MetricId + "-v1-" + _client.ModelId + "-fallback-" + _fallback.VersionHash;

    /// <summary>Fetches and caches vectors for every genome not already primed, in one batched request.</summary>
    /// <param name="genomes">The genomes to prime.</param>
    /// <param name="cancellationToken">A token that cancels the request.</param>
    /// <returns>
    /// <c>true</c> when every requested genome now has a vector, <c>false</c> when the provider could not be
    /// reached; in the latter case nothing was cached and <see cref="Distance"/> keeps using the fallback.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="genomes"/> is <c>null</c>, or an entry is <c>null</c>.</exception>
    public async ValueTask<bool> PrimeAsync(
        IEnumerable<ProgramGenome> genomes,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(genomes);

        // The caller's sequence is materialized before the lock is taken: enumerating arbitrary caller code while
        // holding a lock is how an unexpected re-entrant call becomes a deadlock.
        var requested = new List<ProgramGenome>();
        foreach (ProgramGenome genome in genomes)
        {
            Guard.NotNull(genome, nameof(genomes));
            requested.Add(genome);
        }

        var missingIds = new List<string>();
        var missingSources = new List<string>();
        lock (_gate)
        {
            foreach (ProgramGenome genome in requested)
            {
                if (_vectors.ContainsKey(genome.Id)) continue;
                if (missingIds.Contains(genome.Id, StringComparer.Ordinal)) continue;
                missingIds.Add(genome.Id);
                missingSources.Add(genome.NormalizedSource);
            }
        }

        if (missingSources.Count == 0) return true;

        Interlocked.Increment(ref _primeRequests);
        EmbeddingBatch batch = await _client.EmbedAsync(missingSources, cancellationToken).ConfigureAwait(false);
        if (batch is null || !batch.Succeeded || batch.Vectors.Count != missingIds.Count) return false;

        lock (_gate)
        {
            for (int index = 0; index < missingIds.Count; index++)
            {
                _vectors[missingIds[index]] = batch.Vectors[index];
            }
        }

        return true;
    }

    /// <summary>Reports whether a genome's vector is cached.</summary>
    /// <param name="genome">The genome to check.</param>
    /// <returns><c>true</c> when <see cref="Distance"/> would use cosine rather than the fallback for it.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="genome"/> is <c>null</c>.</exception>
    public bool IsPrimed(ProgramGenome genome)
    {
        Guard.NotNull(genome);
        lock (_gate) return _vectors.ContainsKey(genome.Id);
    }

    /// <summary>Discards every cached vector.</summary>
    public void Clear()
    {
        lock (_gate) _vectors.Clear();
    }

    /// <summary>Computes the cosine similarity of two primed genomes.</summary>
    /// <param name="first">The first genome.</param>
    /// <param name="second">The second genome.</param>
    /// <returns>The cosine similarity, or <c>null</c> when either genome is not primed.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="first"/> or <paramref name="second"/> is <c>null</c>.</exception>
    public double? Similarity(ProgramGenome first, ProgramGenome second)
    {
        Guard.NotNull(first);
        Guard.NotNull(second);

        EmbeddingVector? firstVector;
        EmbeddingVector? secondVector;
        lock (_gate)
        {
            _vectors.TryGetValue(first.Id, out firstVector);
            _vectors.TryGetValue(second.Id, out secondVector);
        }

        if (firstVector is null || secondVector is null) return null;
        return EmbeddingVector.CosineSimilarity(firstVector, secondVector);
    }

    /// <inheritdoc/>
    public double Distance(ProgramGenome first, ProgramGenome second)
    {
        Guard.NotNull(first);
        Guard.NotNull(second);

        double? similarity = Similarity(first, second);
        if (similarity is not { } cosine)
        {
            Interlocked.Increment(ref _fallbackComparisons);
            return _fallback.Distance(first, second);
        }

        Interlocked.Increment(ref _cosineComparisons);
        double distance = 1.0 - cosine;
        return distance < 0.0 ? 0.0 : distance > 1.0 ? 1.0 : distance;
    }
}
