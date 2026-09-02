using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Agentic.Embeddings;

/// <summary>A hashing embedding client that contacts nothing and returns the same vector for the same text forever.</summary>
/// <remarks>
/// <para>
/// Every token of the input is hashed with FNV-1a into a bucket and a sign, and the bucket counts become the
/// vector — the classic hashing-trick bag of tokens. That makes it a genuine similarity model rather than a stub:
/// two texts sharing most of their tokens produce vectors with a high cosine similarity, and two unrelated texts do
/// not, so an embedding-gated code path can be exercised end to end with no provider, no key, and no network. It is
/// not a semantic model: it cannot tell that two differently-worded programs do the same thing.
/// </para>
/// <para>
/// The hash is computed here rather than taken from <see cref="object.GetHashCode"/>, whose string implementation is
/// randomized per process on modern frameworks, so results are stable across processes, machines, and target
/// frameworks — which is what makes a test written against it reproducible.
/// </para>
/// <para><b>For Beginners:</b> Use this wherever you want the embedding path to run without paying a provider: in
/// tests, in a demo, or while you are still deciding which model to buy. It turns text into numbers using simple
/// arithmetic on the words themselves, so similar text still gets similar numbers, and the same text always gets
/// exactly the same numbers.</para>
/// </remarks>
public sealed class DeterministicEmbeddingClient : IEmbeddingClient
{
    /// <summary>The default number of components in a produced vector.</summary>
    public const int DefaultDimensions = 64;

    /// <summary>The model identifier reported when none is supplied.</summary>
    public const string DefaultModelId = "deterministic-hashing-embedding";

    private const uint FnvOffsetBasis = 2166136261;
    private const uint FnvPrime = 16777619;

    private long _calls;
    private long _textsEmbedded;

    /// <summary>Initializes a deterministic client.</summary>
    /// <param name="dimensions">The vector length; 2 to 4096.</param>
    /// <param name="modelId">The identifier reported as <see cref="ModelId"/>.</param>
    /// <exception cref="ArgumentNullException"><paramref name="modelId"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="modelId"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimensions"/> is outside its permitted range.</exception>
    public DeterministicEmbeddingClient(int dimensions = DefaultDimensions, string modelId = DefaultModelId)
    {
        Guard.NotNullOrWhiteSpace(modelId);
        if (dimensions < 2 || dimensions > 4_096)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, "Value must be between 2 and 4096.");
        }

        Dimensions = dimensions;
        ModelId = modelId;
    }

    /// <inheritdoc/>
    public string ModelId { get; }

    /// <summary>Gets the length of every vector this client produces.</summary>
    public int Dimensions { get; }

    /// <summary>Gets how many batches this client has answered.</summary>
    public long Calls => Interlocked.Read(ref _calls);

    /// <summary>Gets how many individual texts this client has embedded.</summary>
    public long TextsEmbedded => Interlocked.Read(ref _textsEmbedded);

    /// <inheritdoc/>
    public ValueTask<EmbeddingBatch> EmbedAsync(
        IReadOnlyList<string> texts,
        CancellationToken cancellationToken = default)
    {
        EmbeddingRequestValidation.Validate(texts);
        cancellationToken.ThrowIfCancellationRequested();

        Interlocked.Increment(ref _calls);
        Interlocked.Add(ref _textsEmbedded, texts.Count);

        var vectors = new List<EmbeddingVector>(texts.Count);
        foreach (string text in texts) vectors.Add(Embed(text));
        return new ValueTask<EmbeddingBatch>(EmbeddingBatch.Success(vectors));
    }

    /// <summary>Computes the vector this client would return for one text.</summary>
    /// <param name="text">The text to embed.</param>
    /// <returns>The deterministic vector.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="text"/> is <c>null</c>.</exception>
    public EmbeddingVector Embed(string text)
    {
        Guard.NotNull(text);
        var components = new double[Dimensions];
        int tokenCount = 0;

        int start = 0;
        while (start < text.Length)
        {
            if (char.IsWhiteSpace(text[start]))
            {
                start++;
                continue;
            }

            int end = start;
            while (end < text.Length && !char.IsWhiteSpace(text[end])) end++;
            Accumulate(components, text, start, end - start);
            tokenCount++;
            start = end;
        }

        // A blank input, or a token multiset whose signed contributions happen to cancel exactly, would otherwise
        // produce a zero-magnitude vector, and such a vector compares as similarity zero against everything —
        // including a copy of itself. Both cases fall back to a single deterministic unit component.
        if (tokenCount == 0 || IsZero(components))
        {
            components[(int)(Hash(text, 0, text.Length) % (uint)Dimensions)] = 1.0;
        }

        return new EmbeddingVector(components);
    }

    private static bool IsZero(double[] components)
    {
        foreach (double component in components)
        {
            if (component != 0.0) return false;
        }

        return true;
    }

    private void Accumulate(double[] components, string text, int start, int length)
    {
        uint hash = Hash(text, start, length);
        int bucket = (int)(hash % (uint)Dimensions);
        double sign = (hash & 0x80000000u) == 0 ? 1.0 : -1.0;
        components[bucket] += sign;
    }

    private static uint Hash(string text, int start, int length)
    {
        uint hash = FnvOffsetBasis;
        for (int index = start; index < start + length; index++)
        {
            char character = text[index];
            unchecked
            {
                hash = (hash ^ (byte)(character & 0xFF)) * FnvPrime;
                hash = (hash ^ (byte)(character >> 8)) * FnvPrime;
            }
        }

        return hash;
    }
}
