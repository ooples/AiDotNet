using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.Embeddings;

/// <summary>
/// An <see cref="IEmbeddingModel{T}"/> decorator that caches embeddings by content hash so that
/// repeated texts are not re-embedded, mirroring LangChain's <c>CacheBackedEmbeddings</c> and
/// LlamaIndex's ingestion cache.
/// </summary>
/// <remarks>
/// <para>
/// Each text is hashed with SHA-256 (see <see cref="ContentHash"/>) and the resulting key is namespaced
/// by the wrapped model's identity and embedding dimension, so different models never collide in a shared
/// cache. On a cache hit the stored vector is returned; on a miss the inner model is invoked and the result
/// is stored. Batch operations de-duplicate texts both within a single call and across calls, invoke the
/// inner model only for the distinct misses, and preserve the input ordering of the returned matrix rows.
/// </para>
/// <para><b>For Beginners:</b> This wraps a "real" embedding model and remembers its answers.
///
/// - The first time you embed a piece of text, it calls the wrapped model and saves the result.
/// - The next time you embed the same text, it returns the saved result instead of calling the model again.
/// - This saves time and money when the same text shows up more than once (very common in RAG pipelines).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector calculations (typically float or double).</typeparam>
public sealed class CacheBackedEmbeddings<T> : IEmbeddingModel<T>
{
    private readonly IEmbeddingModel<T> _inner;
    private readonly IEmbeddingCache<T> _cache;
    private readonly string _namespace;

    /// <summary>
    /// Initializes a new instance of the <see cref="CacheBackedEmbeddings{T}"/> class.
    /// </summary>
    /// <param name="inner">The underlying embedding model to wrap.</param>
    /// <param name="cache">
    /// The cache used to store embeddings. If <c>null</c>, a new unbounded <see cref="InMemoryEmbeddingCache{T}"/> is created.
    /// </param>
    /// <param name="modelNamespace">
    /// An optional key namespace that identifies the inner model. If <c>null</c>, a namespace is derived from the
    /// inner model's runtime type and embedding dimension so that different models do not share cache entries.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="inner"/> is <c>null</c>.</exception>
    public CacheBackedEmbeddings(
        IEmbeddingModel<T> inner,
        IEmbeddingCache<T>? cache = null,
        string? modelNamespace = null)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        _cache = cache ?? new InMemoryEmbeddingCache<T>();
        _namespace = string.IsNullOrEmpty(modelNamespace)
            ? $"{_inner.GetType().FullName}:{_inner.EmbeddingDimension}"
            : modelNamespace!;
    }

    /// <summary>
    /// Gets the underlying embedding model that this cache wraps.
    /// </summary>
    public IEmbeddingModel<T> InnerModel => _inner;

    /// <summary>
    /// Gets the cache used to store embeddings.
    /// </summary>
    public IEmbeddingCache<T> Cache => _cache;

    /// <summary>
    /// Gets the key namespace that identifies the wrapped model within the cache.
    /// </summary>
    public string Namespace => _namespace;

    /// <inheritdoc />
    public int EmbeddingDimension => _inner.EmbeddingDimension;

    /// <inheritdoc />
    public int MaxTokens => _inner.MaxTokens;

    /// <summary>
    /// Builds the namespaced content-hash cache key for a piece of text.
    /// </summary>
    private string BuildKey(string text) => _namespace + ":" + ContentHash.ComputeHash(text);

    /// <inheritdoc />
    public Vector<T> Embed(string text)
    {
        if (text == null)
            throw new ArgumentNullException(nameof(text));

        var key = BuildKey(text);
        if (_cache.TryGet(key, out var cached) && cached != null)
        {
            return cached;
        }

        var embedding = _inner.Embed(text);
        _cache.Set(key, Clone(embedding));
        return embedding;
    }

    /// <inheritdoc />
    public async Task<Vector<T>> EmbedAsync(string text)
    {
        if (text == null)
            throw new ArgumentNullException(nameof(text));

        var key = BuildKey(text);
        if (_cache.TryGet(key, out var cached) && cached != null)
        {
            return cached;
        }

        var embedding = await _inner.EmbedAsync(text).ConfigureAwait(false);
        _cache.Set(key, Clone(embedding));
        return embedding;
    }

    /// <inheritdoc />
    public Matrix<T> EmbedBatch(IEnumerable<string> texts)
    {
        var textList = ValidateBatch(texts);

        var results = new Vector<T>[textList.Count];
        var missTexts = ResolveHitsAndCollectMisses(textList, results, out var missIndexLists, out var missKeys);

        if (missTexts.Count > 0)
        {
            var missMatrix = _inner.EmbedBatch(missTexts);
            StoreAndDistributeMisses(missMatrix, missKeys, missIndexLists, results);
        }

        return BuildMatrix(results);
    }

    /// <inheritdoc />
    public async Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts)
    {
        var textList = ValidateBatch(texts);

        var results = new Vector<T>[textList.Count];
        var missTexts = ResolveHitsAndCollectMisses(textList, results, out var missIndexLists, out var missKeys);

        if (missTexts.Count > 0)
        {
            var missMatrix = await _inner.EmbedBatchAsync(missTexts).ConfigureAwait(false);
            StoreAndDistributeMisses(missMatrix, missKeys, missIndexLists, results);
        }

        return BuildMatrix(results);
    }

    /// <summary>
    /// Validates and materializes the input batch, mirroring the conventions of the other embedding models.
    /// </summary>
    private static IList<string> ValidateBatch(IEnumerable<string> texts)
    {
        if (texts == null)
            throw new ArgumentNullException(nameof(texts));

        var textList = texts.ToList();
        if (textList.Count == 0)
            throw new ArgumentException("Text collection cannot be empty", nameof(texts));

        foreach (var text in textList)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Text cannot be null or empty", nameof(texts));
        }

        return textList;
    }

    /// <summary>
    /// Fills <paramref name="results"/> with cache hits and collects the distinct misses (de-duplicated by key,
    /// in first-seen order) together with the result indices that each miss must populate.
    /// </summary>
    /// <returns>The distinct miss texts to send to the inner model.</returns>
    private List<string> ResolveHitsAndCollectMisses(
        IList<string> textList,
        Vector<T>[] results,
        out List<List<int>> missIndexLists,
        out List<string> missKeys)
    {
        var missTexts = new List<string>();
        missKeys = new List<string>();
        missIndexLists = new List<List<int>>();

        // Maps a miss key to its position within the miss lists so duplicates in the same batch collapse.
        var keyToMissPosition = new Dictionary<string, int>();

        for (int i = 0; i < textList.Count; i++)
        {
            var key = BuildKey(textList[i]);

            if (_cache.TryGet(key, out var cached) && cached != null)
            {
                results[i] = cached;
                continue;
            }

            if (keyToMissPosition.TryGetValue(key, out var pos))
            {
                missIndexLists[pos].Add(i);
            }
            else
            {
                keyToMissPosition[key] = missTexts.Count;
                missTexts.Add(textList[i]);
                missKeys.Add(key);
                missIndexLists.Add(new List<int> { i });
            }
        }

        return missTexts;
    }

    /// <summary>
    /// Stores freshly computed miss embeddings in the cache and copies them into every result slot that requested them.
    /// </summary>
    private void StoreAndDistributeMisses(
        Matrix<T> missMatrix,
        List<string> missKeys,
        List<List<int>> missIndexLists,
        Vector<T>[] results)
    {
        if (missMatrix.Rows != missKeys.Count)
        {
            throw new InvalidOperationException(
                $"Inner model returned {missMatrix.Rows} embeddings for {missKeys.Count} distinct texts.");
        }

        for (int m = 0; m < missKeys.Count; m++)
        {
            var vector = GetRow(missMatrix, m);
            _cache.Set(missKeys[m], Clone(vector));

            foreach (var index in missIndexLists[m])
            {
                results[index] = vector;
            }
        }
    }

    /// <summary>
    /// Extracts a single row of a matrix as a vector.
    /// </summary>
    private static Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var cols = matrix.Columns;
        var values = new T[cols];
        for (int j = 0; j < cols; j++)
        {
            values[j] = matrix[row, j];
        }

        return new Vector<T>(values);
    }

    /// <summary>
    /// Combines the ordered result vectors into a matrix, one row per input text.
    /// </summary>
    private Matrix<T> BuildMatrix(Vector<T>[] results)
    {
        var rows = results.Length;
        var cols = EmbeddingDimension;
        var matrix = new Matrix<T>(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            var vector = results[i];
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = vector[j];
            }
        }

        return matrix;
    }

    /// <summary>
    /// Creates a defensive copy of a vector so that cached data cannot be mutated by callers or the inner model.
    /// </summary>
    private static Vector<T> Clone(Vector<T> vector)
    {
        var values = new T[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            values[i] = vector[i];
        }

        return new Vector<T>(values);
    }
}
