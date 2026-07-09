namespace AiDotNet.Interfaces;

/// <summary>
/// Data loader parameterized by both input and output tensor types — the type-theoretic fit
/// for domains where the batch shape is richer than the row-scalar (X, y) supervised pair
/// (image-space photometric NeRF training, video / audio / sequence corpora, RL trajectories).
/// Added as part of #1834 without renaming the existing <see cref="IDataLoader{T}"/> so
/// currently-shipped loaders (matrix-in / vector-out supervised) migrate at their own pace.
/// </summary>
/// <typeparam name="TInput">Per-batch input structure (e.g. <c>ImageView&lt;T&gt;</c>, <c>Matrix&lt;T&gt;</c>).</typeparam>
/// <typeparam name="TOutput">Per-batch output structure (e.g. <c>PixelBatch&lt;T&gt;</c>, <c>Vector&lt;T&gt;</c>).</typeparam>
/// <remarks>
/// <para>
/// Reference PyTorch <c>DataLoader</c> is generic over its collate output type; this
/// interface expresses the same idea in C# nominal-typing terms. Facade branches on the
/// concrete <c>TInput</c> / <c>TOutput</c> pair via <c>is</c> checks — an image loader is a
/// <c>IDataLoader&lt;ImageView&lt;T&gt;, PixelBatch&lt;T&gt;&gt;</c>, and the model layer sees exactly
/// that shape.
/// </para>
/// <para>
/// The original <see cref="IDataLoader{T}"/> continues to be the base for row-scalar supervised
/// loaders and the specialized <c>IGraphDataLoader</c> / <c>IRLDataLoader</c> / <c>IEpisodicDataLoader</c>
/// hierarchies — no breakage, no rename. When a full migration is done, <see cref="IDataLoader{T}"/>
/// can be re-expressed as <c>IDataLoader&lt;Matrix&lt;T&gt;, Vector&lt;T&gt;&gt;</c>.
/// </para>
/// </remarks>
public interface IDataLoader<TInput, TOutput> : IResettable, ICountable
{
    /// <summary>
    /// Human-readable name of this loader.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Description of the dataset and its intended use.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Whether the underlying data has been fetched into memory and is ready to iterate.
    /// </summary>
    bool IsLoaded { get; }

    /// <summary>
    /// Loads the dataset (file read, network fetch, preprocessing) asynchronously.
    /// </summary>
    System.Threading.Tasks.Task LoadAsync(System.Threading.CancellationToken cancellationToken = default);

    /// <summary>
    /// Releases the loaded dataset from memory. The loader can be re-loaded via
    /// <see cref="LoadAsync"/>.
    /// </summary>
    void Unload();

    /// <summary>
    /// Enumerates the loaded dataset one batch at a time. Each yielded pair carries the
    /// model-shaped input and the corresponding target.
    /// </summary>
    /// <param name="batchSize">Number of items per batch; loader-specific meaning
    /// (rays per photo for image-space training, rows for tabular data).</param>
    System.Collections.Generic.IEnumerable<(TInput Input, TOutput Output)> IterateBatches(int batchSize);
}
