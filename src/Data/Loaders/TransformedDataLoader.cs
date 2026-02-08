using AiDotNet.Data.Transforms;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Wraps a data loader and applies a composable transform pipeline to feature data during batch extraction.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// This wrapper applies an <see cref="ITransform{TInput, TOutput}"/> to every sample's feature
/// data when batches are retrieved. The underlying loader's data remains unmodified.
/// </para>
/// <para><b>For Beginners:</b> Use this to add normalization, scaling, or other preprocessing
/// to any existing data loader without modifying the original loader:
/// <code>
/// var loader = DataLoaders.FromArrays(features, labels);
/// var transform = new NormalizeTransform&lt;float&gt;(mean, std);
/// var transformed = new TransformedDataLoader&lt;float&gt;(loader, transform);
/// await transformed.LoadAsync();
/// foreach (var (feat, lbl) in transformed.GetBatches(batchSize: 32))
/// {
///     // feat has been normalized
/// }
/// </code>
/// </para>
/// </remarks>
public class TransformedDataLoader<T> :
    DataLoaderBase<T>,
    IInputOutputDataLoader<T, Tensor<T>, Tensor<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>> _inner;
    private readonly ITransform<T[], T[]> _transform;

    /// <inheritdoc/>
    public override string Name => $"Transformed({_inner.Name})";

    /// <inheritdoc/>
    public override string Description => $"Transform wrapper over {_inner.Description}";

    /// <inheritdoc/>
    public override int TotalCount => _inner.TotalCount;

    /// <inheritdoc/>
    public Tensor<T> Features => ApplyTransformToTensor(_inner.Features);

    /// <inheritdoc/>
    public Tensor<T> Labels => _inner.Labels;

    /// <inheritdoc/>
    public int FeatureCount => _inner.FeatureCount;

    /// <inheritdoc/>
    public int OutputDimension => _inner.OutputDimension;

    /// <inheritdoc/>
    public bool HasNext => _inner.HasNext;

    /// <inheritdoc/>
    public bool IsShuffled => _inner.IsShuffled;

    /// <summary>
    /// Creates a transformed data loader wrapping the given inner loader.
    /// </summary>
    /// <param name="inner">The data loader to wrap.</param>
    /// <param name="transform">The transform to apply to feature data.</param>
    public TransformedDataLoader(
        InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>> inner,
        ITransform<T[], T[]> transform)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        _transform = transform ?? throw new ArgumentNullException(nameof(transform));
    }

    /// <inheritdoc/>
    public (Tensor<T> Features, Tensor<T> Labels) GetNextBatch()
    {
        var (features, labels) = _inner.GetNextBatch();
        CurrentIndex = _inner.CurrentIndex;
        CurrentBatchIndex = _inner.CurrentBatchIndex;
        return (ApplyTransformToTensor(features), labels);
    }

    /// <inheritdoc/>
    public bool TryGetNextBatch(out (Tensor<T> Features, Tensor<T> Labels) batch)
    {
        if (_inner.TryGetNextBatch(out var innerBatch))
        {
            CurrentIndex = _inner.CurrentIndex;
            CurrentBatchIndex = _inner.CurrentBatchIndex;
            batch = (ApplyTransformToTensor(innerBatch.Features), innerBatch.Labels);
            return true;
        }

        batch = default;
        return false;
    }

    /// <inheritdoc/>
    public void Shuffle(int? seed = null)
    {
        _inner.Shuffle(seed);
    }

    /// <inheritdoc/>
    public void Unshuffle()
    {
        _inner.Unshuffle();
    }

    /// <inheritdoc/>
    public (IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Train,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Validation,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Test) Split(
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        // Split the inner loader, then wrap each part with the same transform
        var (train, val, test) = _inner.Split(trainRatio, validationRatio, seed);

        if (train is not InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>> trainBase ||
            val is not InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>> valBase ||
            test is not InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>> testBase)
        {
            throw new InvalidOperationException(
                "Inner loader's Split returned types that cannot be wrapped with TransformedDataLoader. " +
                "Expected InputOutputDataLoaderBase instances.");
        }

        return (
            new TransformedDataLoader<T>(trainBase, _transform),
            new TransformedDataLoader<T>(valBase, _transform),
            new TransformedDataLoader<T>(testBase, _transform)
        );
    }

    /// <inheritdoc/>
    public IEnumerable<(Tensor<T> Features, Tensor<T> Labels)> GetBatches(
        int? batchSize = null,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null)
    {
        foreach (var (features, labels) in _inner.GetBatches(batchSize, shuffle, dropLast, seed))
        {
            yield return (ApplyTransformToTensor(features), labels);
        }
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<(Tensor<T> Features, Tensor<T> Labels)> GetBatchesAsync(
        int? batchSize = null,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null,
        int prefetchCount = 2,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var (features, labels) in _inner.GetBatchesAsync(
            batchSize, shuffle, dropLast, seed, prefetchCount, cancellationToken))
        {
            yield return (ApplyTransformToTensor(features), labels);
        }
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        return _inner.LoadAsync(cancellationToken);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        _inner.Unload();
    }

    /// <summary>
    /// Applies the transform to each sample row in the tensor.
    /// </summary>
    private Tensor<T> ApplyTransformToTensor(Tensor<T> tensor)
    {
        if (tensor.Shape.Length < 2)
        {
            // 1D tensor: treat as single sample
            var data = new T[tensor.Data.Length];
            tensor.Data.Span.CopyTo(data.AsSpan());
            var transformed = _transform.Apply(data);

            if (transformed.Length != data.Length)
            {
                throw new InvalidOperationException(
                    $"Transform returned {transformed.Length} elements, expected {data.Length}. " +
                    "Transform must preserve sample element count.");
            }

            return new Tensor<T>(transformed, tensor.Shape);
        }

        int sampleCount = tensor.Shape[0];
        int elementsPerSample = 1;
        try
        {
            for (int d = 1; d < tensor.Shape.Length; d++)
            {
                elementsPerSample = checked(elementsPerSample * tensor.Shape[d]);
            }
        }
        catch (OverflowException)
        {
            throw new InvalidOperationException(
                "Tensor shape dimensions overflow int range. Shape is too large for transform.");
        }

        var resultData = new T[sampleCount * elementsPerSample];
        var sourceSpan = tensor.Data.Span;

        for (int i = 0; i < sampleCount; i++)
        {
            // Extract sample as flat array
            var sampleData = new T[elementsPerSample];
            sourceSpan.Slice(i * elementsPerSample, elementsPerSample).CopyTo(sampleData.AsSpan());

            // Apply transform
            var transformedSample = _transform.Apply(sampleData);

            if (transformedSample.Length != elementsPerSample)
            {
                throw new InvalidOperationException(
                    $"Transform returned {transformedSample.Length} elements, expected {elementsPerSample}. " +
                    "Transform must preserve sample element count.");
            }

            // Copy back
            Array.Copy(transformedSample, 0, resultData, i * elementsPerSample, elementsPerSample);
        }

        return new Tensor<T>(resultData, tensor.Shape);
    }
}
