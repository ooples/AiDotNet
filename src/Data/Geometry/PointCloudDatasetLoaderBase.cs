using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Geometry;

/// <summary>
/// Base class for point cloud dataset loaders that expose tensor inputs and outputs.
/// </summary>
public abstract class PointCloudDatasetLoaderBase<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private int _sampleCount;
    private int _pointsPerSample;
    private int _featureDimension;
    private int _outputDimension;

    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;

    /// <summary>
    /// Number of points per sample.
    /// </summary>
    public int PointsPerSample
    {
        get
        {
            EnsureLoaded();
            return _pointsPerSample;
        }
    }

    /// <summary>
    /// Number of features per point.
    /// </summary>
    public int FeatureDimension
    {
        get
        {
            EnsureLoaded();
            return _featureDimension;
        }
    }

    /// <inheritdoc/>
    public override int FeatureCount
    {
        get
        {
            EnsureLoaded();
            return _pointsPerSample;
        }
    }

    /// <inheritdoc/>
    public override int OutputDimension
    {
        get
        {
            EnsureLoaded();
            return _outputDimension;
        }
    }

    /// <summary>
    /// Assigns loaded tensors and initializes indexing metadata.
    /// </summary>
    protected void SetLoadedData(Tensor<T> features, Tensor<T> labels)
    {
        if (features == null)
        {
            throw new ArgumentNullException(nameof(features));
        }

        if (labels == null)
        {
            throw new ArgumentNullException(nameof(labels));
        }

        if (features.Shape.Length < 2)
        {
            throw new ArgumentException("Features tensor must have at least 2 dimensions.", nameof(features));
        }

        if (labels.Shape.Length < 1)
        {
            throw new ArgumentException("Labels tensor must have at least 1 dimension.", nameof(labels));
        }

        if (features.Shape[0] != labels.Shape[0])
        {
            throw new ArgumentException("Feature and label tensors must have matching sample counts.", nameof(labels));
        }

        LoadedFeatures = features;
        LoadedLabels = labels;

        _sampleCount = features.Shape[0];
        _pointsPerSample = features.Shape[1];
        _featureDimension = features.Shape.Length > 2 ? features.Shape[2] : 1;
        _outputDimension = labels.Shape.Length > 1 ? labels.Shape[1] : 1;

        InitializeIndices(_sampleCount);
    }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = GetLoadedFeatures();
        var labels = GetLoadedLabels();

        var batchFeatures = ExtractTensorBatch(features, indices);
        var batchLabels = ExtractTensorBatch(labels, indices);

        return (batchFeatures, batchLabels);
    }

    /// <inheritdoc/>
    public override (IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Train,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Validation,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Test) Split(
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);

        var totalCount = _sampleCount;
        var (trainSize, valSize, testSize) = ComputeSplitSizes(totalCount, trainRatio, validationRatio);

        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var shuffled = Enumerable.Range(0, totalCount).OrderBy(_ => random.Next()).ToArray();

        var trainIndices = shuffled.Take(trainSize).ToArray();
        var valIndices = shuffled.Skip(trainSize).Take(valSize).ToArray();
        var testIndices = shuffled.Skip(trainSize + valSize).Take(testSize).ToArray();

        var trainFeatures = ExtractTensorBatch(GetLoadedFeatures(), trainIndices);
        var trainLabels = ExtractTensorBatch(GetLoadedLabels(), trainIndices);
        var valFeatures = ExtractTensorBatch(GetLoadedFeatures(), valIndices);
        var valLabels = ExtractTensorBatch(GetLoadedLabels(), valIndices);
        var testFeatures = ExtractTensorBatch(GetLoadedFeatures(), testIndices);
        var testLabels = ExtractTensorBatch(GetLoadedLabels(), testIndices);

        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(trainFeatures, trainLabels),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(valFeatures, valLabels),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(testFeatures, testLabels)
        );
    }

    /// <summary>
    /// Builds sample indices for point selection with sampling and padding strategies.
    /// </summary>
    protected static int[] BuildSampleIndices(
        int pointCount,
        int pointsPerSample,
        PointSamplingStrategy samplingStrategy,
        PointPaddingStrategy paddingStrategy,
        Random random)
    {
        if (pointsPerSample <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pointsPerSample), "Points per sample must be positive.");
        }

        if (pointCount <= 0)
        {
            var emptyIndices = new int[pointsPerSample];
            for (int i = 0; i < emptyIndices.Length; i++)
            {
                emptyIndices[i] = -1;
            }
            return emptyIndices;
        }

        if (pointCount >= pointsPerSample)
        {
            if (samplingStrategy == PointSamplingStrategy.Random)
            {
                return SampleWithoutReplacement(random, pointCount, pointsPerSample);
            }

            var sequential = new int[pointsPerSample];
            for (int i = 0; i < sequential.Length; i++)
            {
                sequential[i] = i;
            }
            return sequential;
        }

        var padded = new int[pointsPerSample];
        if (paddingStrategy == PointPaddingStrategy.Zero)
        {
            for (int i = 0; i < padded.Length; i++)
            {
                padded[i] = i < pointCount ? i : -1;
            }
            return padded;
        }

        if (samplingStrategy == PointSamplingStrategy.Random)
        {
            for (int i = 0; i < padded.Length; i++)
            {
                padded[i] = random.Next(pointCount);
            }
            return padded;
        }

        for (int i = 0; i < padded.Length; i++)
        {
            padded[i] = i % pointCount;
        }
        return padded;
    }

    private static int[] SampleWithoutReplacement(Random random, int count, int sampleCount)
    {
        var indices = Enumerable.Range(0, count).ToArray();
        for (int i = 0; i < sampleCount; i++)
        {
            int j = random.Next(i, count);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var result = new int[sampleCount];
        Array.Copy(indices, result, sampleCount);
        return result;
    }

    private static Tensor<T> ExtractTensorBatch(Tensor<T> source, int[] indices)
    {
        if (source.Shape.Length == 0)
        {
            throw new ArgumentException("Source tensor must have at least one dimension.", nameof(source));
        }

        var newShape = (int[])source.Shape.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);

        if (source.Shape.Length == 1)
        {
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = source[indices[i]];
            }
            return result;
        }

        for (int i = 0; i < indices.Length; i++)
        {
            CopyTensorSample(source, result, indices[i], i);
        }

        return result;
    }

    private static void CopyTensorSample(Tensor<T> source, Tensor<T> dest, int srcIndex, int destIndex)
    {
        if (source.Shape.Length != dest.Shape.Length)
        {
            throw new ArgumentException("Source and destination tensors must have the same rank.", nameof(dest));
        }

        var indices = new int[source.Shape.Length];
        CopyTensorSampleRecursive(source, dest, srcIndex, destIndex, 1, indices);
    }

    private static void CopyTensorSampleRecursive(
        Tensor<T> source,
        Tensor<T> dest,
        int srcIndex,
        int destIndex,
        int currentDim,
        int[] indices)
    {
        if (currentDim == source.Rank)
        {
            indices[0] = srcIndex;
            T value = source[indices];
            indices[0] = destIndex;
            dest[indices] = value;
            return;
        }

        for (int i = 0; i < source.Shape[currentDim]; i++)
        {
            indices[currentDim] = i;
            CopyTensorSampleRecursive(source, dest, srcIndex, destIndex, currentDim + 1, indices);
        }
    }

    private Tensor<T> GetLoadedFeatures()
    {
        if (LoadedFeatures == null)
        {
            throw new InvalidOperationException("Features are not loaded.");
        }

        return LoadedFeatures;
    }

    private Tensor<T> GetLoadedLabels()
    {
        if (LoadedLabels == null)
        {
            throw new InvalidOperationException("Labels are not loaded.");
        }

        return LoadedLabels;
    }
}
