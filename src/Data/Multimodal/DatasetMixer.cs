using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Multimodal;

/// <summary>
/// Blends multiple datasets by weight ratios, producing mixed batches for curriculum or domain mixing.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Dataset mixing is a crucial technique for training large language models and multimodal models.
/// It allows combining data from multiple sources (domains) with controlled ratios, enabling:
/// - Domain balancing (upweight rare domains)
/// - Curriculum learning (gradually shift domain mix during training)
/// - Multi-task training (combine datasets for different tasks)
/// </para>
/// <para><b>For Beginners:</b> Imagine you have three types of training data: books, code, and
/// conversations. You want your model to see 50% books, 30% code, and 20% conversations.
/// DatasetMixer handles this:
/// <code>
/// var mixer = new DatasetMixer&lt;float&gt;();
/// mixer.AddSource(booksDataset, weight: 0.5, name: "books");
/// mixer.AddSource(codeDataset, weight: 0.3, name: "code");
/// mixer.AddSource(convDataset, weight: 0.2, name: "conversations");
///
/// // Get mixed batches
/// foreach (var (sourceIndex, sampleIndex) in mixer.GetMixedBatchIndices(batchSize: 32))
/// {
///     var sample = mixer.GetSample(sourceIndex, sampleIndex);
/// }
/// </code>
/// </para>
/// </remarks>
public class DatasetMixer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly List<MixerSource<T>> _sources = new List<MixerSource<T>>();
    private double[]? _normalizedWeights;

    /// <summary>
    /// Gets the number of data sources in the mixer.
    /// </summary>
    public int SourceCount => _sources.Count;

    /// <summary>
    /// Gets the total number of samples across all sources.
    /// </summary>
    public int TotalSamples
    {
        get
        {
            int total = 0;
            foreach (var source in _sources)
            {
                total += source.SampleCount;
            }
            return total;
        }
    }

    /// <summary>
    /// Gets the normalized mixing weights (summing to 1.0).
    /// </summary>
    public IReadOnlyList<double> NormalizedWeights
    {
        get
        {
            EnsureWeightsNormalized();
            return _normalizedWeights ?? Array.Empty<double>();
        }
    }

    /// <summary>
    /// Adds a dataset source with a mixing weight.
    /// </summary>
    /// <param name="dataset">The multimodal dataset to add.</param>
    /// <param name="weight">Relative weight for this source (higher = sampled more often).</param>
    /// <param name="name">Optional name for this source.</param>
    public void AddSource(MultimodalDataset<T> dataset, double weight = 1.0, string? name = null)
    {
        if (dataset is null)
            throw new ArgumentNullException(nameof(dataset));
        if (weight <= 0)
            throw new ArgumentOutOfRangeException(nameof(weight), "Weight must be positive.");
        if (dataset.Count == 0)
            throw new ArgumentException("Cannot add an empty dataset source.", nameof(dataset));

        _sources.Add(new MixerSource<T>(dataset, weight, name ?? $"source_{_sources.Count}"));
        _normalizedWeights = null; // Invalidate
    }

    /// <summary>
    /// Adds a tensor-based data source with a mixing weight.
    /// </summary>
    /// <param name="features">Feature tensor where first dimension is sample count.</param>
    /// <param name="labels">Label tensor where first dimension is sample count.</param>
    /// <param name="modalityType">The modality type of the features.</param>
    /// <param name="weight">Relative weight for this source.</param>
    /// <param name="name">Optional name for this source.</param>
    public void AddTensorSource(
        Tensor<T> features,
        Tensor<T>? labels,
        ModalityType modalityType,
        double weight = 1.0,
        string? name = null)
    {
        if (features is null)
            throw new ArgumentNullException(nameof(features));
        if (features.Shape.Length < 2)
            throw new ArgumentException("Features must have at least 2 dimensions [N, ...].", nameof(features));

        int sampleCount = features.Shape[0];
        var dataset = new MultimodalDataset<T>();

        // Build per-sample shape (all dims except first)
        int[] sampleShape = new int[features.Shape.Length - 1];
        Array.Copy(features.Shape, 1, sampleShape, 0, sampleShape.Length);

        int elementsPerSample = 1;
        for (int d = 0; d < sampleShape.Length; d++)
        {
            elementsPerSample *= sampleShape[d];
        }

        int labelsPerSample = 0;
        int[] labelShape = Array.Empty<int>();
        if (labels is not null && labels.Shape.Length >= 2)
        {
            labelShape = new int[labels.Shape.Length - 1];
            Array.Copy(labels.Shape, 1, labelShape, 0, labelShape.Length);
            labelsPerSample = 1;
            for (int d = 0; d < labelShape.Length; d++)
            {
                labelsPerSample *= labelShape[d];
            }
        }

        var featSpan = features.Data.Span;

        for (int i = 0; i < sampleCount; i++)
        {
            // Extract feature data for this sample
            var sampleData = new T[elementsPerSample];
            featSpan.Slice(i * elementsPerSample, elementsPerSample).CopyTo(sampleData.AsSpan());
            var sampleTensor = new Tensor<T>(sampleData, sampleShape);

            var modality = new ModalitySample<T>(modalityType, sampleTensor);
            var multiSample = new MultimodalSample<T>(modality);

            // Extract label if available
            if (labels is not null && labelsPerSample > 0)
            {
                var lblSpan = labels.Data.Span;
                var labelData = new T[labelsPerSample];
                lblSpan.Slice(i * labelsPerSample, labelsPerSample).CopyTo(labelData.AsSpan());
                multiSample.Label = new Tensor<T>(labelData, labelShape);
            }

            dataset.Add(multiSample);
        }

        AddSource(dataset, weight, name);
    }

    /// <summary>
    /// Gets a sample from a specific source at a specific index.
    /// </summary>
    /// <param name="sourceIndex">The source index.</param>
    /// <param name="sampleIndex">The sample index within that source.</param>
    /// <returns>The multimodal sample.</returns>
    public MultimodalSample<T> GetSample(int sourceIndex, int sampleIndex)
    {
        if (sourceIndex < 0 || sourceIndex >= _sources.Count)
            throw new ArgumentOutOfRangeException(nameof(sourceIndex));

        return _sources[sourceIndex].Dataset[sampleIndex];
    }

    /// <summary>
    /// Generates batch indices following the mixing weights distribution.
    /// </summary>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>Enumerable of (sourceIndex, sampleIndex) pairs for one batch.</returns>
    public IEnumerable<(int SourceIndex, int SampleIndex)> GetMixedBatchIndices(
        int batchSize,
        int? seed = null)
    {
        if (_sources.Count == 0)
            throw new InvalidOperationException("No data sources have been added.");
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize));

        var weights = GetNormalizedWeightsArray();

        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Build cumulative distribution for weighted selection
        var cumulativeWeights = new double[_sources.Count];
        cumulativeWeights[0] = weights[0];
        for (int i = 1; i < _sources.Count; i++)
        {
            cumulativeWeights[i] = cumulativeWeights[i - 1] + weights[i];
        }

        // Track per-source position for sequential access within each source
        var sourcePositions = new int[_sources.Count];

        for (int b = 0; b < batchSize; b++)
        {
            // Weighted random source selection
            double r = random.NextDouble();
            int sourceIdx = 0;
            for (int s = 0; s < cumulativeWeights.Length; s++)
            {
                if (r <= cumulativeWeights[s])
                {
                    sourceIdx = s;
                    break;
                }
            }

            // Get sample index with wrap-around
            int sampleIdx = sourcePositions[sourceIdx] % _sources[sourceIdx].SampleCount;
            sourcePositions[sourceIdx]++;

            yield return (sourceIdx, sampleIdx);
        }
    }

    /// <summary>
    /// Generates an infinite stream of mixed batch indices, following mixing weights.
    /// </summary>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Enumerable of batches, where each batch is an array of (sourceIndex, sampleIndex) pairs.</returns>
    public IEnumerable<(int SourceIndex, int SampleIndex)[]> GetMixedBatches(
        int batchSize,
        int? seed = null)
    {
        if (_sources.Count == 0)
            throw new InvalidOperationException("No data sources have been added.");

        var weights = GetNormalizedWeightsArray();

        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        var cumulativeWeights = new double[_sources.Count];
        cumulativeWeights[0] = weights[0];
        for (int i = 1; i < _sources.Count; i++)
        {
            cumulativeWeights[i] = cumulativeWeights[i - 1] + weights[i];
        }

        var sourcePositions = new int[_sources.Count];

        while (true)
        {
            var batch = new (int SourceIndex, int SampleIndex)[batchSize];

            for (int b = 0; b < batchSize; b++)
            {
                double r = random.NextDouble();
                int sourceIdx = 0;
                for (int s = 0; s < cumulativeWeights.Length; s++)
                {
                    if (r <= cumulativeWeights[s])
                    {
                        sourceIdx = s;
                        break;
                    }
                }

                int sampleIdx = sourcePositions[sourceIdx] % _sources[sourceIdx].SampleCount;
                sourcePositions[sourceIdx]++;

                batch[b] = (sourceIdx, sampleIdx);
            }

            yield return batch;
        }
    }

    /// <summary>
    /// Updates the mixing weight for a specific source (for curriculum learning).
    /// </summary>
    /// <param name="sourceIndex">The source index.</param>
    /// <param name="newWeight">The new weight value.</param>
    public void UpdateWeight(int sourceIndex, double newWeight)
    {
        if (sourceIndex < 0 || sourceIndex >= _sources.Count)
            throw new ArgumentOutOfRangeException(nameof(sourceIndex));
        if (newWeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(newWeight), "Weight must be positive.");

        _sources[sourceIndex] = new MixerSource<T>(
            _sources[sourceIndex].Dataset,
            newWeight,
            _sources[sourceIndex].Name);
        _normalizedWeights = null; // Invalidate
    }

    /// <summary>
    /// Gets information about all sources.
    /// </summary>
    /// <returns>Array of (name, sampleCount, normalizedWeight) tuples.</returns>
    public (string Name, int SampleCount, double NormalizedWeight)[] GetSourceInfo()
    {
        var weights = GetNormalizedWeightsArray();

        var result = new (string, int, double)[_sources.Count];
        for (int i = 0; i < _sources.Count; i++)
        {
            result[i] = (_sources[i].Name, _sources[i].SampleCount, weights[i]);
        }

        return result;
    }

    private double[] GetNormalizedWeightsArray()
    {
        EnsureWeightsNormalized();
        return _normalizedWeights ?? Array.Empty<double>();
    }

    private void EnsureWeightsNormalized()
    {
        if (_normalizedWeights is not null) return;

        if (_sources.Count == 0)
        {
            _normalizedWeights = Array.Empty<double>();
            return;
        }

        double totalWeight = 0;
        foreach (var source in _sources)
        {
            totalWeight += source.Weight;
        }

        _normalizedWeights = new double[_sources.Count];
        for (int i = 0; i < _sources.Count; i++)
        {
            _normalizedWeights[i] = _sources[i].Weight / totalWeight;
        }
    }
}

/// <summary>
/// Represents a single data source within a DatasetMixer.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal class MixerSource<T>
{
    public MultimodalDataset<T> Dataset { get; }
    public double Weight { get; }
    public string Name { get; }
    public int SampleCount => Dataset.Count;

    public MixerSource(MultimodalDataset<T> dataset, double weight, string name)
    {
        Dataset = dataset;
        Weight = weight;
        Name = name;
    }
}
