using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.Data;

/// <summary>
/// In-memory implementation of a dataset for active learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input features.</typeparam>
/// <typeparam name="TOutput">The type of output labels.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This is the most common dataset implementation.
/// It stores all data in memory, making it fast to access but limited by
/// available RAM. Suitable for most datasets that fit in memory.</para>
///
/// <para><b>Features:</b></para>
/// <list type="bullet">
/// <item><description>Fast random access to any sample</description></item>
/// <item><description>Immutable operations return new dataset instances</description></item>
/// <item><description>Supports labeled and unlabeled data</description></item>
/// <item><description>Thread-safe read operations</description></item>
/// </list>
///
/// <para><b>Usage:</b></para>
/// <code>
/// // Create a labeled dataset
/// var inputs = new[] { new double[] { 1, 2 }, new double[] { 3, 4 } };
/// var outputs = new[] { 0, 1 };
/// var dataset = new InMemoryDataset&lt;double, double[], int&gt;(inputs, outputs);
///
/// // Create an unlabeled dataset
/// var unlabeled = InMemoryDataset&lt;double, double[], int&gt;.CreateUnlabeled(inputs);
/// </code>
/// </remarks>
public class InMemoryDataset<T, TInput, TOutput> : IExtendedDataset<T, TInput, TOutput>
{
    private readonly TInput[] _inputs;
    private readonly TOutput[] _outputs;
    private readonly bool _hasLabels;
    private readonly T[]? _sampleWeights;
    private readonly string[]? _featureNames;
    private readonly TOutput[]? _classLabels;
    private readonly DatasetMetadata? _metadata;
    private readonly int _featureCount;
    private readonly bool _isClassification;

    /// <inheritdoc/>
    public int Count => _inputs.Length;

    /// <inheritdoc/>
    public IReadOnlyList<TInput> Inputs => _inputs;

    /// <inheritdoc/>
    public IReadOnlyList<TOutput> Outputs => _outputs;

    /// <inheritdoc/>
    public bool HasLabels => _hasLabels;

    /// <inheritdoc/>
    public IReadOnlyList<string>? FeatureNames => _featureNames;

    /// <inheritdoc/>
    public int FeatureCount => _featureCount;

    /// <inheritdoc/>
    public int ClassCount => _classLabels?.Length ?? 0;

    /// <inheritdoc/>
    public IReadOnlyList<TOutput>? ClassLabels => _classLabels;

    /// <inheritdoc/>
    public IReadOnlyList<T>? SampleWeights => _sampleWeights;

    /// <inheritdoc/>
    public bool IsClassification => _isClassification;

    /// <inheritdoc/>
    public DatasetMetadata? Metadata => _metadata;

    /// <summary>
    /// Creates a new labeled dataset.
    /// </summary>
    /// <param name="inputs">The input features.</param>
    /// <param name="outputs">The output labels.</param>
    public InMemoryDataset(TInput[] inputs, TOutput[] outputs)
        : this(inputs, outputs, hasLabels: true)
    {
    }

    /// <summary>
    /// Creates a new dataset with optional labeling.
    /// </summary>
    /// <param name="inputs">The input features.</param>
    /// <param name="outputs">The output labels (can be default values for unlabeled).</param>
    /// <param name="hasLabels">Whether the dataset has valid labels.</param>
    public InMemoryDataset(TInput[] inputs, TOutput[] outputs, bool hasLabels)
        : this(inputs, outputs, hasLabels, null, null, null, null, 0, false)
    {
    }

    /// <summary>
    /// Creates a new dataset with full configuration.
    /// </summary>
    private InMemoryDataset(
        TInput[] inputs,
        TOutput[] outputs,
        bool hasLabels,
        T[]? sampleWeights,
        string[]? featureNames,
        TOutput[]? classLabels,
        DatasetMetadata? metadata,
        int featureCount,
        bool isClassification)
    {
        if (inputs.Length != outputs.Length)
        {
            throw new ArgumentException(
                $"Inputs length ({inputs.Length}) must match outputs length ({outputs.Length}).");
        }

        _inputs = inputs;
        _outputs = outputs;
        _hasLabels = hasLabels;
        _sampleWeights = sampleWeights;
        _featureNames = featureNames;
        _classLabels = classLabels;
        _metadata = metadata;
        _featureCount = featureCount > 0 ? featureCount : InferFeatureCount(inputs);
        _isClassification = isClassification;
    }

    /// <inheritdoc/>
    public TInput GetInput(int index)
    {
        ValidateIndex(index);
        return _inputs[index];
    }

    /// <inheritdoc/>
    public TOutput GetOutput(int index)
    {
        ValidateIndex(index);
        return _outputs[index];
    }

    /// <inheritdoc/>
    public (TInput Input, TOutput Output) GetSample(int index)
    {
        ValidateIndex(index);
        return (_inputs[index], _outputs[index]);
    }

    /// <inheritdoc/>
    public IDataset<T, TInput, TOutput> Subset(int[] indices)
    {
        if (indices.Length == 0)
        {
            return CreateEmpty();
        }

        var newInputs = new TInput[indices.Length];
        var newOutputs = new TOutput[indices.Length];
        T[]? newWeights = _sampleWeights != null ? new T[indices.Length] : null;

        for (int i = 0; i < indices.Length; i++)
        {
            ValidateIndex(indices[i]);
            newInputs[i] = _inputs[indices[i]];
            newOutputs[i] = _outputs[indices[i]];
            if (newWeights != null && _sampleWeights != null)
            {
                newWeights[i] = _sampleWeights[indices[i]];
            }
        }

        return new InMemoryDataset<T, TInput, TOutput>(
            newInputs, newOutputs, _hasLabels, newWeights,
            _featureNames, _classLabels, _metadata, _featureCount, _isClassification);
    }

    /// <inheritdoc/>
    public IDataset<T, TInput, TOutput> Except(int[] indices)
    {
        if (indices.Length == 0)
        {
            return Clone();
        }

        var excludeSet = new HashSet<int>(indices);
        var keepIndices = new List<int>();

        for (int i = 0; i < Count; i++)
        {
            if (!excludeSet.Contains(i))
            {
                keepIndices.Add(i);
            }
        }

        return Subset(keepIndices.ToArray());
    }

    /// <inheritdoc/>
    public IDataset<T, TInput, TOutput> Merge(IDataset<T, TInput, TOutput> other)
    {
        var newInputs = new TInput[Count + other.Count];
        var newOutputs = new TOutput[Count + other.Count];

        Array.Copy(_inputs, 0, newInputs, 0, Count);
        Array.Copy(_outputs, 0, newOutputs, 0, Count);

        for (int i = 0; i < other.Count; i++)
        {
            newInputs[Count + i] = other.GetInput(i);
            newOutputs[Count + i] = other.GetOutput(i);
        }

        bool newHasLabels = _hasLabels && other.HasLabels;

        return new InMemoryDataset<T, TInput, TOutput>(
            newInputs, newOutputs, newHasLabels, null,
            _featureNames, _classLabels, _metadata, _featureCount, _isClassification);
    }

    /// <inheritdoc/>
    public IDataset<T, TInput, TOutput> AddSamples(TInput[] inputs, TOutput[] outputs)
    {
        if (inputs.Length != outputs.Length)
        {
            throw new ArgumentException("Inputs and outputs must have the same length.");
        }

        var newInputs = new TInput[Count + inputs.Length];
        var newOutputs = new TOutput[Count + inputs.Length];

        Array.Copy(_inputs, 0, newInputs, 0, Count);
        Array.Copy(_outputs, 0, newOutputs, 0, Count);
        Array.Copy(inputs, 0, newInputs, Count, inputs.Length);
        Array.Copy(outputs, 0, newOutputs, Count, outputs.Length);

        return new InMemoryDataset<T, TInput, TOutput>(
            newInputs, newOutputs, _hasLabels, null,
            _featureNames, _classLabels, _metadata, _featureCount, _isClassification);
    }

    /// <inheritdoc/>
    public IDataset<T, TInput, TOutput> RemoveSamples(int[] indices)
    {
        return Except(indices);
    }

    /// <inheritdoc/>
    public IDataset<T, TInput, TOutput> UpdateLabels(int[] indices, TOutput[] labels)
    {
        if (indices.Length != labels.Length)
        {
            throw new ArgumentException("Indices and labels must have the same length.");
        }

        var newOutputs = (TOutput[])_outputs.Clone();
        for (int i = 0; i < indices.Length; i++)
        {
            ValidateIndex(indices[i]);
            newOutputs[indices[i]] = labels[i];
        }

        return new InMemoryDataset<T, TInput, TOutput>(
            _inputs, newOutputs, true, _sampleWeights,
            _featureNames, _classLabels, _metadata, _featureCount, _isClassification);
    }

    /// <inheritdoc/>
    public IDataset<T, TInput, TOutput> Shuffle(Random? random = null)
    {
        random ??= RandomHelper.Shared;
        var indices = GetIndices();
        RandomHelper.Shuffle(indices, random);
        return Subset(indices);
    }

    /// <inheritdoc/>
    public (IDataset<T, TInput, TOutput> Train, IDataset<T, TInput, TOutput> Test) Split(
        double trainRatio = 0.8,
        Random? random = null)
    {
        trainRatio = MathHelper.Clamp(trainRatio, 0.0, 1.0);
        random ??= RandomHelper.Shared;

        var indices = GetIndices();
        RandomHelper.Shuffle(indices, random);

        int trainCount = (int)(Count * trainRatio);
        var trainIndices = new int[trainCount];
        Array.Copy(indices, 0, trainIndices, 0, trainCount);
        var testIndices = new int[indices.Length - trainCount];
        Array.Copy(indices, trainCount, testIndices, 0, indices.Length - trainCount);

        return (Subset(trainIndices), Subset(testIndices));
    }

    /// <inheritdoc/>
    public int[] GetIndices()
    {
        var indices = new int[Count];
        for (int i = 0; i < Count; i++)
        {
            indices[i] = i;
        }
        return indices;
    }

    /// <inheritdoc/>
    public IDataset<T, TInput, TOutput> Clone()
    {
        return new InMemoryDataset<T, TInput, TOutput>(
            (TInput[])_inputs.Clone(),
            (TOutput[])_outputs.Clone(),
            _hasLabels,
            _sampleWeights != null ? (T[])_sampleWeights.Clone() : null,
            _featureNames,
            _classLabels,
            _metadata,
            _featureCount,
            _isClassification);
    }

    /// <summary>
    /// Creates an empty dataset.
    /// </summary>
    /// <returns>An empty dataset.</returns>
    public static InMemoryDataset<T, TInput, TOutput> CreateEmpty()
    {
        return new InMemoryDataset<T, TInput, TOutput>(
            Array.Empty<TInput>(),
            Array.Empty<TOutput>(),
            true);
    }

    /// <summary>
    /// Creates an unlabeled dataset.
    /// </summary>
    /// <param name="inputs">The input features.</param>
    /// <returns>An unlabeled dataset.</returns>
    public static InMemoryDataset<T, TInput, TOutput> CreateUnlabeled(TInput[] inputs)
    {
        var outputs = new TOutput[inputs.Length];
        return new InMemoryDataset<T, TInput, TOutput>(inputs, outputs, hasLabels: false);
    }

    /// <summary>
    /// Creates a dataset with sample weights.
    /// </summary>
    /// <param name="inputs">The input features.</param>
    /// <param name="outputs">The output labels.</param>
    /// <param name="weights">Sample weights.</param>
    /// <returns>A weighted dataset.</returns>
    public static InMemoryDataset<T, TInput, TOutput> CreateWeighted(
        TInput[] inputs,
        TOutput[] outputs,
        T[] weights)
    {
        if (weights.Length != inputs.Length)
        {
            throw new ArgumentException("Weights length must match inputs length.");
        }

        return new InMemoryDataset<T, TInput, TOutput>(
            inputs, outputs, true, weights, null, null, null, 0, false);
    }

    /// <summary>
    /// Creates a classification dataset with class labels.
    /// </summary>
    /// <param name="inputs">The input features.</param>
    /// <param name="outputs">The class labels.</param>
    /// <param name="classLabels">All unique class labels.</param>
    /// <returns>A classification dataset.</returns>
    public static InMemoryDataset<T, TInput, TOutput> CreateClassification(
        TInput[] inputs,
        TOutput[] outputs,
        TOutput[] classLabels)
    {
        return new InMemoryDataset<T, TInput, TOutput>(
            inputs, outputs, true, null, null, classLabels, null, 0, true);
    }

    /// <summary>
    /// Creates a dataset with metadata.
    /// </summary>
    /// <param name="inputs">The input features.</param>
    /// <param name="outputs">The output labels.</param>
    /// <param name="metadata">Dataset metadata.</param>
    /// <returns>A dataset with metadata.</returns>
    public static InMemoryDataset<T, TInput, TOutput> CreateWithMetadata(
        TInput[] inputs,
        TOutput[] outputs,
        DatasetMetadata metadata)
    {
        return new InMemoryDataset<T, TInput, TOutput>(
            inputs, outputs, true, null, null, null, metadata, 0, false);
    }

    #region Private Methods

    private void ValidateIndex(int index)
    {
        if (index < 0 || index >= Count)
        {
            throw new ArgumentOutOfRangeException(
                nameof(index),
                $"Index {index} is out of range [0, {Count - 1}].");
        }
    }

    private static int InferFeatureCount(TInput[] inputs)
    {
        if (inputs.Length == 0)
        {
            return 0;
        }

        var first = inputs[0];
        return first switch
        {
            Array arr => arr.Length,
            ICollection<T> col => col.Count,
            Vector<T> vec => vec.Length,
            _ => 1 // Scalar input
        };
    }

    #endregion
}

/// <summary>
/// Factory implementation for creating in-memory datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input features.</typeparam>
/// <typeparam name="TOutput">The type of output labels.</typeparam>
public class InMemoryDatasetFactory<T, TInput, TOutput> : IDatasetFactory<T, TInput, TOutput>
{
    /// <inheritdoc/>
    public IDataset<T, TInput, TOutput> Create(TInput[] inputs, TOutput[] outputs)
    {
        return new InMemoryDataset<T, TInput, TOutput>(inputs, outputs);
    }

    /// <inheritdoc/>
    public IDataset<T, TInput, TOutput> CreateUnlabeled(TInput[] inputs)
    {
        return InMemoryDataset<T, TInput, TOutput>.CreateUnlabeled(inputs);
    }

    /// <inheritdoc/>
    public IDataset<T, TInput, TOutput> CreateEmpty()
    {
        return InMemoryDataset<T, TInput, TOutput>.CreateEmpty();
    }
}
