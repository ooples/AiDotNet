using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// A simple in-memory data loader for supervised learning data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// InMemoryDataLoader is the simplest way to create a data loader from existing data.
/// It's ideal for:
/// - Small to medium datasets that fit in memory
/// - Quick prototyping and testing
/// - Converting raw arrays or matrices to the IDataLoader interface
/// </para>
/// <para><b>For Beginners:</b> This is the easiest data loader to use. Simply pass your
/// feature data (X) and label data (Y) to the constructor, and you're ready to train!
///
/// **Example:**
/// ```csharp
/// // Create feature matrix and label vector
/// var features = new Matrix&lt;double&gt;(100, 5);  // 100 samples, 5 features
/// var labels = new Vector&lt;double&gt;(100);       // 100 labels
///
/// // Create the loader
/// var loader = new InMemoryDataLoader&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(features, labels);
///
/// // Use with PredictionModelBuilder
/// var result = await builder
///     .ConfigureDataLoader(loader)
///     .ConfigureModel(model)
///     .BuildAsync();
/// ```
/// </para>
/// </remarks>
public class InMemoryDataLoader<T, TInput, TOutput> : InputOutputDataLoaderBase<T, TInput, TOutput>
{
    private readonly TInput _features;
    private readonly TOutput _labels;
    private readonly int _sampleCount;
    private readonly int _featureCount;
    private readonly int _outputDimension;

    /// <inheritdoc/>
    public override string Name => "InMemoryDataLoader";

    /// <inheritdoc/>
    public override string Description => "In-memory data loader for supervised learning";

    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;

    /// <inheritdoc/>
    public override int FeatureCount => _featureCount;

    /// <inheritdoc/>
    public override int OutputDimension => _outputDimension;

    /// <summary>
    /// Creates a new in-memory data loader with the specified features and labels.
    /// </summary>
    /// <param name="features">The input features (X data).</param>
    /// <param name="labels">The output labels (Y data).</param>
    /// <exception cref="ArgumentNullException">Thrown when features or labels is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the sample counts don't match.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> The features are your input data - the information you use to make predictions.
    /// The labels are the correct answers you're trying to predict.
    ///
    /// For example, if you're predicting house prices:
    /// - Features: Square footage, number of bedrooms, location (as numbers)
    /// - Labels: The actual house prices
    ///
    /// Both must have the same number of samples (rows).
    /// </para>
    /// </remarks>
    public InMemoryDataLoader(TInput features, TOutput labels)
    {
        if (features is null)
        {
            throw new ArgumentNullException(nameof(features), "Features cannot be null.");
        }

        if (labels is null)
        {
            throw new ArgumentNullException(nameof(labels), "Labels cannot be null.");
        }

        _features = features;
        _labels = labels;

        // Determine sample count and dimensions based on input types
        (_sampleCount, _featureCount) = GetInputDimensions(features);
        var (labelCount, outputDim) = GetOutputDimensions(labels);
        _outputDimension = outputDim;

        if (_sampleCount != labelCount)
        {
            throw new ArgumentException(
                $"Feature sample count ({_sampleCount}) must match label sample count ({labelCount}).",
                nameof(labels));
        }
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Data is already in memory - just initialize indices
        LoadedFeatures = _features;
        LoadedLabels = _labels;
        InitializeIndices(_sampleCount);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default;
        LoadedLabels = default;
        Indices = null;
    }

    /// <inheritdoc/>
    protected override (TInput Features, TOutput Labels) ExtractBatch(int[] indices)
    {
        // Extract features and labels at the given indices
        return (ExtractFeatures(indices), ExtractLabels(indices));
    }

    /// <inheritdoc/>
    public override (IInputOutputDataLoader<T, TInput, TOutput> Train,
        IInputOutputDataLoader<T, TInput, TOutput> Validation,
        IInputOutputDataLoader<T, TInput, TOutput> Test) Split(
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);

        var (trainSize, valSize, testSize) = ComputeSplitSizes(_sampleCount, trainRatio, validationRatio);

        // Create shuffled indices
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var shuffledIndices = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();

        var trainIndices = shuffledIndices.Take(trainSize).ToArray();
        var valIndices = shuffledIndices.Skip(trainSize).Take(valSize).ToArray();
        var testIndices = shuffledIndices.Skip(trainSize + valSize).ToArray();

        // Create split loaders
        var trainLoader = CreateSubsetLoader(trainIndices);
        var valLoader = CreateSubsetLoader(valIndices);
        var testLoader = CreateSubsetLoader(testIndices);

        return (trainLoader, valLoader, testLoader);
    }

    /// <summary>
    /// Creates a new loader containing only the data at the specified indices.
    /// </summary>
    private InMemoryDataLoader<T, TInput, TOutput> CreateSubsetLoader(int[] indices)
    {
        var subsetFeatures = ExtractFeatures(indices);
        var subsetLabels = ExtractLabels(indices);
        return new InMemoryDataLoader<T, TInput, TOutput>(subsetFeatures, subsetLabels);
    }

    /// <summary>
    /// Extracts features at the specified indices.
    /// </summary>
    private TInput ExtractFeatures(int[] indices)
    {
        if (_features is Matrix<T> matrix)
        {
            var result = new Matrix<T>(indices.Length, matrix.Columns);
            for (int i = 0; i < indices.Length; i++)
            {
                result.SetRow(i, matrix.GetRow(indices[i]));
            }
            return (TInput)(object)result;
        }

        if (_features is Tensor<T> tensor)
        {
            // Clone shape but change first dimension
            var newShape = (int[])tensor.Shape.Clone();
            newShape[0] = indices.Length;
            var result = new Tensor<T>(newShape);

            for (int i = 0; i < indices.Length; i++)
            {
                CopyTensorSample(tensor, result, indices[i], i);
            }
            return (TInput)(object)result;
        }

        throw new NotSupportedException($"Unsupported input type: {typeof(TInput).Name}");
    }

    /// <summary>
    /// Extracts labels at the specified indices.
    /// </summary>
    private TOutput ExtractLabels(int[] indices)
    {
        if (_labels is Vector<T> vector)
        {
            var result = new Vector<T>(indices.Length);
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = vector[indices[i]];
            }
            return (TOutput)(object)result;
        }

        if (_labels is Tensor<T> tensor)
        {
            // Clone shape but change first dimension
            var newShape = (int[])tensor.Shape.Clone();
            newShape[0] = indices.Length;
            var result = new Tensor<T>(newShape);

            for (int i = 0; i < indices.Length; i++)
            {
                CopyTensorSample(tensor, result, indices[i], i);
            }
            return (TOutput)(object)result;
        }

        throw new NotSupportedException($"Unsupported output type: {typeof(TOutput).Name}");
    }

    /// <summary>
    /// Gets dimensions from the input features.
    /// </summary>
    private static (int SampleCount, int FeatureCount) GetInputDimensions(TInput input)
    {
        if (input is Matrix<T> matrix)
        {
            return (matrix.Rows, matrix.Columns);
        }

        if (input is Tensor<T> tensor)
        {
            int features = tensor.Shape.Length > 1 ? tensor.Shape[1] : 1;
            return (tensor.Shape[0], features);
        }

        throw new NotSupportedException($"Unsupported input type: {typeof(TInput).Name}");
    }

    /// <summary>
    /// Gets dimensions from the output labels.
    /// </summary>
    private static (int SampleCount, int OutputDimension) GetOutputDimensions(TOutput output)
    {
        if (output is Vector<T> vector)
        {
            return (vector.Length, 1);
        }

        if (output is Tensor<T> tensor)
        {
            int outputDim = tensor.Shape.Length > 1 ? tensor.Shape[1] : 1;
            return (tensor.Shape[0], outputDim);
        }

        throw new NotSupportedException($"Unsupported output type: {typeof(TOutput).Name}");
    }

    /// <summary>
    /// Copies a single sample from source tensor to destination tensor.
    /// </summary>
    private static void CopyTensorSample(Tensor<T> source, Tensor<T> dest, int srcIndex, int destIndex)
    {
        if (source.Shape.Length == 1)
        {
            dest[destIndex] = source[srcIndex];
            return;
        }

        // For multi-dimensional tensors, copy the entire sample
        CopyTensorSampleRecursive(source, dest, srcIndex, destIndex, 1, new int[source.Rank]);
    }

    /// <summary>
    /// Recursively copies tensor values across multiple dimensions.
    /// </summary>
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
        }
        else
        {
            for (int i = 0; i < source.Shape[currentDim]; i++)
            {
                indices[currentDim] = i;
                CopyTensorSampleRecursive(source, dest, srcIndex, destIndex, currentDim + 1, indices);
            }
        }
    }
}
