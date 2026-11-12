using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Retrieval module for TabR (Retrieval-Augmented Tabular Learning).
/// </summary>
/// <remarks>
/// <para>
/// The retrieval module finds similar training examples to a query sample
/// and provides their features and labels as additional context for prediction.
/// This is similar to k-NN but integrated into a neural network.
/// </para>
/// <para>
/// <b>For Beginners:</b> The retrieval module works like a smart lookup:
/// 1. For each test sample, find similar training samples
/// 2. Retrieve their features and labels
/// 3. Use this information to help make better predictions
///
/// Think of it as "looking up similar past cases" before making a decision.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Retriever)]
[PipelineStage(PipelineStage.Retrieval)]
public class RetrievalModule<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _embeddingDim;
    private readonly int _numNeighbors;
    private readonly double _temperature;

    // Stored training data
    private Tensor<T>? _trainingKeys;
    private Tensor<T>? _trainingValues;
    private Tensor<T>? _trainingLabels;
    private int _numTrainingSamples;

    /// <summary>
    /// Gets the number of neighbors to retrieve.
    /// </summary>
    public int NumNeighbors => _numNeighbors;

    /// <summary>
    /// Gets whether training data has been stored.
    /// </summary>
    public bool HasTrainingData => _trainingKeys != null;

    /// <summary>
    /// Initializes the retrieval module.
    /// </summary>
    /// <param name="embeddingDim">Dimension of the embedding space.</param>
    /// <param name="numNeighbors">Number of neighbors to retrieve.</param>
    /// <param name="temperature">Temperature for softmax attention over neighbors.</param>
    public RetrievalModule(int embeddingDim, int numNeighbors = 96, double temperature = 1.0)
    {
        _embeddingDim = embeddingDim;
        _numNeighbors = numNeighbors;
        _temperature = temperature;
    }

    /// <summary>
    /// Stores training data for retrieval.
    /// </summary>
    /// <param name="keys">Key embeddings for retrieval [numSamples, embeddingDim].</param>
    /// <param name="values">Value embeddings to retrieve [numSamples, embeddingDim].</param>
    /// <param name="labels">Labels for retrieved samples [numSamples, labelDim].</param>
    public void StoreTrainingData(Tensor<T> keys, Tensor<T> values, Tensor<T> labels)
    {
        _trainingKeys = keys;
        _trainingValues = values;
        _trainingLabels = labels;
        _numTrainingSamples = keys.Shape[0];
    }

    /// <summary>
    /// Retrieves similar samples for a batch of queries.
    /// </summary>
    /// <param name="queryKeys">Query key embeddings [batchSize, embeddingDim].</param>
    /// <returns>Retrieved context containing values, labels, and attention weights.</returns>
    public RetrievalContext<T> Retrieve(Tensor<T> queryKeys)
    {
        if (_trainingKeys == null || _trainingValues == null || _trainingLabels == null)
        {
            throw new InvalidOperationException("Training data must be stored before retrieval");
        }

        var engine = AiDotNetEngine.Current;
        int batchSize = queryKeys.Shape[0];
        int actualNeighbors = Math.Min(_numNeighbors, _numTrainingSamples);
        int labelDim = _trainingLabels.Shape[1];

        // For each query, find k nearest neighbors (stable: equal distances keep training order)
        var distances = ComputeDistances(queryKeys);
        var neighborIndices = TieBreakArgMax.StableSmallestK(distances, actualNeighbors);
        var neighborDistances = engine.TensorTakeAlongDim(distances, neighborIndices, 1);

        // Compute softmax attention weights over exp(-distance / temperature)
        var attentionWeights = ComputeAttentionWeights(neighborDistances);

        // Copy retrieved values and labels
        var flatIndices = neighborIndices.Reshape(batchSize * actualNeighbors);
        var retrievedValues = engine.TensorIndexSelect(
                _trainingValues.Reshape(_numTrainingSamples, _embeddingDim), flatIndices, 0)
            .Reshape(batchSize, actualNeighbors, _embeddingDim);
        var retrievedLabels = engine.TensorIndexSelect(
                _trainingLabels.Reshape(_numTrainingSamples, labelDim), flatIndices, 0)
            .Reshape(batchSize, actualNeighbors, labelDim);

        return new RetrievalContext<T>
        {
            Values = retrievedValues,
            Labels = retrievedLabels,
            AttentionWeights = attentionWeights,
            NumNeighbors = actualNeighbors
        };
    }

    /// <summary>
    /// Squared Euclidean distance from every query to every stored training key: [batchSize, numTrainingSamples].
    /// </summary>
    /// <remarks>
    /// Each query row is differenced against all keys and squared directly (no ||q||^2 - 2 q.k + ||k||^2
    /// expansion, which cancels catastrophically for close points), one query row at a time so the
    /// intermediate stays [numTrainingSamples, embeddingDim].
    /// </remarks>
    private Tensor<T> ComputeDistances(Tensor<T> queryKeys)
    {
        var engine = AiDotNetEngine.Current;
        var trainingKeys = (_trainingKeys ?? throw new InvalidOperationException("_trainingKeys has not been initialized."))
            .Reshape(_numTrainingSamples, _embeddingDim);
        int batchSize = queryKeys.Shape[0];
        var queries = TabularMetrics.Flat(queryKeys, batchSize * _embeddingDim).Reshape(batchSize, _embeddingDim);

        var rows = new Tensor<T>[batchSize];
        for (int b = 0; b < batchSize; b++)
        {
            var diff = engine.TensorSubtract(engine.TensorNarrow(queries, 0, b, 1), trainingKeys);
            rows[b] = engine.ReduceSum(engine.TensorMultiply(diff, diff), new[] { 1 }, false)
                .Reshape(1, _numTrainingSamples);
        }

        return batchSize == 0
            ? new Tensor<T>(new[] { 0, _numTrainingSamples })
            : engine.TensorConcatenate(rows, 0);
    }

    /// <summary>
    /// Softmax of -distance / temperature over each row of neighbour distances.
    /// </summary>
    /// <remarks>
    /// The softmax subtracts each row's largest logit, i.e. its SMALLEST distance. Shifting by the largest
    /// distance instead (as this method once did) gives exp((d_max - d) / temperature) &gt;= 1, which
    /// overflows to infinity for widely spread distances and turns the weights into NaN.
    /// </remarks>
    private Tensor<T> ComputeAttentionWeights(Tensor<T> distances)
    {
        var engine = AiDotNetEngine.Current;
        var tempScale = NumOps.FromDouble(-1.0 / _temperature);
        return engine.Softmax(engine.TensorMultiplyScalar(distances, tempScale), -1);
    }

    /// <summary>
    /// Clears stored training data.
    /// </summary>
    public void ClearTrainingData()
    {
        _trainingKeys = null;
        _trainingValues = null;
        _trainingLabels = null;
        _numTrainingSamples = 0;
    }
}

/// <summary>
/// Context returned by the retrieval module.
/// </summary>
public class RetrievalContext<T>
{
    /// <summary>
    /// Retrieved value embeddings [batchSize, numNeighbors, embeddingDim].
    /// </summary>
    public required Tensor<T> Values { get; init; }

    /// <summary>
    /// Retrieved labels [batchSize, numNeighbors, labelDim].
    /// </summary>
    public required Tensor<T> Labels { get; init; }

    /// <summary>
    /// Attention weights over neighbors [batchSize, numNeighbors].
    /// </summary>
    public required Tensor<T> AttentionWeights { get; init; }

    /// <summary>
    /// Number of neighbors retrieved.
    /// </summary>
    public int NumNeighbors { get; init; }
}
