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

        int batchSize = queryKeys.Shape[0];
        int actualNeighbors = Math.Min(_numNeighbors, _numTrainingSamples);
        int labelDim = _trainingLabels.Shape[1];

        var neighborIndices = new int[batchSize, actualNeighbors];
        var neighborDistances = new T[batchSize, actualNeighbors];
        var attentionWeights = new Tensor<T>([batchSize, actualNeighbors]);
        var retrievedValues = new Tensor<T>([batchSize, actualNeighbors, _embeddingDim]);
        var retrievedLabels = new Tensor<T>([batchSize, actualNeighbors, labelDim]);

        // For each query, find k nearest neighbors
        for (int b = 0; b < batchSize; b++)
        {
            var distances = ComputeDistances(queryKeys, b);
            var topK = GetTopKIndices(distances, actualNeighbors);

            for (int k = 0; k < actualNeighbors; k++)
            {
                neighborIndices[b, k] = topK[k].index;
                neighborDistances[b, k] = topK[k].distance;
            }

            // Compute softmax attention weights
            var weights = ComputeAttentionWeights(neighborDistances, b, actualNeighbors);
            for (int k = 0; k < actualNeighbors; k++)
            {
                attentionWeights[b * actualNeighbors + k] = weights[k];

                // Copy retrieved values
                int trainIdx = neighborIndices[b, k];
                for (int d = 0; d < _embeddingDim; d++)
                {
                    retrievedValues[b * actualNeighbors * _embeddingDim + k * _embeddingDim + d] =
                        _trainingValues[trainIdx * _embeddingDim + d];
                }

                // Copy retrieved labels
                for (int l = 0; l < labelDim; l++)
                {
                    retrievedLabels[b * actualNeighbors * labelDim + k * labelDim + l] =
                        _trainingLabels[trainIdx * labelDim + l];
                }
            }
        }

        return new RetrievalContext<T>
        {
            Values = retrievedValues,
            Labels = retrievedLabels,
            AttentionWeights = attentionWeights,
            NumNeighbors = actualNeighbors
        };
    }

    private T[] ComputeDistances(Tensor<T> queryKeys, int batchIdx)
    {
        var distances = new T[_numTrainingSamples];

        for (int t = 0; t < _numTrainingSamples; t++)
        {
            var dist = NumOps.Zero;
            for (int d = 0; d < _embeddingDim; d++)
            {
                var diff = NumOps.Subtract(
                    queryKeys[batchIdx * _embeddingDim + d],
                    _trainingKeys![t * _embeddingDim + d]);
                dist = NumOps.Add(dist, NumOps.Multiply(diff, diff));
            }
            distances[t] = dist;
        }

        return distances;
    }

    private (int index, T distance)[] GetTopKIndices(T[] distances, int k)
    {
        var indexed = distances.Select((d, i) => (index: i, distance: d)).ToArray();
        Array.Sort(indexed, (a, b) => NumOps.Compare(a.distance, b.distance));
        return indexed.Take(k).ToArray();
    }

    private T[] ComputeAttentionWeights(T[,] distances, int batchIdx, int numNeighbors)
    {
        var weights = new T[numNeighbors];
        var tempScale = NumOps.FromDouble(-1.0 / _temperature);

        // Compute exp(-distance / temperature)
        var maxVal = distances[batchIdx, 0];
        for (int k = 1; k < numNeighbors; k++)
        {
            if (NumOps.Compare(distances[batchIdx, k], maxVal) > 0)
                maxVal = distances[batchIdx, k];
        }

        var sumExp = NumOps.Zero;
        for (int k = 0; k < numNeighbors; k++)
        {
            var scaledDist = NumOps.Multiply(
                NumOps.Subtract(distances[batchIdx, k], maxVal),
                tempScale);
            weights[k] = NumOps.Exp(scaledDist);
            sumExp = NumOps.Add(sumExp, weights[k]);
        }

        // Normalize
        for (int k = 0; k < numNeighbors; k++)
        {
            weights[k] = NumOps.Divide(weights[k], sumExp);
        }

        return weights;
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
