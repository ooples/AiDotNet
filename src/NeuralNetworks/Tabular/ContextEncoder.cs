using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Context encoder for TabR that processes retrieved neighbors into a context representation.
/// </summary>
/// <remarks>
/// <para>
/// The context encoder takes retrieved neighbors (values and labels) and combines them
/// using attention-weighted aggregation and cross-attention with the query.
/// </para>
/// <para>
/// <b>For Beginners:</b> The context encoder does three things:
/// 1. Takes the similar samples found by retrieval
/// 2. Weights them by how similar they are (attention)
/// 3. Creates a single "context" representation that summarizes the neighbors
///
/// This context is then combined with the query to make the final prediction.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ContextEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;

    // Cross-attention layers
    private readonly FullyConnectedLayer<T> _queryProjection;
    private readonly FullyConnectedLayer<T> _keyProjection;
    private readonly FullyConnectedLayer<T> _valueProjection;
    private readonly FullyConnectedLayer<T> _outputProjection;

    // Label embedding
    private readonly FullyConnectedLayer<T> _labelEmbedding;

    // Cached values
    private Tensor<T>? _queryCache;
    private Tensor<T>? _keyCache;
    private Tensor<T>? _valueCache;

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public int ParameterCount =>
        _queryProjection.ParameterCount +
        _keyProjection.ParameterCount +
        _valueProjection.ParameterCount +
        _outputProjection.ParameterCount +
        _labelEmbedding.ParameterCount;

    /// <summary>
    /// Initializes the context encoder.
    /// </summary>
    /// <param name="embeddingDim">Embedding dimension.</param>
    /// <param name="labelDim">Label dimension.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    public ContextEncoder(int embeddingDim, int labelDim, int numHeads = 4)
    {
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _headDim = embeddingDim / numHeads;

        // Cross-attention projections
        _queryProjection = new FullyConnectedLayer<T>(
            embeddingDim, embeddingDim, (IActivationFunction<T>?)null);
        _keyProjection = new FullyConnectedLayer<T>(
            embeddingDim, embeddingDim, (IActivationFunction<T>?)null);
        _valueProjection = new FullyConnectedLayer<T>(
            embeddingDim, embeddingDim, (IActivationFunction<T>?)null);
        _outputProjection = new FullyConnectedLayer<T>(
            embeddingDim, embeddingDim, (IActivationFunction<T>?)null);

        // Label embedding to same dimension as values
        _labelEmbedding = new FullyConnectedLayer<T>(
            labelDim, embeddingDim, new ReLUActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Encodes the retrieved context using cross-attention.
    /// </summary>
    /// <param name="queryEmbedding">Query embedding [batchSize, embeddingDim].</param>
    /// <param name="context">Retrieved context from RetrievalModule.</param>
    /// <returns>Context-enhanced representation [batchSize, embeddingDim].</returns>
    public Tensor<T> Forward(Tensor<T> queryEmbedding, RetrievalContext<T> context)
    {
        int batchSize = queryEmbedding.Shape[0];
        int numNeighbors = context.NumNeighbors;

        // Embed labels and combine with values
        var labelEmbeddings = EmbedLabels(context.Labels, batchSize, numNeighbors);
        var combinedContext = CombineValuesAndLabels(context.Values, labelEmbeddings, batchSize, numNeighbors);

        // Project query, keys, values
        var queries = _queryProjection.Forward(queryEmbedding);
        _queryCache = queries;

        // For keys and values, we need to flatten and project
        var flatContext = FlattenContext(combinedContext, batchSize, numNeighbors);
        var keys = ProjectContext(flatContext, _keyProjection, batchSize, numNeighbors);
        var values = ProjectContext(flatContext, _valueProjection, batchSize, numNeighbors);
        _keyCache = keys;
        _valueCache = values;

        // Cross-attention: query attends to retrieved context
        var attended = CrossAttention(queries, keys, values, context.AttentionWeights, batchSize, numNeighbors);

        // Final projection
        var output = _outputProjection.Forward(attended);

        // Residual connection
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = NumOps.Add(output[i], queryEmbedding[i]);
        }

        return output;
    }

    private Tensor<T> EmbedLabels(Tensor<T> labels, int batchSize, int numNeighbors)
    {
        int labelDim = labels.Shape[2];
        var flatLabels = new Tensor<T>([batchSize * numNeighbors, labelDim]);

        for (int i = 0; i < flatLabels.Length; i++)
        {
            flatLabels[i] = labels[i];
        }

        var embedded = _labelEmbedding.Forward(flatLabels);
        return embedded;
    }

    private Tensor<T> CombineValuesAndLabels(Tensor<T> values, Tensor<T> labelEmbeddings, int batchSize, int numNeighbors)
    {
        var combined = new Tensor<T>([batchSize, numNeighbors, _embeddingDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNeighbors; n++)
            {
                for (int d = 0; d < _embeddingDim; d++)
                {
                    int valueIdx = b * numNeighbors * _embeddingDim + n * _embeddingDim + d;
                    int labelIdx = (b * numNeighbors + n) * _embeddingDim + d;
                    combined[valueIdx] = NumOps.Add(values[valueIdx], labelEmbeddings[labelIdx]);
                }
            }
        }

        return combined;
    }

    private Tensor<T> FlattenContext(Tensor<T> context, int batchSize, int numNeighbors)
    {
        var flat = new Tensor<T>([batchSize * numNeighbors, _embeddingDim]);
        for (int i = 0; i < flat.Length; i++)
        {
            flat[i] = context[i];
        }
        return flat;
    }

    private Tensor<T> ProjectContext(Tensor<T> flatContext, FullyConnectedLayer<T> layer, int batchSize, int numNeighbors)
    {
        var projected = layer.Forward(flatContext);
        var reshaped = new Tensor<T>([batchSize, numNeighbors, _embeddingDim]);
        for (int i = 0; i < projected.Length; i++)
        {
            reshaped[i] = projected[i];
        }
        return reshaped;
    }

    private Tensor<T> CrossAttention(Tensor<T> queries, Tensor<T> keys, Tensor<T> values,
        Tensor<T> priorWeights, int batchSize, int numNeighbors)
    {
        var output = new Tensor<T>([batchSize, _embeddingDim]);
        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDim));

        for (int b = 0; b < batchSize; b++)
        {
            // Compute attention scores
            var scores = new T[numNeighbors];
            for (int n = 0; n < numNeighbors; n++)
            {
                var dot = NumOps.Zero;
                for (int d = 0; d < _embeddingDim; d++)
                {
                    dot = NumOps.Add(dot, NumOps.Multiply(
                        queries[b * _embeddingDim + d],
                        keys[b * numNeighbors * _embeddingDim + n * _embeddingDim + d]));
                }
                scores[n] = NumOps.Multiply(dot, scale);
            }

            // Softmax
            var maxScore = scores[0];
            for (int n = 1; n < numNeighbors; n++)
            {
                if (NumOps.Compare(scores[n], maxScore) > 0)
                    maxScore = scores[n];
            }

            var sumExp = NumOps.Zero;
            for (int n = 0; n < numNeighbors; n++)
            {
                scores[n] = NumOps.Exp(NumOps.Subtract(scores[n], maxScore));
                sumExp = NumOps.Add(sumExp, scores[n]);
            }

            for (int n = 0; n < numNeighbors; n++)
            {
                scores[n] = NumOps.Divide(scores[n], sumExp);
            }

            // Weighted sum of values
            for (int d = 0; d < _embeddingDim; d++)
            {
                var sum = NumOps.Zero;
                for (int n = 0; n < numNeighbors; n++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(
                        scores[n],
                        values[b * numNeighbors * _embeddingDim + n * _embeddingDim + d]));
                }
                output[b * _embeddingDim + d] = sum;
            }
        }

        return output;
    }

    /// <summary>
    /// Backward pass through the context encoder.
    /// </summary>
    public Tensor<T> Backward(Tensor<T> gradient)
    {
        // Simplified backward
        var grad = _outputProjection.Backward(gradient);
        return _queryProjection.Backward(grad);
    }

    /// <summary>
    /// Updates parameters.
    /// </summary>
    public void UpdateParameters(T learningRate)
    {
        _queryProjection.UpdateParameters(learningRate);
        _keyProjection.UpdateParameters(learningRate);
        _valueProjection.UpdateParameters(learningRate);
        _outputProjection.UpdateParameters(learningRate);
        _labelEmbedding.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public void ResetState()
    {
        _queryCache = null;
        _keyCache = null;
        _valueCache = null;

        _queryProjection.ResetState();
        _keyProjection.ResetState();
        _valueProjection.ResetState();
        _outputProjection.ResetState();
        _labelEmbedding.ResetState();
    }
}
