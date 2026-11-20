using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Training)]
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

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public long ParameterCount =>
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
        _queryProjection = new FullyConnectedLayer<T>(embeddingDim, (IActivationFunction<T>?)null);
        _keyProjection = new FullyConnectedLayer<T>(embeddingDim, (IActivationFunction<T>?)null);
        _valueProjection = new FullyConnectedLayer<T>(embeddingDim, (IActivationFunction<T>?)null);
        _outputProjection = new FullyConnectedLayer<T>(embeddingDim, (IActivationFunction<T>?)null);

        // Label embedding to same dimension as values
        _labelEmbedding = new FullyConnectedLayer<T>(embeddingDim, new ReLUActivation<T>() as IActivationFunction<T>);
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

        // For keys and values, we need to flatten and project
        var flatContext = FlattenContext(combinedContext, batchSize, numNeighbors);
        var keys = ProjectContext(flatContext, _keyProjection, batchSize, numNeighbors);
        var values = ProjectContext(flatContext, _valueProjection, batchSize, numNeighbors);

        // Cross-attention: query attends to retrieved context
        var attended = CrossAttention(queries, keys, values, context.AttentionWeights, batchSize, numNeighbors);

        // Final projection
        var output = _outputProjection.Forward(attended);

        // Residual connection (a new tensor, so the projection's output is not mutated in place).
        return AiDotNetEngine.Current.TensorAdd(output, AiDotNetEngine.Current.Reshape(queryEmbedding, output._shape));
    }

    private Tensor<T> EmbedLabels(Tensor<T> labels, int batchSize, int numNeighbors)
    {
        int labelDim = labels.Shape[2];
        var flatLabels = AiDotNetEngine.Current.Reshape(labels, new[] { batchSize * numNeighbors, labelDim });

        var embedded = _labelEmbedding.Forward(flatLabels);
        return embedded;
    }

    private Tensor<T> CombineValuesAndLabels(Tensor<T> values, Tensor<T> labelEmbeddings, int batchSize, int numNeighbors)
    {
        var engine = AiDotNetEngine.Current;
        var shape = new[] { batchSize, numNeighbors, _embeddingDim };
        return engine.TensorAdd(engine.Reshape(values, shape), engine.Reshape(labelEmbeddings, shape));
    }

    private Tensor<T> FlattenContext(Tensor<T> context, int batchSize, int numNeighbors)
    {
        return AiDotNetEngine.Current.Reshape(context, new[] { batchSize * numNeighbors, _embeddingDim });
    }

    private Tensor<T> ProjectContext(Tensor<T> flatContext, FullyConnectedLayer<T> layer, int batchSize, int numNeighbors)
    {
        var projected = layer.Forward(flatContext);
        return AiDotNetEngine.Current.Reshape(projected, new[] { batchSize, numNeighbors, _embeddingDim });
    }

    private Tensor<T> CrossAttention(Tensor<T> queries, Tensor<T> keys, Tensor<T> values,
        Tensor<T> priorWeights, int batchSize, int numNeighbors)
    {
        var engine = AiDotNetEngine.Current;
        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDim));
        var contextShape = new[] { batchSize, numNeighbors, _embeddingDim };

        // scores[b, n] = (q[b] . k[b, n]) * scale, softmax over the neighbours.
        var query = engine.TensorBroadcastTo(engine.Reshape(queries, new[] { batchSize, 1, _embeddingDim }), contextShape);
        var dots = engine.ReduceSum(engine.TensorMultiply(query, engine.Reshape(keys, contextShape)), new[] { 2 }, false);
        var weights = engine.Softmax(engine.TensorMultiplyScalar(dots, scale), 1);

        // Weighted sum of values: out[b, :] = sum_n weights[b, n] * v[b, n, :]
        var weightGrid = engine.TensorBroadcastTo(engine.Reshape(weights, new[] { batchSize, numNeighbors, 1 }), contextShape);
        return engine.ReduceSum(engine.TensorMultiply(weightGrid, engine.Reshape(values, contextShape)), new[] { 1 }, false);
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
        _queryProjection.ResetState();
        _keyProjection.ResetState();
        _valueProjection.ResetState();
        _outputProjection.ResetState();
        _labelEmbedding.ResetState();
    }
}
