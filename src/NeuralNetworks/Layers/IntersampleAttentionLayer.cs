using AiDotNet.Autodiff;
using AiDotNet.Attributes;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Intersample (Row) Attention for SAINT architecture.
/// </summary>
/// <remarks>
/// <para>
/// Intersample attention allows samples in a batch to attend to each other,
/// enabling the model to learn relationships between different data points.
/// This is a key innovation in SAINT that helps with semi-supervised learning.
/// </para>
/// <para>
/// <b>For Beginners:</b> While column attention looks at relationships between features
/// within a single sample, intersample attention looks at relationships between different
/// samples in the batch:
/// - Column attention: "How does age relate to income for this person?"
/// - Intersample attention: "How does this person compare to others in the batch?"
///
/// This allows the model to learn from the distribution of data, not just individual samples.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public partial class IntersampleAttentionLayer<T> : LayerBase<T>
{
    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly double _dropoutRate;

    // Attention projections
    private readonly FullyConnectedLayer<T> _queryProjection;
    private readonly FullyConnectedLayer<T> _keyProjection;
    private readonly FullyConnectedLayer<T> _valueProjection;
    private readonly FullyConnectedLayer<T> _outputProjection;

    // Layer normalization parameters
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]

    private Tensor<T> _layerNormGamma;
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]

    private Tensor<T> _layerNormBeta;

    // Cached values for backward pass
    private Tensor<T>? _inputCache;
    private Tensor<T>? _normalizedCache;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override int ParameterCount =>
        _queryProjection.ParameterCount +
        _keyProjection.ParameterCount +
        _valueProjection.ParameterCount +
        _outputProjection.ParameterCount +
        _embeddingDim * 2; // LayerNorm gamma and beta

    /// <summary>
    /// Initializes intersample attention.
    /// </summary>
    /// <param name="embeddingDim">Embedding dimension.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="dropoutRate">Dropout rate for attention.</param>
    public IntersampleAttentionLayer(int embeddingDim, int numHeads = 8, double dropoutRate = 0.1)
        : base([embeddingDim], [embeddingDim])
    {
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _headDim = embeddingDim / numHeads;
        _dropoutRate = dropoutRate;

        // Initialize attention projections
        _queryProjection = new FullyConnectedLayer<T>(
            embeddingDim, embeddingDim, (Interfaces.IActivationFunction<T>?)null);
        _keyProjection = new FullyConnectedLayer<T>(
            embeddingDim, embeddingDim, (Interfaces.IActivationFunction<T>?)null);
        _valueProjection = new FullyConnectedLayer<T>(
            embeddingDim, embeddingDim, (Interfaces.IActivationFunction<T>?)null);
        _outputProjection = new FullyConnectedLayer<T>(
            embeddingDim, embeddingDim, (Interfaces.IActivationFunction<T>?)null);

        // Initialize layer normalization parameters
        _layerNormGamma = Tensor<T>.CreateDefault([embeddingDim], NumOps.One);
        _layerNormBeta = new Tensor<T>([embeddingDim]);

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_layerNormGamma, PersistentTensorRole.NormalizationParams);
        RegisterTrainableParameter(_layerNormBeta, PersistentTensorRole.NormalizationParams);

    }

    /// <summary>
    /// Forward pass through intersample attention.
    /// </summary>
    /// <param name="input">Input tensor [batchSize, numFeatures, embeddingDim].</param>
    /// <returns>Output with intersample attention applied [batchSize, numFeatures, embeddingDim].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _inputCache = input;

        int batchSize = input.Shape[0];
        int numFeatures = input.Shape[1];
        int embDim = input.Shape[2];

        // Transpose to [numFeatures, batchSize, embeddingDim] for intersample attention
        var transposed = Engine.TensorPermute(input, new[] { 1, 0, 2 });

        // Apply multi-head self-attention across samples
        var attended = ApplyMultiHeadAttention(transposed, numFeatures, batchSize, embDim);

        // Transpose back to [batchSize, numFeatures, embeddingDim]
        var output = Engine.TensorPermute(attended, new[] { 1, 0, 2 });

        // Residual connection and layer normalization
        output = AddResidualAndNormalize(input, output, batchSize, numFeatures, embDim);
        _normalizedCache = output;

        return output;
    }

    private Tensor<T> ApplyMultiHeadAttention(Tensor<T> input, int numFeatures, int batchSize, int embDim)
    {
        var output = new Tensor<T>([numFeatures, batchSize, embDim]);
        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDim));

        // For each feature position, apply attention across samples
        for (int f = 0; f < numFeatures; f++)
        {
            // Extract feature slice [batchSize, embDim]
            var featureSlice = TensorAllocator.Rent<T>([batchSize, embDim]);
            int fOffset = f * batchSize * embDim;
            for (int i = 0; i < batchSize * embDim; i++)
            {
                featureSlice[i] = input[fOffset + i];
            }

            // Project to Q, K, V
            var queries = _queryProjection.Forward(featureSlice);
            var keys = _keyProjection.Forward(featureSlice);
            var values = _valueProjection.Forward(featureSlice);

            // Compute attention: Q * K^T -> [batchSize, batchSize]
            var kT = keys.Transpose(new[] { 1, 0 });
            var scores = Engine.TensorMatMul(queries, kT);
            scores = Engine.TensorMultiplyScalar(scores, scale);

            // Softmax over samples
            scores = Engine.Softmax(scores);

            // Apply attention to values: scores * V -> [batchSize, embDim]
            var attended = Engine.TensorMatMul(scores, values);

            // Store result
            int outOffset = f * batchSize * embDim;
            for (int i = 0; i < batchSize * embDim; i++)
            {
                output[outOffset + i] = attended[i];
            }
        }

        return output;
    }

    private Tensor<T> AddResidualAndNormalize(Tensor<T> residual, Tensor<T> output,
        int batchSize, int numFeatures, int embDim)
    {
        // Residual connection
        var result = Engine.TensorAdd(residual, output);

        // Layer normalization across the embedding dimension (last axis), one
        // normalization per (batch, feature) position. The previous manual loop
        // dispatched 3 × batchSize × numFeatures × embDim virtual NumOps calls
        // which JIT can't vectorize through IInterface<T>; Engine.LayerNorm runs
        // it as one fused op with the same semantics.
        const double epsilon = 1e-5;
        return Engine.LayerNorm(result, _layerNormGamma, _layerNormBeta, epsilon, out _, out _);
    }

    /// <summary>
    /// Updates parameters.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        _queryProjection.UpdateParameters(learningRate);
        _keyProjection.UpdateParameters(learningRate);
        _valueProjection.UpdateParameters(learningRate);
        _outputProjection.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var qParams = _queryProjection.GetParameters();
        var kParams = _keyProjection.GetParameters();
        var vParams = _valueProjection.GetParameters();
        var oParams = _outputProjection.GetParameters();

        int total = qParams.Length + kParams.Length + vParams.Length + oParams.Length + _embeddingDim * 2;
        var result = new Vector<T>(total);
        int offset = 0;

        CopyVectorToVector(qParams, result, ref offset);
        CopyVectorToVector(kParams, result, ref offset);
        CopyVectorToVector(vParams, result, ref offset);
        CopyVectorToVector(oParams, result, ref offset);

        for (int i = 0; i < _embeddingDim; i++)
            result[offset++] = _layerNormGamma[i];
        for (int i = 0; i < _embeddingDim; i++)
            result[offset++] = _layerNormBeta[i];

        return result;
    }

    private static void CopyVectorToVector(Vector<T> source, Vector<T> target, ref int offset)
    {
        for (int i = 0; i < source.Length; i++)
        {
            target[offset++] = source[i];
        }
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public override void ResetState()
    {
        _inputCache = null;
        _normalizedCache = null;

        _queryProjection.ResetState();
        _keyProjection.ResetState();
        _valueProjection.ResetState();
        _outputProjection.ResetState();
    }
}
