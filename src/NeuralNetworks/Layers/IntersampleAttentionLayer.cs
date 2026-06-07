using System;
using System.Collections.Generic;
using AiDotNet.Autodiff;
using AiDotNet.Attributes;
using AiDotNet.Helpers;

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
    public override long ParameterCount =>
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
        _queryProjection = new FullyConnectedLayer<T>(embeddingDim, (Interfaces.IActivationFunction<T>?)null);
        _keyProjection = new FullyConnectedLayer<T>(embeddingDim, (Interfaces.IActivationFunction<T>?)null);
        _valueProjection = new FullyConnectedLayer<T>(embeddingDim, (Interfaces.IActivationFunction<T>?)null);
        _outputProjection = new FullyConnectedLayer<T>(embeddingDim, (Interfaces.IActivationFunction<T>?)null);

        // Initialize layer normalization parameters
        _layerNormGamma = Tensor<T>.CreateDefault([embeddingDim], NumOps.One);
        _layerNormBeta = new Tensor<T>([embeddingDim]);

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_layerNormGamma, PersistentTensorRole.NormalizationParams);
        RegisterTrainableParameter(_layerNormBeta, PersistentTensorRole.NormalizationParams);

        // Eagerly resolve the four FullyConnectedLayer projections (embed -> embed) with a probe
        // forward. They resolve their input dim lazily on first Forward, but ParameterCount /
        // GetParameters / SetParameters and the Clone serialize→deserialize round-trip need their
        // weights materialized up front (a deserialized layer is asked for parameters before any
        // real forward pass). The probe input is discarded.
        var probe = new Tensor<T>([1, 1, embeddingDim]);
        Forward(probe);
        _inputCache = null;
        _normalizedCache = null;
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

        // Project Q, K, V on every (sample, feature) token. The FC projections map
        // embed -> embed over the last axis, so flatten the (sample, feature) axes to
        // apply them per token, then restore the [batch, feature, embed] shape. All
        // Engine ops, so the tape carries gradients back to the projection weights
        // (the previous per-feature manual slice copy was a tape dead-end — gradients
        // never reached _queryProjection / _keyProjection / _valueProjection).
        var flat = Engine.Reshape(input, new[] { batchSize * numFeatures, embDim });
        var q = Engine.Reshape(_queryProjection.Forward(flat), new[] { batchSize, numFeatures, embDim });
        var k = Engine.Reshape(_keyProjection.Forward(flat), new[] { batchSize, numFeatures, embDim });
        var v = Engine.Reshape(_valueProjection.Forward(flat), new[] { batchSize, numFeatures, embDim });

        // Intersample attention attends ACROSS samples for each feature independently
        // (SAINT's row attention, Somepalli et al. 2021). Arrange as
        // [feature, head, sample, headDim] so the scaled-dot-product primitive treats
        // `feature` as the batch of independent attention problems, `sample` as the
        // sequence, and runs multi-head attention across the samples.
        var qh = Engine.TensorPermute(Engine.Reshape(q, new[] { batchSize, numFeatures, _numHeads, _headDim }), new[] { 1, 2, 0, 3 });
        var kh = Engine.TensorPermute(Engine.Reshape(k, new[] { batchSize, numFeatures, _numHeads, _headDim }), new[] { 1, 2, 0, 3 });
        var vh = Engine.TensorPermute(Engine.Reshape(v, new[] { batchSize, numFeatures, _numHeads, _headDim }), new[] { 1, 2, 0, 3 });

        var context = Engine.ScaledDotProductAttention(
            qh, kh, vh, mask: null, scale: 1.0 / Math.Sqrt(_headDim), out _);

        // context is [feature, head, sample, headDim]; restore [batch(sample), feature, embed].
        var merged = Engine.Reshape(
            Engine.TensorPermute(context, new[] { 2, 0, 1, 3 }),
            new[] { batchSize * numFeatures, embDim });
        var projected = Engine.Reshape(_outputProjection.Forward(merged), new[] { batchSize, numFeatures, embDim });

        // Residual connection and layer normalization over the embedding axis.
        var output = AddResidualAndNormalize(input, projected, batchSize, numFeatures, embDim);
        _normalizedCache = output;

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

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int qCount = ParameterCountHelper.ToFlatVectorSize(_queryProjection.ParameterCount);
        int kCount = ParameterCountHelper.ToFlatVectorSize(_keyProjection.ParameterCount);
        int vCount = ParameterCountHelper.ToFlatVectorSize(_valueProjection.ParameterCount);
        int oCount = ParameterCountHelper.ToFlatVectorSize(_outputProjection.ParameterCount);
        int expected = qCount + kCount + vCount + oCount + _embeddingDim * 2;
        if (parameters.Length != expected)
        {
            throw new ArgumentException(
                $"Expected {expected} parameters, got {parameters.Length}.", nameof(parameters));
        }

        int offset = 0;
        _queryProjection.SetParameters(parameters.SubVector(offset, qCount)); offset += qCount;
        _keyProjection.SetParameters(parameters.SubVector(offset, kCount)); offset += kCount;
        _valueProjection.SetParameters(parameters.SubVector(offset, vCount)); offset += vCount;
        _outputProjection.SetParameters(parameters.SubVector(offset, oCount)); offset += oCount;
        for (int i = 0; i < _embeddingDim; i++) _layerNormGamma[i] = parameters[offset++];
        for (int i = 0; i < _embeddingDim; i++) _layerNormBeta[i] = parameters[offset++];
    }

    /// <summary>
    /// Persists the head count and dropout rate so the Clone serialize→deserialize round-trip
    /// can reconstruct the layer (embedding dim is recoverable from the saved shape).
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var m = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        m["NumHeads"] = _numHeads.ToString(inv);
        m["DropoutRate"] = _dropoutRate.ToString(inv);
        return m;
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
