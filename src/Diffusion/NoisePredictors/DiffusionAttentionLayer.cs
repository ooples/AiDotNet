using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Lazy multi-head attention with distinct query and context dimensions.
/// </summary>
/// <remarks>
/// This is the projection contract required by diffusion cross attention:
/// Q maps from the U-Net channel width while K/V map from the text-encoder
/// width. All math uses engine operations, including fused scaled-dot-product
/// attention, so the same graph is tape-connected on CPU and every GPU backend.
/// </remarks>
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public sealed partial class DiffusionAttentionLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 3
        ?
        [
            new(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new(TensorAxis.Features, AxisRelation.Fixed(_queryDimension))
        ]
        : null;

    private readonly int _queryDimension;
    private readonly int _contextDimension;
    private readonly int _headCount;
    private readonly int _headDimension;
    private readonly bool _zeroOutputProjection;
    private readonly AttentionBiasInitialization _outputBiasInitialization;

    [TrainableParameter(Role = PersistentTensorRole.Weights,
        Shape = "_queryDimension, _queryDimension")]
    private Tensor<T> _queryWeights = new([0, 0]);
    [TrainableParameter(Role = PersistentTensorRole.Weights,
        Shape = "_contextDimension, _queryDimension")]
    private Tensor<T> _keyWeights = new([0, 0]);
    [TrainableParameter(Role = PersistentTensorRole.Weights,
        Shape = "_contextDimension, _queryDimension")]
    private Tensor<T> _valueWeights = new([0, 0]);
    [TrainableParameter(Role = PersistentTensorRole.Weights,
        Shape = "_queryDimension, _queryDimension")]
    private Tensor<T> _outputWeights = new([0, 0]);
    [TrainableParameter(Role = PersistentTensorRole.Biases, Shape = "_queryDimension")]
    private Tensor<T> _outputBias = new([0]);

    /// <summary>Gets the query/output feature width.</summary>
    public int QueryDimension => _queryDimension;

    /// <summary>Gets the key/value context feature width.</summary>
    public int ContextDimension => _contextDimension;

    /// <summary>Gets the number of attention heads.</summary>
    public int HeadCount => _headCount;

    /// <summary>Gets whether the output projection starts at zero.</summary>
    public bool IsOutputProjectionZeroInitialized => _zeroOutputProjection;

    /// <summary>Gets how the output projection's bias is initialized.</summary>
    public AttentionBiasInitialization OutputBiasInitialization => _outputBiasInitialization;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates self- or cross-attention without eager weight allocation.</summary>
    /// <param name="queryDimension">The query and output width.</param>
    /// <param name="contextDimension">The key/value context width.</param>
    /// <param name="headCount">The number of heads; must divide <paramref name="queryDimension"/>.</param>
    /// <param name="zeroOutputProjection">Whether the output projection's weights (and bias) start at zero.</param>
    /// <param name="outputBiasInitialization">
    /// How the output projection's bias starts when <paramref name="zeroOutputProjection"/> is false:
    /// <see cref="AttentionBiasInitialization.Zero"/> (the default, unchanged for existing callers) or
    /// <see cref="AttentionBiasInitialization.TorchDefault"/>, PyTorch's <c>nn.Linear</c> uniform in
    /// <c>±1/sqrt(queryDimension)</c> as the reference <c>to_out</c> projections start.
    /// </param>
    public DiffusionAttentionLayer(
        int queryDimension,
        int contextDimension,
        int headCount,
        bool zeroOutputProjection = false,
        AttentionBiasInitialization outputBiasInitialization = AttentionBiasInitialization.Zero)
        : base([-1, queryDimension], [-1, queryDimension])
    {
        if (queryDimension <= 0) throw new ArgumentOutOfRangeException(nameof(queryDimension));
        if (contextDimension <= 0) throw new ArgumentOutOfRangeException(nameof(contextDimension));
        if (headCount <= 0 || queryDimension % headCount != 0)
            throw new ArgumentException("Head count must evenly divide the query dimension.", nameof(headCount));
        _queryDimension = queryDimension;
        _contextDimension = contextDimension;
        _headCount = headCount;
        _headDimension = queryDimension / headCount;
        _zeroOutputProjection = zeroOutputProjection;
        _outputBiasInitialization = outputBiasInitialization;
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (_queryDimension != _contextDimension)
            throw new ArgumentException(
                "Cross attention requires an explicit context tensor.", nameof(input));
        return Forward(input, input);
    }

    /// <summary>Runs attention with query [B,S,Q] and context [B,N,C].</summary>
    public Tensor<T> Forward(Tensor<T> query, Tensor<T> context)
    {
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (context is null) throw new ArgumentNullException(nameof(context));
        if (query.Rank != 3 || context.Rank != 3)
            throw new ArgumentException("Attention requires query/context tensors of rank 3.");
        if (query.Shape[0] != context.Shape[0])
            throw new ArgumentException("Query and context batch sizes must match.", nameof(context));
        if (query.Shape[2] != _queryDimension)
            throw new ArgumentException(
                $"Expected query width {_queryDimension}, got {query.Shape[2]}.", nameof(query));
        if (context.Shape[2] != _contextDimension)
            throw new ArgumentException(
                $"Expected context width {_contextDimension}, got {context.Shape[2]}.", nameof(context));
        MaterializeParameters();

        int batch = query.Shape[0];
        int queryLength = query.Shape[1];
        int contextLength = context.Shape[1];
        var q = Engine.TensorMatMul(query, _queryWeights);
        var k = Engine.TensorMatMul(context, _keyWeights);
        var v = Engine.TensorMatMul(context, _valueWeights);
        q = Engine.TensorPermute(
            Engine.Reshape(q, [batch, queryLength, _headCount, _headDimension]),
            [0, 2, 1, 3]);
        k = Engine.TensorPermute(
            Engine.Reshape(k, [batch, contextLength, _headCount, _headDimension]),
            [0, 2, 1, 3]);
        v = Engine.TensorPermute(
            Engine.Reshape(v, [batch, contextLength, _headCount, _headDimension]),
            [0, 2, 1, 3]);
        var attended = Engine.ScaledDotProductAttention(
            q, k, v, mask: null, scale: null, out _);
        attended = Engine.TensorPermute(attended, [0, 2, 1, 3]).Contiguous();
        attended = Engine.Reshape(attended, [batch, queryLength, _queryDimension]);
        var projected = Engine.TensorMatMul(attended, _outputWeights);
        return Engine.TensorAdd(projected, _outputBias);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        if (gradients.Length != ParameterCount || _queryWeights.Length == 0) return;
        var parameters = GetParameters();
        for (int i = 0; i < parameters.Length; i++)
            parameters[i] = NumOps.Subtract(parameters[i], NumOps.Multiply(learningRate, gradients[i]));
        SetParameters(parameters);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }



    /// <inheritdoc />
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["QueryDimension"] = _queryDimension.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["ContextDimension"] = _contextDimension.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["HeadCount"] = _headCount.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["ZeroOutputProjection"] = _zeroOutputProjection.ToString();
        metadata["OutputBiasInitialization"] = _outputBiasInitialization.ToString();
        return metadata;
    }

    /// <inheritdoc />
    protected override void EnsureParametersMaterialized()
    {
        if (_queryWeights.Length > 0) return;
        _queryWeights = AllocateLazyWeight([_queryDimension, _queryDimension]);
        _keyWeights = AllocateLazyWeight([_contextDimension, _queryDimension]);
        _valueWeights = AllocateLazyWeight([_contextDimension, _queryDimension]);
        _outputWeights = AllocateLazyWeight([_queryDimension, _queryDimension]);
        _outputBias = AllocateLazyWeight([_queryDimension]);
        InitializeLinear(_queryWeights, _queryDimension);
        InitializeLinear(_keyWeights, _contextDimension);
        InitializeLinear(_valueWeights, _contextDimension);
        if (_zeroOutputProjection)
            Engine.TensorFill(_outputWeights, NumOps.Zero);
        else
            InitializeLinear(_outputWeights, _queryDimension);
        // nn.Linear's bias bound is 1/sqrt(fan_in) with fan_in the projection's input width (the query width here).
        if (!_zeroOutputProjection && _outputBiasInitialization == AttentionBiasInitialization.TorchDefault)
            InitializeLinear(_outputBias, _queryDimension);
        else
            Engine.TensorFill(_outputBias, NumOps.Zero);
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputBias, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Loads a reference attention module's <c>to_q</c>, <c>to_k</c>, <c>to_v</c> (bias-free, <c>[out, in]</c>) and
    /// <c>to_out.0</c> (weight and bias) under <paramref name="prefix"/>, as LDM's <c>CrossAttention</c> names them.
    /// </summary>
    internal IReadOnlyList<string> LoadTorchWeights(AiDotNet.ModelLoading.TorchTensorSource source, string prefix)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        MaterializeParameters();
        var names = new List<string>(5);
        void Transposed(string name, int outputs, int inputs, Tensor<T> destination)
        {
            source.CopyTransposedTo(name, new[] { outputs, inputs }, outputs, inputs, destination);
            Engine.InvalidatePersistentTensor(destination);
            names.Add(name);
        }
        Transposed(prefix + "to_q.weight", _queryDimension, _queryDimension, _queryWeights);
        Transposed(prefix + "to_k.weight", _queryDimension, _contextDimension, _keyWeights);
        Transposed(prefix + "to_v.weight", _queryDimension, _contextDimension, _valueWeights);
        Transposed(prefix + "to_out.0.weight", _queryDimension, _queryDimension, _outputWeights);
        source.CopyTo(prefix + "to_out.0.bias", new[] { _queryDimension }, _outputBias);
        Engine.InvalidatePersistentTensor(_outputBias);
        names.Add(prefix + "to_out.0.bias");
        return names;
    }

    private void InitializeLinear(Tensor<T> tensor, int fanIn)
    {
        T bound = NumOps.FromDouble(1.0 / System.Math.Sqrt(fanIn));
        var initialized = Engine.TensorRandomUniformRange<T>(
            tensor.Shape.ToArray(), NumOps.Negate(bound), bound, Random.Next());
        initialized.AsSpan().CopyTo(tensor.Data.Span);
    }

}
