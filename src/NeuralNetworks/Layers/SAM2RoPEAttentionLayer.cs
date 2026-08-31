using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Multi-head query/context attention with SAM 2's axial rotary position encoding.
/// </summary>
/// <remarks>
/// Rotary encoding is applied to the projected query and key heads, before the scaled dot-product,
/// which is the placement used by Meta's <c>RoPEAttention</c>. Context sequences may contain several
/// flattened memory frames; in that case spatial positions repeat once per frame.
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.AttentionComputation)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = false, ApiShape = LayerApiShape.DualTensor,
    TestInputShape = "1, 4, 16", TestConstructorArgs = "16, 16, 4, 2, 2")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class SAM2RoPEAttentionLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _queryDimension;
    private readonly int _contextDimension;
    private readonly int _headCount;
    private readonly int _spatialHeight;
    private readonly int _spatialWidth;
    private readonly double _ropeTheta;

    private readonly DenseLayer<T> _queryProjection;
    private readonly DenseLayer<T> _keyProjection;
    private readonly DenseLayer<T> _valueProjection;
    private readonly DenseLayer<T> _outputProjection;

    /// <summary>Gets the number of attention heads.</summary>
    public int HeadCount => _headCount;

    /// <summary>Gets the rotary base used by the official SAM 2 configuration.</summary>
    public double RopeTheta => _ropeTheta;

    /// <summary>Creates projected multi-head attention with axial RoPE.</summary>
    public SAM2RoPEAttentionLayer(
        [LayerState] int queryDimension,
        [LayerState] int contextDimension,
        [LayerState] int headCount,
        [LayerState] int spatialHeight,
        [LayerState] int spatialWidth,
        [LayerState] double ropeTheta = 10000.0)
        : base([-1, queryDimension], [-1, queryDimension])
    {
        if (queryDimension <= 0) throw new ArgumentOutOfRangeException(nameof(queryDimension));
        if (contextDimension <= 0) throw new ArgumentOutOfRangeException(nameof(contextDimension));
        if (headCount <= 0 || queryDimension % headCount != 0)
            throw new ArgumentOutOfRangeException(nameof(headCount),
                "The positive head count must divide the query dimension.");
        if (spatialHeight <= 0) throw new ArgumentOutOfRangeException(nameof(spatialHeight));
        if (spatialWidth <= 0) throw new ArgumentOutOfRangeException(nameof(spatialWidth));
        if (double.IsNaN(ropeTheta) || double.IsInfinity(ropeTheta) || ropeTheta <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(ropeTheta));

        _queryDimension = queryDimension;
        _contextDimension = contextDimension;
        _headCount = headCount;
        _spatialHeight = spatialHeight;
        _spatialWidth = spatialWidth;
        _ropeTheta = ropeTheta;

        var identity = new IdentityActivation<T>();
        _queryProjection = new DenseLayer<T>(queryDimension, (IActivationFunction<T>)identity);
        _keyProjection = new DenseLayer<T>(queryDimension, (IActivationFunction<T>)identity);
        _valueProjection = new DenseLayer<T>(queryDimension, (IActivationFunction<T>)identity);
        _outputProjection = new DenseLayer<T>(queryDimension, (IActivationFunction<T>)identity);

        RegisterSubLayer(_queryProjection);
        RegisterSubLayer(_keyProjection);
        RegisterSubLayer(_valueProjection);
        RegisterSubLayer(_outputProjection);

        // Dense layers resolve their input width lazily. Resolve it now so parameter manifests and
        // clone/serialization are stable even before the first real video frame is seen.
        _ = _queryProjection.Forward(new Tensor<T>([1, 1, queryDimension]));
        _ = _keyProjection.Forward(new Tensor<T>([1, 1, contextDimension]));
        _ = _valueProjection.Forward(new Tensor<T>([1, 1, contextDimension]));
        _ = _outputProjection.Forward(new Tensor<T>([1, 1, queryDimension]));
        ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input) => ForwardAttention(input, input, 0);

    /// <inheritdoc />
    protected override Tensor<T> ForwardTracedMany(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 1) return ForwardAttention(inputs[0], inputs[0], 0);
        if (inputs.Length >= 2) return ForwardAttention(inputs[0], inputs[1], 0);
        throw new ArgumentException("SAM2 RoPE attention requires query and optional context tensors.");
    }

    /// <summary>
    /// Runs attention while leaving a tail of non-spatial context tokens (SAM 2 object pointers)
    /// outside RoPE, as in Meta's <c>num_k_exclude_rope</c> path.
    /// </summary>
    internal Tensor<T> ForwardWithUnrotatedContextTail(
        Tensor<T> query, Tensor<T> context, int unrotatedContextTailTokens)
        => ForwardAttention(query, context, unrotatedContextTailTokens);

    private Tensor<T> ForwardAttention(
        Tensor<T> query, Tensor<T> context, int unrotatedContextTailTokens)
    {
        bool unbatchedQuery = query.Rank == 2;
        bool unbatchedContext = context.Rank == 2;
        if (unbatchedQuery)
            query = Engine.Reshape(query, [1, query.Shape[0], query.Shape[1]]);
        if (unbatchedContext)
            context = Engine.Reshape(context, [1, context.Shape[0], context.Shape[1]]);

        if (query.Rank != 3 || query.Shape[2] != _queryDimension)
            throw new ArgumentException(
                $"Expected query [B,L,{_queryDimension}], got [{string.Join(",", query.Shape)}].",
                nameof(query));
        if (context.Rank != 3 || context.Shape[2] != _contextDimension)
            throw new ArgumentException(
                $"Expected context [B,L,{_contextDimension}], got [{string.Join(",", context.Shape)}].",
                nameof(context));
        if (context.Shape[0] != query.Shape[0])
        {
            if (context.Shape[0] != 1)
                throw new ArgumentException("Context batch must equal query batch or be one.", nameof(context));
            context = Engine.TensorTile(context, [query.Shape[0], 1, 1]);
        }

        int batch = query.Shape[0];
        int queryLength = query.Shape[1];
        int contextLength = context.Shape[1];
        if (unrotatedContextTailTokens < 0 || unrotatedContextTailTokens > contextLength)
            throw new ArgumentOutOfRangeException(nameof(unrotatedContextTailTokens));
        int headDimension = _queryDimension / _headCount;

        var q = ToHeads(_queryProjection.Forward(query), batch, queryLength, headDimension);
        var k = ToHeads(_keyProjection.Forward(context), batch, contextLength, headDimension);
        var v = ToHeads(_valueProjection.Forward(context), batch, contextLength, headDimension);

        q = ApplyAxialRotary(q, queryLength, headDimension);
        int rotaryKeyLength = contextLength - unrotatedContextTailTokens;
        if (rotaryKeyLength == contextLength)
        {
            k = ApplyAxialRotary(k, contextLength, headDimension);
        }
        else if (rotaryKeyLength > 0)
        {
            var spatialKeys = Engine.TensorNarrow(k, dim: 2, start: 0, length: rotaryKeyLength);
            spatialKeys = ApplyAxialRotary(spatialKeys, rotaryKeyLength, headDimension);
            var pointerKeys = Engine.TensorNarrow(
                k, dim: 2, start: rotaryKeyLength, length: unrotatedContextTailTokens);
            k = Engine.TensorConcatenate([spatialKeys, pointerKeys], axis: 2);
        }

        var attended = Engine.ScaledDotProductAttention(
            q, k, v, mask: null, scale: 1.0 / Math.Sqrt(headDimension), out _);
        var merged = FromHeads(attended, batch, queryLength, headDimension);
        var output = _outputProjection.Forward(merged);
        return unbatchedQuery
            ? Engine.Reshape(output, [queryLength, _queryDimension])
            : output;
    }

    private Tensor<T> ToHeads(Tensor<T> input, int batch, int length, int headDimension)
    {
        var shaped = Engine.Reshape(input, [batch, length, _headCount, headDimension]);
        return Engine.TensorPermute(shaped, [0, 2, 1, 3]);
    }

    private Tensor<T> FromHeads(Tensor<T> input, int batch, int length, int headDimension)
    {
        var ordered = Engine.TensorPermute(input, [0, 2, 1, 3]);
        return Engine.Reshape(ordered, [batch, length, _headCount * headDimension]);
    }

    private Tensor<T> ApplyAxialRotary(Tensor<T> input, int sequenceLength, int headDimension)
    {
        // RoPE rotates pairs. An odd tail cannot form a pair and is therefore left unchanged.
        int rotaryDimension = headDimension - (headDimension % 2);
        if (rotaryDimension == 0) return input;

        int batch = input.Shape[0];
        int heads = input.Shape[1];
        int spatialTokens = _spatialHeight * _spatialWidth;
        var partnerIndices = new int[input.Length];
        var cos = new Tensor<T>(input.Shape.ToArray());
        var signedSin = new Tensor<T>(input.Shape.ToArray());

        int flat = 0;
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < heads; h++)
            {
                for (int token = 0; token < sequenceLength; token++)
                {
                    int spatial = token % spatialTokens;
                    int y = spatial / _spatialWidth;
                    int x = spatial % _spatialWidth;
                    for (int d = 0; d < headDimension; d++, flat++)
                    {
                        int partnerDimension = d < rotaryDimension ? d ^ 1 : d;
                        partnerIndices[flat] =
                            (((b * heads + h) * sequenceLength + token) * headDimension) + partnerDimension;

                        if (d >= rotaryDimension)
                        {
                            cos[flat] = NumOps.One;
                            signedSin[flat] = NumOps.Zero;
                            continue;
                        }

                        int pair = d / 2;
                        int pairCount = rotaryDimension / 2;
                        bool useX = pair < (pairCount + 1) / 2;
                        int axisPair = useX ? pair : pair - (pairCount + 1) / 2;
                        int axisPairs = Math.Max(1, useX ? (pairCount + 1) / 2 : pairCount / 2);
                        double frequency = Math.Pow(_ropeTheta, -((double)axisPair / axisPairs));
                        double angle = (useX ? x : y) * frequency;
                        cos[flat] = NumOps.FromDouble(Math.Cos(angle));
                        double sign = (d & 1) == 0 ? -1.0 : 1.0;
                        signedSin[flat] = NumOps.FromDouble(sign * Math.Sin(angle));
                    }
                }
            }
        }

        var flattened = Engine.Reshape(input, [input.Length]);
        var partner = Engine.TensorGather(
            flattened, new Tensor<int>(partnerIndices, [partnerIndices.Length]), axis: 0);
        partner = Engine.Reshape(partner, input.Shape.ToArray());
        return Engine.TensorAdd(
            Engine.TensorMultiply(input, cos),
            Engine.TensorMultiply(partner, signedSin));
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _queryProjection.ResetState();
        _keyProjection.ResetState();
        _valueProjection.ResetState();
        _outputProjection.ResetState();
    }
}
