using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// LiLT's layout-flow embedding (Wang et al., ACL 2022; reference <c>LiltLayoutEmbeddings</c>). It runs
/// these steps:
/// <list type="bullet">
/// <item>Six coordinate embeddings, each <c>hidden / 6</c> wide: x0, y0, x1, y1, the height and the width.
/// The x table serves both horizontal edges, and the y table both vertical edges.</item>
/// <item>The six are concatenated, which gives back <c>hidden</c> whenever it divides by 6, as in every released LiLT;
/// otherwise it gives the largest multiple of 6 below it, and the linear map sizes itself to that.</item>
/// <item>A linear map to the layout flow's width <c>hidden / channelShrinkRatio</c>.</item>
/// <item>A learned reading-order position embedding of that width is added.</item>
/// <item>LayerNorm.</item>
/// </list>
/// </summary>
/// <remarks>
/// <para>The input is one box per token, <c>[S, 4]</c> or <c>[B, S, 4]</c>, as integer (x0, y0, x1, y1) on
/// the 0-1000 page grid. Coordinates are clamped into the tables, as the reference requires them to be.</para>
/// <para><b>For Beginners:</b> This turns each word's bounding box into a vector for LiLT's small layout
/// transformer, which runs beside the text transformer.</para>
/// </remarks>
[LayerCategory(LayerCategory.Embedding)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 2, Cost = ComputeCost.Low, TestInputShape = "4, 4", TestConstructorArgs = "12, 8, 16, 4")]
[TensorPort("input", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices,
    Role = TensorPortRole.PositionIds, MaxExclusiveMember = "_maxPosition2D")]
[TensorPort("output", TensorPortDirection.Output, LayerInputDomainKind.Continuous,
    Role = TensorPortRole.Features)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input, Note = "One (x0, y0, x1, y1) box per token.")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output, Note = "The layout flow's width, hidden / channelShrinkRatio.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input, Note = "One (x0, y0, x1, y1) box per token.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class LiltLayoutEmbeddingLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _hiddenDim;
    private readonly int _maxSequenceLength;
    private readonly int _maxPosition2D;
    private readonly int _channelShrinkRatio;

    private readonly EmbeddingLayer<T> _xEmbeddings;
    private readonly EmbeddingLayer<T> _yEmbeddings;
    private readonly EmbeddingLayer<T> _heightEmbeddings;
    private readonly EmbeddingLayer<T> _widthEmbeddings;
    [SubLayerInput("_hiddenDim")]
    private readonly DenseLayer<T> _boxLinear;
    private readonly EmbeddingLayer<T> _boxPositionEmbeddings;
    private readonly LayerNormalizationLayer<T> _norm;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>Creates the embedding. LiLT-base: hidden 768 (tables of 128), shrink ratio 4 (a 192-wide layout flow).</summary>
    public LiltLayoutEmbeddingLayer(
        [LayerState] int hiddenDim,
        [LayerState] int maxSequenceLength = 512,
        [LayerState] int maxPosition2D = 1024,
        [LayerState] int channelShrinkRatio = 4)
        : base(new[] { -1, 4 }, new[] { -1, hiddenDim / Math.Max(1, channelShrinkRatio) })
    {
        if (hiddenDim < 6)
            throw new ArgumentException($"hiddenDim ({hiddenDim}) must be at least 6: each of the six coordinate tables is hidden / 6 wide.", nameof(hiddenDim));
        if (channelShrinkRatio <= 0 || hiddenDim % channelShrinkRatio != 0)
            throw new ArgumentException($"hiddenDim ({hiddenDim}) must be divisible by channelShrinkRatio ({channelShrinkRatio}).", nameof(channelShrinkRatio));
        if (maxSequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));
        if (maxPosition2D <= 0) throw new ArgumentOutOfRangeException(nameof(maxPosition2D));

        _hiddenDim = hiddenDim;
        _maxSequenceLength = maxSequenceLength;
        _maxPosition2D = maxPosition2D;
        _channelShrinkRatio = channelShrinkRatio;
        int table = hiddenDim / 6, layoutDim = hiddenDim / channelShrinkRatio;
        _xEmbeddings = new EmbeddingLayer<T>(maxPosition2D, table);
        _yEmbeddings = new EmbeddingLayer<T>(maxPosition2D, table);
        _heightEmbeddings = new EmbeddingLayer<T>(maxPosition2D, table);
        _widthEmbeddings = new EmbeddingLayer<T>(maxPosition2D, table);
        _boxLinear = new DenseLayer<T>(layoutDim, (IActivationFunction<T>)new IdentityActivation<T>());
        _boxPositionEmbeddings = new EmbeddingLayer<T>(maxSequenceLength, layoutDim);
        _norm = new LayerNormalizationLayer<T>();

        RegisterSubLayer(_xEmbeddings);
        RegisterSubLayer(_yEmbeddings);
        RegisterSubLayer(_heightEmbeddings);
        RegisterSubLayer(_widthEmbeddings);
        RegisterSubLayer(_boxLinear);
        RegisterSubLayer(_boxPositionEmbeddings);
        RegisterSubLayer(_norm);
    }

    /// <inheritdoc/>
    /// <remarks>The box axis (4 coordinates) becomes the layout width, a constructor argument, so it is <c>Fixed</c>.</remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank switch
    {
        2 => new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(LayoutDim)),
        },
        3 => new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(LayoutDim)),
        },
        _ => null,
    };

    /// <summary>Width of the layout flow this embedding feeds: <c>hidden / channelShrinkRatio</c>.</summary>
    public int LayoutDim => _hiddenDim / _channelShrinkRatio;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Shape.Length < 2 || input.Shape[input.Shape.Length - 1] != 4)
            throw new ArgumentException("LiltLayoutEmbeddingLayer expects boxes shaped [S, 4] or [B, S, 4] as (x0, y0, x1, y1).", nameof(input));
        int rank = input.Shape.Length;
        var leading = new int[rank - 1];
        for (int i = 0; i < rank - 1; i++) leading[i] = input.Shape[i];
        int sequence = leading[leading.Length - 1];

        // Integer lookup indices, built off the gradient tape as the previous fresh tensors were:
        // clamp(round(value), 0, maxPosition2D - 1) per box coordinate and extent, and the
        // per-row token position clamp(s, 0, maxSequenceLength - 1).
        Tensor<T> x0, y0, x1, y1, heights, widths, positions;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var sliceLength = new int[rank];
            Array.Copy(leading, sliceLength, rank - 1);
            sliceLength[rank - 1] = 1;
            Tensor<T> Coordinate(int k)
            {
                var start = new int[rank];
                start[rank - 1] = k;
                return Engine.Reshape(Engine.TensorSlice(input, start, sliceLength), leading);
            }

            T lastPosition = NumOps.FromDouble(_maxPosition2D - 1);
            Tensor<T> Index(Tensor<T> value) => Engine.TensorClamp(Engine.TensorRound(value), NumOps.Zero, lastPosition);

            var rawX0 = Coordinate(0);
            var rawY0 = Coordinate(1);
            var rawX1 = Coordinate(2);
            var rawY1 = Coordinate(3);
            x0 = Index(rawX0);
            y0 = Index(rawY0);
            x1 = Index(rawX1);
            y1 = Index(rawY1);
            heights = Index(Engine.TensorSubtract(rawY1, rawY0));
            widths = Index(Engine.TensorSubtract(rawX1, rawX0));
            positions = Engine.TensorBroadcastTo(
                Engine.TensorClampMax(Engine.TensorArange<T>(0, sequence, 1), NumOps.FromDouble(_maxSequenceLength - 1)),
                leading);
        }

        var parts = new[]
        {
            _xEmbeddings.Forward(x0),
            _yEmbeddings.Forward(y0),
            _xEmbeddings.Forward(x1),
            _yEmbeddings.Forward(y1),
            _heightEmbeddings.Forward(heights),
            _widthEmbeddings.Forward(widths),
        };
        var spatial = _boxLinear.Forward(Engine.TensorConcatenate(parts, parts[0].Shape.Length - 1));
        return _norm.Forward(Engine.TensorAdd(spatial, _boxPositionEmbeddings.Forward(positions)));
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["HiddenDim"] = _hiddenDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MaxSequenceLength"] = _maxSequenceLength.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MaxPosition2D"] = _maxPosition2D.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["ChannelShrinkRatio"] = _channelShrinkRatio.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _xEmbeddings.ResetState(); _yEmbeddings.ResetState(); _heightEmbeddings.ResetState(); _widthEmbeddings.ResetState();
        _boxLinear.ResetState(); _boxPositionEmbeddings.ResetState(); _norm.ResetState();
    }
}
