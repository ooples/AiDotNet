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
        int tokens = leading.Aggregate(1, (a, b) => a * b);
        int sequence = leading[leading.Length - 1];
        var span = input.ToArray();

        Tensor<T> Column(Func<double[], double> select)
        {
            var column = new Tensor<T>(leading);
            for (int t = 0; t < tokens; t++)
            {
                var box = new[] { NumOps.ToDouble(span[t * 4]), NumOps.ToDouble(span[(t * 4) + 1]), NumOps.ToDouble(span[(t * 4) + 2]), NumOps.ToDouble(span[(t * 4) + 3]) };
                column.Data.Span[t] = NumOps.FromDouble(Math.Min(Math.Max(Math.Round(select(box)), 0), _maxPosition2D - 1));
            }
            return column;
        }

        var parts = new[]
        {
            _xEmbeddings.Forward(Column(b => b[0])),
            _yEmbeddings.Forward(Column(b => b[1])),
            _xEmbeddings.Forward(Column(b => b[2])),
            _yEmbeddings.Forward(Column(b => b[3])),
            _heightEmbeddings.Forward(Column(b => b[3] - b[1])),
            _widthEmbeddings.Forward(Column(b => b[2] - b[0])),
        };
        var spatial = _boxLinear.Forward(Engine.TensorConcatenate(parts, parts[0].Shape.Length - 1));

        var positions = new Tensor<T>(leading);
        for (int t = 0; t < tokens; t++)
            positions.Data.Span[t] = NumOps.FromDouble(Math.Min(t % sequence, _maxSequenceLength - 1));
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
