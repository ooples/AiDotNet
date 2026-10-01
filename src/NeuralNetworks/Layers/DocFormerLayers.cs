using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.ComputerVision;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// DocFormer's spatial embedding of one stream (Appalaraju et al., ICCV 2021; reference DocFormerEmbeddings).
/// </summary>
/// <remarks>
/// <para>
/// Per axis, a token is described by eight lookups:
/// <list type="bullet">
/// <item>Its two edges (tables of width <c>coordinateSize</c>).</item>
/// <item>Its extent.</item>
/// <item>The distances of its four corners and its centroid to the previous token's (tables of width
/// <c>coordinateSize</c>; distances are clamped to <c>+-maxPosition2D</c>).</item>
/// </list>
/// The eight are concatenated. The x and y concatenations and a sinusoidal position are then summed, giving
/// <c>8 x coordinateSize = hidden</c>. The visual and text streams each own one of these layers, as in the
/// reference.
/// </para>
/// <para>Input: boxes <c>[S, 4]</c> as integer (x0, y0, x1, y1) on the 0 .. maxPosition2D - 1 grid.</para>
/// <para><b>For Beginners:</b> This turns each word's box, and where it sits relative to the word before
/// it, into a vector the attention can compare.</para>
/// </remarks>
[LayerCategory(LayerCategory.Embedding)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 2, Cost = ComputeCost.Low, TestInputShape = "4, 4", TestConstructorArgs = "16, 32")]
[TensorPort("input", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices, Role = TensorPortRole.PositionIds, MaxExclusiveMember = "_maxPosition2D")]
[TensorPort("output", TensorPortDirection.Output, LayerInputDomainKind.Continuous, Role = TensorPortRole.Features)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input, Note = "One (x0, y0, x1, y1) box per token.")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class DocFormerSpatialEmbeddingLayer<T> : LayerBase<T>, IShapeContract
{
    private const int Features = 8;
    private readonly int _hidden;
    private readonly int _maxPosition2D;
    private readonly int _coordinateSize;

    // x tables then y tables: [topleft, bottomright, extent, d_topleft, d_bottomleft, d_topright, d_bottomright, d_centroid].
    private readonly List<EmbeddingLayer<T>> _tables = new();

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>Creates the embedding. DocFormer-base: hidden 768 (tables of 96), a 1024-position grid.</summary>
    public DocFormerSpatialEmbeddingLayer([LayerState] int hidden, [LayerState] int maxPosition2D = 1024)
        : base(new[] { -1, 4 }, new[] { -1, hidden })
    {
        if (hidden <= 0 || hidden % Features != 0 || hidden % 2 != 0)
            throw new ArgumentException($"hidden ({hidden}) must be a positive multiple of {Features} (eight tables per axis).", nameof(hidden));
        if (maxPosition2D <= 0) throw new ArgumentOutOfRangeException(nameof(maxPosition2D));
        _hidden = hidden;
        _maxPosition2D = maxPosition2D;
        _coordinateSize = hidden / Features;
        for (int axis = 0; axis < 2; axis++)
            for (int f = 0; f < Features; f++)
            {
                var table = new EmbeddingLayer<T>(f < 3 ? maxPosition2D : (2 * maxPosition2D) + 1, _coordinateSize);
                _tables.Add(table);
                RegisterSubLayer(table);
            }
    }

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 2
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_hidden)),
        }
        : null;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 2 || input.Shape[1] != 4)
            throw new ArgumentException(
                $"DocFormerSpatialEmbeddingLayer expects boxes of shape [S, 4]; got shape [{string.Join(", ", input.Shape.ToArray())}].", nameof(input));
        int s = input.Shape[0];
        var box = new double[s][];
        for (int i = 0; i < s; i++)
            box[i] = Enumerable.Range(0, 4).Select(c => Math.Min(Math.Max(Math.Round(NumOps.ToDouble(input[i, c])), 0), _maxPosition2D - 1)).ToArray();

        Tensor<T> Axis(int axis)
        {
            // axis 0: x (edges x0, x1); axis 1: y (edges y0, y1).
            int lo = axis == 0 ? 0 : 1, hi = axis == 0 ? 2 : 3;
            var parts = new Tensor<T>[Features];
            for (int f = 0; f < Features; f++)
            {
                var ids = new Tensor<T>(new[] { s });
                for (int i = 0; i < s; i++)
                {
                    var b = box[i];
                    var p = i > 0 ? box[i - 1] : b;
                    double value = f switch
                    {
                        0 => b[lo],
                        1 => b[hi],
                        2 => b[hi] - b[lo],
                        3 => b[lo] - p[lo],                                   // top-left corner
                        4 => axis == 0 ? b[lo] - p[lo] : b[hi] - p[hi],       // bottom-left corner
                        5 => axis == 0 ? b[hi] - p[hi] : b[lo] - p[lo],       // top-right corner
                        6 => b[hi] - p[hi],                                   // bottom-right corner
                        _ => ((b[lo] + b[hi]) / 2) - ((p[lo] + p[hi]) / 2),   // centroid
                    };
                    if (f >= 3) value = Math.Min(Math.Max(Math.Round(value), -_maxPosition2D), _maxPosition2D) + _maxPosition2D;
                    ids[i] = NumOps.FromDouble(Math.Max(0, value));
                }
                parts[f] = _tables[(axis * Features) + f].Forward(ids);
            }
            return Engine.TensorConcatenate(parts, 1);
        }

        var pe = new Tensor<T>(new[] { s, _hidden });
        for (int p = 0; p < s; p++)
            for (int i = 0; i < _hidden; i += 2)
            {
                double angle = p / Math.Pow(10000, (double)i / _hidden);
                pe[p, i] = NumOps.FromDouble(Math.Sin(angle));
                pe[p, i + 1] = NumOps.FromDouble(Math.Cos(angle));
            }
        return Engine.TensorAdd(Engine.TensorAdd(Axis(0), Axis(1)), pe);
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Hidden"] = _hidden.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MaxPosition2D"] = _maxPosition2D.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var table in _tables) table.ResetState(); }
}

/// <summary>
/// DocFormer's multi-modal self-attention encoder (reference DocFormerEncoder + MultiModalAttentionLayer).
/// </summary>
/// <remarks>
/// <para>
/// Each layer sees text features <c>t</c>, visual features <c>v</c>, and the two streams' spatial features
/// <c>ts</c>, <c>vs</c>, all <c>[L, hidden]</c>. The stream scores are
/// <c>q k / sqrt(dh) + q . a_lr + k_r . a_lr + q_s k_s / sqrt(dh)</c>, where <c>a</c> is the stream's
/// clipped relative-position table and the spatial projections <c>q_s</c>, <c>k_s</c> are shared by the two
/// streams. The layer output is <c>out(softmax(S_t) V_t + softmax(S_v) V_v)</c>.
/// </para>
/// <para>
/// A pre-LN residual wraps each layer. It adds the skip <c>t + v + ts + vs</c> and then the GELU FFN, and only
/// <c>t</c> is carried to the next layer: <c>v</c>, <c>ts</c> and <c>vs</c> are re-injected unchanged from
/// layer 0, which is the paper's multi-modal feature re-use.
/// </para>
/// <para>
/// Two choices follow the paper rather than the unofficial port. The attention is scaled by
/// <c>sqrt(head_dim)</c>, where the port uses <c>sqrt(hidden)</c>. The image stream has its own relative
/// table, where the port reuses the text table.
/// </para>
/// <para>The single-input forward takes <c>t</c> with the other streams zero.</para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = false, ExpectedInputRank = 2, Cost = ComputeCost.High, TestInputShape = "4, 8", TestConstructorArgs = "8, 2, 16, 2, 1")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class DocFormerEncoderLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _hidden;
    private readonly int _numHeads;
    private readonly int _ffnDim;
    private readonly int _maxRelative;
    private readonly int _numLayers;
    private readonly int _headDim;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private readonly Tensor<T> _relativeText;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private readonly Tensor<T> _relativeImage;

    [SubLayerInput("_hidden")] private readonly List<LayerNormalizationLayer<T>> _normT = new();
    [SubLayerInput("_hidden")] private readonly List<LayerNormalizationLayer<T>> _normV = new();
    [SubLayerInput("_hidden")] private readonly List<LayerNormalizationLayer<T>> _normTs = new();
    [SubLayerInput("_hidden")] private readonly List<LayerNormalizationLayer<T>> _normVs = new();
    [SubLayerInput("_hidden")] private readonly List<DenseLayer<T>> _qText = new();
    [SubLayerInput("_hidden")] private readonly List<DenseLayer<T>> _kText = new();
    [SubLayerInput("_hidden")] private readonly List<DenseLayer<T>> _vText = new();
    [SubLayerInput("_hidden")] private readonly List<DenseLayer<T>> _qImage = new();
    [SubLayerInput("_hidden")] private readonly List<DenseLayer<T>> _kImage = new();
    [SubLayerInput("_hidden")] private readonly List<DenseLayer<T>> _vImage = new();
    [SubLayerInput("_hidden")] private readonly List<DenseLayer<T>> _qSpatial = new();
    [SubLayerInput("_hidden")] private readonly List<DenseLayer<T>> _kSpatial = new();
    [SubLayerInput("_hidden")] private readonly List<DenseLayer<T>> _output = new();
    [SubLayerInput("_hidden")] private readonly List<LayerNormalizationLayer<T>> _ffnNorm = new();
    [SubLayerInput("_hidden")] private readonly List<DenseLayer<T>> _fc1 = new();
    [SubLayerInput("_ffnDim")] private readonly List<DenseLayer<T>> _fc2 = new();

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>Creates the encoder. DocFormer-base: 768 wide, 12 heads, FFN 3072, relative clip 8, 12 layers.</summary>
    public DocFormerEncoderLayer([LayerState] int hidden, [LayerState] int numHeads, [LayerState] int ffnDim,
        [LayerState] int maxRelative, [LayerState] int numLayers)
        : base(new[] { -1, hidden }, new[] { -1, hidden })
    {
        if (hidden <= 0 || numHeads <= 0 || hidden % numHeads != 0)
            throw new ArgumentException($"hidden ({hidden}) must be a positive multiple of numHeads ({numHeads}).", nameof(numHeads));
        if (ffnDim <= 0) throw new ArgumentOutOfRangeException(nameof(ffnDim));
        if (maxRelative <= 0) throw new ArgumentOutOfRangeException(nameof(maxRelative));
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        _hidden = hidden;
        _numHeads = numHeads;
        _ffnDim = ffnDim;
        _maxRelative = maxRelative;
        _numLayers = numLayers;
        _headDim = hidden / numHeads;

        // RelativePosition: xavier_uniform over [2 * maxRelative + 1, head_dim].
        var random = LayerInitializationSeedScope.NextRandom();
        _relativeText = Xavier(new[] { (2 * maxRelative) + 1, _headDim }, random);
        _relativeImage = Xavier(new[] { (2 * maxRelative) + 1, _headDim }, random);
        RegisterTrainableParameter(_relativeText, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_relativeImage, PersistentTensorRole.Weights);

        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        for (int i = 0; i < numLayers; i++)
        {
            _normT.Add(new LayerNormalizationLayer<T>());
            _normV.Add(new LayerNormalizationLayer<T>());
            _normTs.Add(new LayerNormalizationLayer<T>());
            _normVs.Add(new LayerNormalizationLayer<T>());
            _qText.Add(new DenseLayer<T>(hidden, identity));
            _kText.Add(new DenseLayer<T>(hidden, identity));
            _vText.Add(new DenseLayer<T>(hidden, identity));
            _qImage.Add(new DenseLayer<T>(hidden, identity));
            _kImage.Add(new DenseLayer<T>(hidden, identity));
            _vImage.Add(new DenseLayer<T>(hidden, identity));
            _qSpatial.Add(new DenseLayer<T>(hidden, identity));
            _kSpatial.Add(new DenseLayer<T>(hidden, identity));
            _output.Add(new DenseLayer<T>(hidden, identity));
            _ffnNorm.Add(new LayerNormalizationLayer<T>());
            _fc1.Add(new DenseLayer<T>(ffnDim, (IActivationFunction<T>)new GELUActivation<T>()));
            _fc2.Add(new DenseLayer<T>(hidden, identity));
        }
        foreach (var layer in SubLayers()) RegisterSubLayer(layer);
    }

    private IEnumerable<LayerBase<T>> SubLayers()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            yield return _normT[i]; yield return _normV[i]; yield return _normTs[i]; yield return _normVs[i];
            yield return _qText[i]; yield return _kText[i]; yield return _vText[i];
            yield return _qImage[i]; yield return _kImage[i]; yield return _vImage[i];
            yield return _qSpatial[i]; yield return _kSpatial[i]; yield return _output[i];
            yield return _ffnNorm[i]; yield return _fc1[i]; yield return _fc2[i];
        }
    }

    private Tensor<T> Xavier(int[] shape, Random random)
    {
        var tensor = new Tensor<T>(shape);
        double bound = Math.Sqrt(6.0 / (shape[0] + shape[1]));
        for (int i = 0; i < tensor.Length; i++) tensor[i] = NumOps.FromDouble(((random.NextDouble() * 2) - 1) * bound);
        return tensor;
    }

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 2
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_hidden)),
        }
        : null;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 2 || input.Shape[1] != _hidden)
            throw new ArgumentException(
                $"DocFormerEncoderLayer expects text features of shape [L, {_hidden}]; got shape [{string.Join(", ", input.Shape.ToArray())}].", nameof(input));
        var zero = new Tensor<T>(input._shape);
        return Forward(input, zero, zero, zero);
    }

    /// <summary>Runs the encoder on text, visual and the two spatial feature streams, each <c>[L, hidden]</c>.</summary>
    internal Tensor<T> Forward(Tensor<T> text, Tensor<T> visual, Tensor<T> textSpatial, Tensor<T> visualSpatial)
    {
        int l = text.Shape[0];
        foreach (var stream in new[] { visual, textSpatial, visualSpatial })
            if (stream.Rank != 2 || stream.Shape[0] != l || stream.Shape[1] != _hidden)
                throw new ArgumentException($"Every DocFormer stream must have shape [{l}, {_hidden}].", nameof(visual));

        var relativeIndex = new int[l * l];
        for (int q = 0; q < l; q++)
            for (int k = 0; k < l; k++)
                relativeIndex[(q * l) + k] = Math.Min(Math.Max(k - q, -_maxRelative), _maxRelative) + _maxRelative;
        var relText = Engine.Reshape(CvTensorOps<T>.Select(_relativeText, relativeIndex, 0), new[] { l, l, _headDim });
        var relImage = Engine.Reshape(CvTensorOps<T>.Select(_relativeImage, relativeIndex, 0), new[] { l, l, _headDim });

        var t = text;
        for (int i = 0; i < _numLayers; i++)
        {
            var skip = Engine.TensorAdd(Engine.TensorAdd(t, visual), Engine.TensorAdd(textSpatial, visualSpatial));
            var tn = _normT[i].Forward(t);
            var vn = _normV[i].Forward(visual);
            var tsn = _normTs[i].Forward(textSpatial);
            var vsn = _normVs[i].Forward(visualSpatial);

            var textContext = StreamContext(_qText[i].Forward(tn), _kText[i].Forward(tn), _vText[i].Forward(tn),
                _qSpatial[i].Forward(tsn), _kSpatial[i].Forward(tsn), relText, l);
            var imageContext = StreamContext(_qImage[i].Forward(vn), _kImage[i].Forward(vn), _vImage[i].Forward(vn),
                _qSpatial[i].Forward(vsn), _kSpatial[i].Forward(vsn), relImage, l);
            var context = Engine.Reshape(Engine.TensorPermute(Engine.TensorAdd(textContext, imageContext), new[] { 1, 0, 2 }), new[] { l, _hidden });

            var x = Engine.TensorAdd(_output[i].Forward(context), skip);
            t = Engine.TensorAdd(x, _fc2[i].Forward(_fc1[i].Forward(_ffnNorm[i].Forward(x))));
        }
        return t;
    }

    /// <summary>softmax(scores) V for one stream, <c>[heads, L, head_dim]</c>.</summary>
    private Tensor<T> StreamContext(Tensor<T> q, Tensor<T> k, Tensor<T> v, Tensor<T> qs, Tensor<T> ks, Tensor<T> relative, int l)
    {
        Tensor<T> Heads(Tensor<T> x) => Engine.TensorPermute(Engine.Reshape(x, new[] { l, _numHeads, _headDim }), new[] { 1, 0, 2 });
        var qh = Heads(q);
        var kh = Heads(k);
        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDim));
        var scores = Engine.TensorMultiplyScalar(Engine.TensorBatchMatMul<T>(qh, Engine.TensorPermute(kh, new[] { 0, 2, 1 })), scale);
        scores = Engine.TensorAdd(scores, Engine.TensorMultiplyScalar(
            Engine.TensorBatchMatMul<T>(Heads(qs), Engine.TensorPermute(Heads(ks), new[] { 0, 2, 1 })), scale));

        // q_l . a_lr: batch over l, [L, H, dh] x [L, dh, L] -> [L, H, L] -> [H, L, L].
        var relT = Engine.TensorPermute(relative, new[] { 0, 2, 1 });
        var queryTerm = Engine.TensorBatchMatMul<T>(Engine.TensorPermute(qh, new[] { 1, 0, 2 }), relT);
        scores = Engine.TensorAdd(scores, Engine.TensorPermute(queryTerm, new[] { 1, 0, 2 }));
        // k_r . a_lr: batch over r, [L, H, dh] x [L(r), dh, L(l)] -> [r, H, l] -> [H, l, r].
        var relByKey = Engine.TensorPermute(relative, new[] { 1, 2, 0 });
        var keyTerm = Engine.TensorBatchMatMul<T>(Engine.TensorPermute(kh, new[] { 1, 0, 2 }), relByKey);
        scores = Engine.TensorAdd(scores, Engine.TensorPermute(keyTerm, new[] { 1, 2, 0 }));

        return Engine.TensorBatchMatMul<T>(Engine.TensorSoftmax(scores, 2), Heads(v));
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Hidden"] = _hidden.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["FfnDim"] = _ffnDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MaxRelative"] = _maxRelative.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumLayers"] = _numLayers.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var layer in SubLayers()) layer.ResetState(); }
}
