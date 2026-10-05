using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.ComputerVision;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// DaViT, the dual-attention vision transformer that Florence-2 uses as its image encoder (Ding et al. 2022;
/// HF <c>Florence2VisionBackbone</c>).
/// </summary>
/// <remarks>
/// <para>
/// There are four stages, and each starts with a strided convolutional patch embedding (kernels 7/3/3/3,
/// strides 4/2/2/2). The total stride is 32. Stage 0 applies its LayerNorm after the convolution; the later
/// stages apply it before.
/// </para>
/// <para>
/// Every block pairs a spatial block with a channel block. Each has the form
/// <c>x += dwconv(x); x += attention(LN(x)); x += dwconv(x); x += MLP(LN(x))</c>:
/// </para>
/// <list type="bullet">
/// <item>Spatial attention is multi-head attention inside non-overlapping windows. The map is zero-padded to
/// whole windows.</item>
/// <item>Channel attention splits the channels into groups and attends channel-to-channel. Its scores are
/// scaled by <c>N^-1/2</c>, where N is the token count.</item>
/// </list>
/// <para>
/// Widths and heads (also the group counts) double at every stage from <c>baseDim</c> and <c>baseHeads</c>.
/// Depths are 1/1/<c>thirdStageDepth</c>/1. Florence-2-base is (128, 4, 9) and Florence-2-large is (256, 8, 9).
/// Stochastic depth, a pretraining regularizer, is not applied.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "3, 64, 64", TestConstructorArgs = "64, 8, 1, 1, 2, 2")]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output, Note = "The last stage's feature map at stride 32.")]
[AutoParameters]
public partial class DaViTLayer<T> : LayerBase<T>, IShapeContract
{
    private const int Stages = 4;
    private static readonly int[] KernelSizes = { 7, 3, 3, 3 };
    private static readonly int[] Strides = { 4, 2, 2, 2 };
    private static readonly int[] Paddings = { 3, 1, 1, 1 };

    private readonly int _imageSize;
    private readonly int _baseDim;
    private readonly int _baseHeads;
    private readonly int _thirdStageDepth;
    private readonly int _windowSize;
    private readonly int _mlpRatio;

    private readonly List<ConvolutionalLayer<T>> _embed = new();
    private readonly List<LayerNormalizationLayer<T>> _embedNorm = new();
    private readonly List<ConvolutionalLayer<T>> _depthwise = new();
    private readonly List<LayerNormalizationLayer<T>> _norm = new();
    private readonly List<DenseLayer<T>> _qkv = new();
    private readonly List<DenseLayer<T>> _proj = new();
    private readonly List<DenseLayer<T>> _fc1 = new();
    private readonly List<DenseLayer<T>> _fc2 = new();

    public override bool SupportsTraining => true;

    public DaViTLayer([LayerState] int imageSize, [LayerState] int baseDim, [LayerState] int baseHeads,
        [LayerState] int thirdStageDepth, [LayerState] int windowSize, [LayerState] int mlpRatio)
        : base(new[] { 3, imageSize, imageSize }, new[] { baseDim * 8, imageSize / 32, imageSize / 32 })
    {
        if (imageSize <= 0 || imageSize % 32 != 0)
            throw new ArgumentException($"imageSize ({imageSize}) must be a positive multiple of DaViT's total stride, 32.", nameof(imageSize));
        if (baseDim <= 0 || baseHeads <= 0 || baseDim % baseHeads != 0)
            throw new ArgumentException($"baseDim ({baseDim}) must be a positive multiple of baseHeads ({baseHeads}).", nameof(baseHeads));
        if (thirdStageDepth <= 0) throw new ArgumentOutOfRangeException(nameof(thirdStageDepth));
        if (windowSize <= 0) throw new ArgumentOutOfRangeException(nameof(windowSize));
        if (mlpRatio <= 0) throw new ArgumentOutOfRangeException(nameof(mlpRatio));
        _imageSize = imageSize;
        _baseDim = baseDim;
        _baseHeads = baseHeads;
        _thirdStageDepth = thirdStageDepth;
        _windowSize = windowSize;
        _mlpRatio = mlpRatio;

        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        for (int s = 0; s < Stages; s++)
        {
            int dim = Dim(s);
            _embed.Add(new ConvolutionalLayer<T>(dim, KernelSizes[s], Strides[s], Paddings[s], identity));
            _embedNorm.Add(new LayerNormalizationLayer<T>());
            for (int b = 0; b < Depth(s); b++)
                for (int kind = 0; kind < 2; kind++)           // 0 = spatial block, 1 = channel block
                {
                    _depthwise.Add(new ConvolutionalLayer<T>(dim, 3, 1, 1, identity, groups: dim));
                    _norm.Add(new LayerNormalizationLayer<T>());
                    _qkv.Add(new DenseLayer<T>(3 * dim, identity));
                    _proj.Add(new DenseLayer<T>(dim, identity));
                    _depthwise.Add(new ConvolutionalLayer<T>(dim, 3, 1, 1, identity, groups: dim));
                    _norm.Add(new LayerNormalizationLayer<T>());
                    _fc1.Add(new DenseLayer<T>(dim * mlpRatio, (IActivationFunction<T>)new GELUActivation<T>()));
                    _fc2.Add(new DenseLayer<T>(dim, identity));
                }
        }
        foreach (var layer in SubLayers()) RegisterSubLayer(layer);
    }

    private int Dim(int stage) => _baseDim << stage;
    private int Heads(int stage) => _baseHeads << stage;
    private int Depth(int stage) => stage == 2 ? _thirdStageDepth : 1;

    /// <summary>Channels of the stride-32 output map.</summary>
    public int OutputChannels => Dim(Stages - 1);

    /// <summary>Side of the square stride-32 output map.</summary>
    public int OutputGrid => _imageSize / 32;

    private IEnumerable<LayerBase<T>> SubLayers()
    {
        foreach (var layer in _embed) yield return layer;
        foreach (var layer in _embedNorm) yield return layer;
        foreach (var layer in _depthwise) yield return layer;
        foreach (var layer in _norm) yield return layer;
        foreach (var layer in _qkv) yield return layer;
        foreach (var layer in _proj) yield return layer;
        foreach (var layer in _fc1) yield return layer;
        foreach (var layer in _fc2) yield return layer;
    }

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank is 3 or 4
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(OutputChannels)),
            new OutputAxisContract(TensorAxis.Height, AxisRelation.Fixed(OutputGrid)),
            new OutputAxisContract(TensorAxis.Width, AxisRelation.Fixed(OutputGrid)),
        }
        : null;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var x = input.Rank == 3 ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] }) : input;
        if (x.Rank != 4 || x.Shape[0] != 1 || x.Shape[1] != 3 || x.Shape[2] != _imageSize || x.Shape[3] != _imageSize)
            throw new ArgumentException(
                $"DaViTLayer expects one image of shape [3, {_imageSize}, {_imageSize}]; got shape [{string.Join(", ", input.Shape.ToArray())}].",
                nameof(input));
        int block = 0;
        for (int s = 0; s < Stages; s++)
        {
            if (s > 0) x = ChannelNorm(_embedNorm[s], x);
            x = _embed[s].Forward(x);
            if (s == 0) x = ChannelNorm(_embedNorm[s], x);
            for (int b = 0; b < Depth(s); b++)
            {
                x = Block(block++, x, Heads(s), spatial: true);
                x = Block(block++, x, Heads(s), spatial: false);
            }
        }
        int c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        return Engine.Reshape(x, new[] { c, h, w });
    }

    /// <summary>
    /// One spatial or channel block: <c>x += dwconv(x); x += attn(LN(x)); x += dwconv(x); x += MLP(LN(x))</c>.
    /// </summary>
    private Tensor<T> Block(int block, Tensor<T> x, int heads, bool spatial)
    {
        int c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        x = Engine.TensorAdd(x, _depthwise[2 * block].Forward(x));
        var t = Tokens(x);
        var normed = _norm[2 * block].Forward(t);
        var attended = spatial ? WindowAttention(block, normed, h, w, heads) : ChannelAttention(block, normed, heads);
        x = Map(Engine.TensorAdd(t, attended), c, h, w);
        x = Engine.TensorAdd(x, _depthwise[(2 * block) + 1].Forward(x));
        t = Tokens(x);
        t = Engine.TensorAdd(t, _fc2[block].Forward(_fc1[block].Forward(_norm[(2 * block) + 1].Forward(t))));
        return Map(t, c, h, w);
    }

    /// <summary>Spatial-block attention of block <paramref name="block"/> over row-major tokens <c>[h*w, C]</c>.</summary>
    internal Tensor<T> WindowAttention(int block, Tensor<T> t, int h, int w, int heads)
    {
        int c = t.Shape[1], ws = _windowSize, dh = c / heads, perWindow = ws * ws;
        // Window order is (windowY, windowX, y, x), as in the reference's view/permute; see WindowPartition.
        var (gather, inverse, windows) = WindowPartition.Indices(h, w, ws);
        var windowed = WindowPartition.Partition(t, gather);                                      // [W*P, C]
        var qkv = Engine.Reshape(_qkv[block].Forward(windowed), new[] { windows, perWindow, 3, heads, dh });
        qkv = Engine.Reshape(Engine.TensorPermute(qkv, new[] { 2, 0, 3, 1, 4 }), new[] { 3, windows * heads, perWindow, dh });
        Tensor<T> Part(int i) => Engine.Reshape(
            Engine.TensorSlice(qkv, new[] { i, 0, 0, 0 }, new[] { 1, windows * heads, perWindow, dh }), new[] { windows * heads, perWindow, dh });
        var q = Part(0);
        var k = Part(1);
        var v = Part(2);
        var scores = Engine.TensorMultiplyScalar(
            Engine.TensorBatchMatMul<T>(q, Engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(Math.Pow(dh, -0.5)));
        var context = Engine.TensorBatchMatMul<T>(Engine.TensorSoftmax(scores, 2), v);              // [W*H, P, dh]
        context = Engine.TensorPermute(Engine.Reshape(context, new[] { windows, heads, perWindow, dh }), new[] { 0, 2, 1, 3 });
        var projected = _proj[block].Forward(Engine.Reshape(context, new[] { windows * perWindow, c }));
        return WindowPartition.Merge(projected, inverse);                                         // [N, C]
    }

    /// <summary>Channel-block attention of block <paramref name="block"/> over tokens <c>[N, C]</c>.</summary>
    internal Tensor<T> ChannelAttention(int block, Tensor<T> t, int groups)
    {
        int n = t.Shape[0], c = t.Shape[1], cg = c / groups;
        var qkv = Engine.Reshape(_qkv[block].Forward(t), new[] { n, 3, groups, cg });
        qkv = Engine.TensorPermute(qkv, new[] { 1, 2, 3, 0 });                                    // [3, G, C/G, N]
        Tensor<T> Part(int i) => Engine.Reshape(
            Engine.TensorSlice(qkv, new[] { i, 0, 0, 0 }, new[] { 1, groups, cg, n }), new[] { groups, cg, n });
        var q = Part(0);
        var k = Part(1);
        var v = Part(2);
        var scores = Engine.TensorMultiplyScalar(
            Engine.TensorBatchMatMul<T>(q, Engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(Math.Pow(n, -0.5)));
        var context = Engine.TensorBatchMatMul<T>(Engine.TensorSoftmax(scores, 2), v);              // [G, C/G, N]
        var merged = Engine.Reshape(Engine.TensorPermute(context, new[] { 2, 0, 1 }), new[] { n, c });
        return _proj[block].Forward(merged);
    }

    private Tensor<T> ChannelNorm(LayerNormalizationLayer<T> norm, Tensor<T> x)
    {
        int c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        return Map(norm.Forward(Tokens(x)), c, h, w);
    }

    private Tensor<T> Tokens(Tensor<T> x)
    {
        int c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        return Engine.TensorPermute(Engine.Reshape(x, new[] { c, h * w }), new[] { 1, 0 });        // [HW, C]
    }

    private Tensor<T> Map(Tensor<T> t, int c, int h, int w) =>
        Engine.Reshape(Engine.TensorPermute(t, new[] { 1, 0 }), new[] { 1, c, h, w });

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ImageSize"] = _imageSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["BaseDim"] = _baseDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["BaseHeads"] = _baseHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["ThirdStageDepth"] = _thirdStageDepth.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["WindowSize"] = _windowSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MlpRatio"] = _mlpRatio.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    public override void ResetState() { foreach (var layer in SubLayers()) layer.ResetState(); }
}

/// <summary>
/// Florence-2's visual projector (HF <c>Florence2MultiModalProjector</c>). It turns DaViT's last feature map
/// into the image tokens that are spliced into the language encoder.
/// </summary>
/// <remarks>
/// The steps are:
/// <list type="number">
/// <item>Add a learned 2-D position embedding. Its first half comes from a column table and its second half from
/// a row table.</item>
/// <item>Flatten the map and add the 1-D cosine temporal embedding of frame 0.</item>
/// <item>Prepend the mean over all positions as one extra token.</item>
/// <item>Project with a bias-free linear layer, then apply LayerNorm. The result is <c>1 + h*w</c> tokens.</item>
/// </list>
/// </remarks>
[LayerCategory(LayerCategory.Embedding)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.Low, TestInputShape = "8, 2, 2", TestConstructorArgs = "8, 6, 2")]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output, Note = "The mean token, then one token per position.")]
[AutoParameters]
public partial class Florence2ProjectorLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _visionDim;
    private readonly int _projectionDim;
    private readonly int _grid;
    private readonly int _maxPositions;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _projection;

    private readonly EmbeddingLayer<T> _columns;
    private readonly EmbeddingLayer<T> _rows;
    [SubLayerInput("_projectionDim")] private readonly LayerNormalizationLayer<T> _norm;

    public override bool SupportsTraining => true;

    public Florence2ProjectorLayer([LayerState] int visionDim, [LayerState] int projectionDim, [LayerState] int grid,
        [LayerState] int maxPositions = 50)
        : base(new[] { visionDim, grid, grid }, new[] { 1 + (grid * grid), projectionDim })
    {
        if (visionDim < 2) throw new ArgumentOutOfRangeException(nameof(visionDim));
        if (projectionDim <= 0) throw new ArgumentOutOfRangeException(nameof(projectionDim));
        if (grid <= 0 || grid > maxPositions)
            throw new ArgumentException($"grid ({grid}) must be positive and within the {maxPositions} learned positions.", nameof(grid));
        _visionDim = visionDim;
        _projectionDim = projectionDim;
        _grid = grid;
        _maxPositions = maxPositions;

        var random = LayerInitializationSeedScope.NextRandom();
        _projection = NormalTensor(new[] { visionDim, projectionDim }, 0.02, random);
        RegisterTrainableParameter(_projection, PersistentTensorRole.Weights);
        // The reference splits the width as column = C - C/2, row = C/2.
        _columns = LayerGraphContract.FromDerivedInput(new EmbeddingLayer<T>(maxPositions, visionDim - (visionDim / 2)), "positions");
        _rows = LayerGraphContract.FromDerivedInput(new EmbeddingLayer<T>(maxPositions, visionDim / 2), "positions");
        _norm = new LayerNormalizationLayer<T>();
        RegisterSubLayer(_columns);
        RegisterSubLayer(_rows);
        RegisterSubLayer(_norm);
    }

    /// <summary>Number of image tokens produced: the mean token plus one per position.</summary>
    public int TokenCount => 1 + (_grid * _grid);

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank is 3 or 4
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Fixed(TokenCount)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_projectionDim)),
        }
        : null;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var map = input.Rank == 4 && input.Shape[0] == 1
            ? Engine.Reshape(input, new[] { input.Shape[1], input.Shape[2], input.Shape[3] })
            : input;
        if (map.Rank != 3 || map.Shape[0] != _visionDim || map.Shape[1] != _grid || map.Shape[2] != _grid)
            throw new ArgumentException(
                $"Florence2ProjectorLayer expects a feature map of shape [{_visionDim}, {_grid}, {_grid}]; got shape [{string.Join(", ", input.Shape.ToArray())}].",
                nameof(input));
        int n = _grid * _grid;
        var tokens = Engine.TensorPermute(Engine.Reshape(map, new[] { _visionDim, n }), new[] { 1, 0 });   // [N, C]

        var positions = new Tensor<T>(new[] { _grid });
        for (int i = 0; i < _grid; i++) positions[i] = NumOps.FromDouble(i);
        var columnTable = _columns.Forward(positions);                                                     // [w, C - C/2]
        var rowTable = _rows.Forward(positions);                                                           // [h, C/2]
        var columnOf = new int[n];
        var rowOf = new int[n];
        for (int y = 0; y < _grid; y++)
            for (int x = 0; x < _grid; x++) { columnOf[(y * _grid) + x] = x; rowOf[(y * _grid) + x] = y; }
        var position = Engine.TensorConcatenate(new[]
        {
            CvTensorOps<T>.Select(columnTable, columnOf, 0),
            CvTensorOps<T>.Select(rowTable, rowOf, 0)
        }, 1);
        tokens = Engine.TensorAdd(tokens, position);

        // Cosine temporal embedding of frame 0: sin(0) = 0 on even channels, cos(0) = 1 on odd ones.
        var temporal = new Tensor<T>(new[] { n, _visionDim });
        for (int i = 0; i < n; i++)
            for (int d = 1; d < _visionDim; d += 2) temporal[i, d] = NumOps.One;
        tokens = Engine.TensorAdd(tokens, temporal);

        var mean = Engine.ReduceMean(tokens, new[] { 0 }, keepDims: true);                                // [1, C]
        var all = Engine.TensorConcatenate(new[] { mean, tokens }, 0);                                     // [1 + N, C]
        return _norm.Forward(Engine.TensorMatMul(all, _projection));
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["VisionDim"] = _visionDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["ProjectionDim"] = _projectionDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Grid"] = _grid.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MaxPositions"] = _maxPositions.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    public override void ResetState()
    {
        _columns.ResetState();
        _rows.ResetState();
        _norm.ResetState();
    }
}

/// <summary>
/// The BART encoder-decoder Florence-2 uses as its language model (Lewis et al. 2020; HF <c>BartModel</c>).
/// </summary>
/// <remarks>
/// The architecture follows the reference:
/// <list type="bullet">
/// <item>Learned absolute positions with BART's offset of 2, then <c>layernorm_embedding</c>.</item>
/// <item>Post-norm blocks with biased projections and GELU feed-forwards.</item>
/// <item>A causal decoder with cross-attention.</item>
/// <item>A head tied to the shared embedding.</item>
/// </list>
/// The encoder takes precomputed embeddings, so Florence-2 can splice image tokens ahead of its text. BART's
/// <c>final_logits_bias</c> is a zero buffer and is omitted. Dropout is applied to the residual branches in
/// training mode.
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 2, Cost = ComputeCost.High, TestInputShape = "5, 16", TestConstructorArgs = "20, 16, 2, 32, 1, 1, 64, 0")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input, Note = "Encoder input embeddings.")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output, Note = "Next-token logits for the decoder start token.")]
[AutoParameters]
public partial class BartEncoderDecoderLayer<T> : LayerBase<T>, IShapeContract
{
    /// <summary>BART's learned-position offset.</summary>
    private const int PositionOffset = 2;

    /// <summary>BART's decoder start token (also EOS).</summary>
    public const int DecoderStartTokenId = 2;

    private readonly int _vocabSize;
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _ffnDim;
    private readonly int _numEncoderLayers;
    private readonly int _numDecoderLayers;
    private readonly int _maxPositions;
    private readonly double _dropoutRate;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _shared;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _encoderPositions;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _decoderPositions;

    [SubLayerInput("_dim")] private readonly LayerNormalizationLayer<T> _encoderEmbedNorm;
    [SubLayerInput("_dim")] private readonly LayerNormalizationLayer<T> _decoderEmbedNorm;
    [SubLayerInput("_dim")] private readonly List<DenseLayer<T>> _query = new();
    [SubLayerInput("_dim")] private readonly List<DenseLayer<T>> _key = new();
    [SubLayerInput("_dim")] private readonly List<DenseLayer<T>> _value = new();
    [SubLayerInput("_dim")] private readonly List<DenseLayer<T>> _output = new();
    [SubLayerInput("_dim")] private readonly List<LayerNormalizationLayer<T>> _norm = new();
    [SubLayerInput("_dim")] private readonly List<DenseLayer<T>> _fc1 = new();
    [SubLayerInput("_ffnDim")] private readonly List<DenseLayer<T>> _fc2 = new();
    private readonly DropoutLayer<T> _dropout;

    public override bool SupportsTraining => true;

    public BartEncoderDecoderLayer([LayerState] int vocabSize, [LayerState] int dim, [LayerState] int numHeads,
        [LayerState] int ffnDim, [LayerState] int numEncoderLayers, [LayerState] int numDecoderLayers,
        [LayerState] int maxPositions, [LayerState] double dropoutRate)
        : base(new[] { -1, dim }, new[] { 1, vocabSize })
    {
        if (vocabSize <= DecoderStartTokenId) throw new ArgumentOutOfRangeException(nameof(vocabSize), "BART needs at least BOS (0), PAD (1) and EOS (2).");
        if (dim <= 0 || numHeads <= 0 || dim % numHeads != 0)
            throw new ArgumentException($"dim ({dim}) must be a positive multiple of numHeads ({numHeads}).", nameof(numHeads));
        if (ffnDim <= 0) throw new ArgumentOutOfRangeException(nameof(ffnDim));
        if (numEncoderLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numEncoderLayers));
        if (numDecoderLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numDecoderLayers));
        if (maxPositions <= 0) throw new ArgumentOutOfRangeException(nameof(maxPositions));
        if (dropoutRate < 0 || dropoutRate >= 1) throw new ArgumentOutOfRangeException(nameof(dropoutRate));
        _vocabSize = vocabSize;
        _dim = dim;
        _numHeads = numHeads;
        _ffnDim = ffnDim;
        _numEncoderLayers = numEncoderLayers;
        _numDecoderLayers = numDecoderLayers;
        _maxPositions = maxPositions;
        _dropoutRate = dropoutRate;

        // BartPreTrainedModel._init_weights: N(0, init_std = 0.02); the padding row (id 1) starts at zero.
        var random = LayerInitializationSeedScope.NextRandom();
        _shared = NormalTensor(new[] { vocabSize, dim }, 0.02, random);
        for (int c = 0; c < dim; c++) _shared[1, c] = NumOps.Zero;
        _encoderPositions = NormalTensor(new[] { maxPositions + PositionOffset, dim }, 0.02, random);
        _decoderPositions = NormalTensor(new[] { maxPositions + PositionOffset, dim }, 0.02, random);
        RegisterTrainableParameter(_shared, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_encoderPositions, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_decoderPositions, PersistentTensorRole.Weights);

        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        _encoderEmbedNorm = new LayerNormalizationLayer<T>();
        _decoderEmbedNorm = new LayerNormalizationLayer<T>();
        // Encoder layer i owns attention block i and norms 2i, 2i+1. Decoder layer j owns self-attention block
        // E + 2j, cross-attention block E + 2j + 1, and norms 2E + 3j .. 2E + 3j + 2. Feed-forward i is E + j.
        int attentionBlocks = numEncoderLayers + (2 * numDecoderLayers);
        for (int i = 0; i < attentionBlocks; i++)
        {
            _query.Add(new DenseLayer<T>(dim, identity));
            _key.Add(new DenseLayer<T>(dim, identity));
            _value.Add(new DenseLayer<T>(dim, identity));
            _output.Add(new DenseLayer<T>(dim, identity));
        }
        for (int i = 0; i < (2 * numEncoderLayers) + (3 * numDecoderLayers); i++) _norm.Add(new LayerNormalizationLayer<T>());
        for (int i = 0; i < numEncoderLayers + numDecoderLayers; i++)
        {
            _fc1.Add(new DenseLayer<T>(ffnDim, (IActivationFunction<T>)new GELUActivation<T>()));
            _fc2.Add(new DenseLayer<T>(dim, identity));
        }
        _dropout = new DropoutLayer<T>(dropoutRate);
        foreach (var layer in SubLayers()) RegisterSubLayer(layer);
    }

    private IEnumerable<LayerBase<T>> SubLayers()
    {
        yield return _encoderEmbedNorm;
        yield return _decoderEmbedNorm;
        foreach (var layer in _query) yield return layer;
        foreach (var layer in _key) yield return layer;
        foreach (var layer in _value) yield return layer;
        foreach (var layer in _output) yield return layer;
        foreach (var layer in _norm) yield return layer;
        foreach (var layer in _fc1) yield return layer;
        foreach (var layer in _fc2) yield return layer;
        yield return _dropout;
    }

    public int VocabSize => _vocabSize;

    public int Dim => _dim;

    /// <summary>Longest encoder or decoder sequence the learned positions cover.</summary>
    public int MaxPositions => _maxPositions;

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 2
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Fixed(1)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_vocabSize)),
        }
        : null;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 2 || input.Shape[1] != _dim)
            throw new ArgumentException(
                $"BartEncoderDecoderLayer expects encoder embeddings of shape [S, {_dim}]; got shape [{string.Join(", ", input.Shape.ToArray())}].",
                nameof(input));
        return Decode(new[] { DecoderStartTokenId }, Encode(input));
    }

    /// <summary>Looks token ids up in the shared embedding: <c>[S, dim]</c>.</summary>
    internal Tensor<T> Embed(IReadOnlyList<int> ids) =>
        CvTensorOps<T>.Select(_shared, ids.Select(ClampToken).ToArray(), 0);

    internal int ClampToken(int id) => Math.Min(Math.Max(id, 0), _vocabSize - 1);

    /// <summary>Runs the encoder over precomputed input embeddings <c>[S, dim]</c>.</summary>
    internal Tensor<T> Encode(Tensor<T> embeddings)
    {
        int s = embeddings.Shape[0];
        if (s == 0 || s > _maxPositions)
            throw new ArgumentException($"The encoder input ({s} positions) must be within 1..{_maxPositions}.", nameof(embeddings));
        var x = _encoderEmbedNorm.Forward(Engine.TensorAdd(embeddings, Positions(_encoderPositions, s)));
        x = _dropout.Forward(x);
        for (int i = 0; i < _numEncoderLayers; i++)
        {
            x = _norm[2 * i].Forward(Engine.TensorAdd(x, _dropout.Forward(Attention(i, x, x, causal: false))));
            x = _norm[(2 * i) + 1].Forward(Engine.TensorAdd(x, _dropout.Forward(_fc2[i].Forward(_fc1[i].Forward(x)))));
        }
        return x;
    }

    /// <summary>Runs the causal decoder over <paramref name="ids"/> and returns next-token logits <c>[T, vocab]</c>.</summary>
    internal Tensor<T> Decode(IReadOnlyList<int> ids, Tensor<T> memory)
    {
        int t = ids.Count;
        if (t == 0 || t > _maxPositions)
            throw new ArgumentException($"The decoder input ({t} tokens) must be within 1..{_maxPositions}.", nameof(ids));
        var y = _decoderEmbedNorm.Forward(Engine.TensorAdd(Embed(ids), Positions(_decoderPositions, t)));
        y = _dropout.Forward(y);
        int e = _numEncoderLayers;
        for (int j = 0; j < _numDecoderLayers; j++)
        {
            int norms = (2 * e) + (3 * j);
            y = _norm[norms].Forward(Engine.TensorAdd(y, _dropout.Forward(Attention(e + (2 * j), y, y, causal: true))));
            y = _norm[norms + 1].Forward(Engine.TensorAdd(y, _dropout.Forward(Attention(e + (2 * j) + 1, y, memory, causal: false))));
            y = _norm[norms + 2].Forward(Engine.TensorAdd(y, _dropout.Forward(_fc2[e + j].Forward(_fc1[e + j].Forward(y)))));
        }
        return Engine.TensorMatMul(y, Engine.TensorTranspose(_shared));
    }

    private Tensor<T> Attention(int block, Tensor<T> x, Tensor<T> memory, bool causal) =>
        _output[block].Forward(MultiHeadAttentionMath<T>.Attend(
            _query[block].Forward(x), _key[block].Forward(memory), _value[block].Forward(memory), _numHeads, causal, xPos: false));

    private Tensor<T> Positions(Tensor<T> table, int count) =>
        CvTensorOps<T>.Select(table, Enumerable.Range(PositionOffset, count).ToArray(), 0);

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["VocabSize"] = _vocabSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Dim"] = _dim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["FfnDim"] = _ffnDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumEncoderLayers"] = _numEncoderLayers.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumDecoderLayers"] = _numDecoderLayers.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MaxPositions"] = _maxPositions.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["DropoutRate"] = _dropoutRate.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    public override void ResetState() { foreach (var layer in SubLayers()) layer.ResetState(); }
}
