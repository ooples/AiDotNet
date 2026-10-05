using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.ComputerVision;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The SAM ViTDet image encoder that GOT-OCR2 uses (Kirillov et al. 2023; Wei et al. 2024; HF
/// <c>GotOcr2VisionEncoder</c>).
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item>A <c>patchSize</c>-stride patch convolution, then an absolute position table.</item>
/// <item>Pre-LN blocks <c>x += attn(LN x); x += MLP(LN x)</c>. Attention is windowed, except in every
/// <c>globalEvery</c>-th block, which attends over the whole grid. For ViTDet-B that is blocks 2, 5, 8 and
/// 11.</item>
/// <item>Each block adds decomposed relative positions (Li et al. 2022, MViTv2). The bias between query
/// (qh, qw) and key (kh, kw) is <c>q . Rh[qh - kh] + q . Rw[qw - kw]</c>, using the unscaled query.</item>
/// <item>A neck of conv1x1, LN2d, conv3x3, LN2d down to <c>neckChannels</c>, with bias-free convolutions.</item>
/// </list>
/// The position and relative tables start at zero, as the reference initialises them.
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "3, 32, 32", TestConstructorArgs = "32, 8, 8, 2, 2, 16, 2, 2, 4")]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output, Note = "The neck's map at the patch stride.")]
[AutoParameters]
public partial class SamViTDetEncoderLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _imageSize;
    private readonly int _patchSize;
    private readonly int _dim;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _mlpDim;
    private readonly int _windowSize;
    private readonly int _globalEvery;
    private readonly int _neckChannels;
    private readonly int _grid;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _absolutePositions;

    // Relative tables, one slice per block: windowed blocks [windowLayers, 2*window - 1, headDim], global blocks
    // [globalLayers, 2*grid - 1, headDim].
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _relativeWindowHeight;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _relativeWindowWidth;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _relativeGlobalHeight;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _relativeGlobalWidth;

    private readonly ConvolutionalLayer<T> _patchEmbedding;
    [SubLayerInput("_dim")] private readonly List<LayerNormalizationLayer<T>> _norm1 = new();
    [SubLayerInput("_dim")] private readonly List<DenseLayer<T>> _qkv = new();
    [SubLayerInput("_dim")] private readonly List<DenseLayer<T>> _proj = new();
    [SubLayerInput("_dim")] private readonly List<LayerNormalizationLayer<T>> _norm2 = new();
    [SubLayerInput("_dim")] private readonly List<DenseLayer<T>> _fc1 = new();
    [SubLayerInput("_mlpDim")] private readonly List<DenseLayer<T>> _fc2 = new();
    private readonly ConvolutionalLayer<T> _neckConv1;
    private readonly LayerNormalizationLayer<T> _neckNorm1;
    private readonly ConvolutionalLayer<T> _neckConv2;
    private readonly LayerNormalizationLayer<T> _neckNorm2;

    public override bool SupportsTraining => true;

    public SamViTDetEncoderLayer([LayerState] int imageSize, [LayerState] int patchSize, [LayerState] int dim,
        [LayerState] int numLayers, [LayerState] int numHeads, [LayerState] int mlpDim, [LayerState] int windowSize,
        [LayerState] int globalEvery, [LayerState] int neckChannels)
        : base(new[] { 3, imageSize, imageSize }, new[] { neckChannels, imageSize / patchSize, imageSize / patchSize })
    {
        if (patchSize <= 0 || imageSize <= 0 || imageSize % patchSize != 0)
            throw new ArgumentException($"imageSize ({imageSize}) must be a positive multiple of patchSize ({patchSize}).", nameof(patchSize));
        if (dim <= 0 || numHeads <= 0 || dim % numHeads != 0)
            throw new ArgumentException($"dim ({dim}) must be a positive multiple of numHeads ({numHeads}).", nameof(numHeads));
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (mlpDim <= 0) throw new ArgumentOutOfRangeException(nameof(mlpDim));
        if (windowSize <= 0) throw new ArgumentOutOfRangeException(nameof(windowSize));
        if (globalEvery < 2 || globalEvery > numLayers)
        if (neckChannels <= 0) throw new ArgumentOutOfRangeException(nameof(neckChannels));
        _imageSize = imageSize;
        _patchSize = patchSize;
        _dim = dim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _mlpDim = mlpDim;
        _windowSize = windowSize;
        _globalEvery = globalEvery;
        _neckChannels = neckChannels;
        _grid = imageSize / patchSize;

        int headDim = dim / numHeads;
        _absolutePositions = new Tensor<T>(new[] { _grid * _grid, dim });
        RegisterTrainableParameter(_absolutePositions, PersistentTensorRole.Weights);
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        int globalLayers = numLayers / globalEvery, windowLayers = numLayers - globalLayers;
        _relativeWindowHeight = new Tensor<T>(new[] { windowLayers, (2 * windowSize) - 1, headDim });
        _relativeWindowWidth = new Tensor<T>(new[] { windowLayers, (2 * windowSize) - 1, headDim });
        _relativeGlobalHeight = new Tensor<T>(new[] { globalLayers, (2 * _grid) - 1, headDim });
        _relativeGlobalWidth = new Tensor<T>(new[] { globalLayers, (2 * _grid) - 1, headDim });
        RegisterTrainableParameter(_relativeWindowHeight, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_relativeWindowWidth, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_relativeGlobalHeight, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_relativeGlobalWidth, PersistentTensorRole.Weights);
        _patchEmbedding = new ConvolutionalLayer<T>(dim, patchSize, patchSize, 0, identity);
        for (int i = 0; i < numLayers; i++)
        {
            _norm1.Add(new LayerNormalizationLayer<T>(dim, 1e-6));
            _qkv.Add(new DenseLayer<T>(3 * dim, identity));
            _proj.Add(new DenseLayer<T>(dim, identity));
            _norm2.Add(new LayerNormalizationLayer<T>(dim, 1e-6));
            _fc1.Add(new DenseLayer<T>(mlpDim, (IActivationFunction<T>)new GELUActivation<T>()));
            _fc2.Add(new DenseLayer<T>(dim, identity));
        }
        _neckConv1 = new ConvolutionalLayer<T>(neckChannels, 1, 1, 0, identity, biasMode: BiasMode.Never);
        _neckNorm1 = new LayerNormalizationLayer<T>(neckChannels, 1e-6);
        _neckConv2 = new ConvolutionalLayer<T>(neckChannels, 3, 1, 1, identity, biasMode: BiasMode.Never);
        _neckNorm2 = new LayerNormalizationLayer<T>(neckChannels, 1e-6);
        foreach (var layer in SubLayers()) RegisterSubLayer(layer);
    }

    private bool IsGlobal(int layer) => (layer + 1) % _globalEvery == 0;

    /// <summary>Block <paramref name="layer"/>'s height or width table <c>[2S - 1, headDim]</c>.</summary>
    private Tensor<T> RelativeTable(int layer, bool height)
    {
        bool global = IsGlobal(layer);
        int slot = global ? ((layer + 1) / _globalEvery) - 1 : layer - ((layer + 1) / _globalEvery);
        var table = global ? (height ? _relativeGlobalHeight : _relativeGlobalWidth) : (height ? _relativeWindowHeight : _relativeWindowWidth);
        int rows = table.Shape[1], hd = table.Shape[2];
        return Engine.Reshape(Engine.TensorSlice(table, new[] { slot, 0, 0 }, new[] { 1, rows, hd }), new[] { rows, hd });
    }

    /// <summary>Side of the output map.</summary>
    public int Grid => _grid;

    /// <summary>Channels of the output map.</summary>
    public int NeckChannels => _neckChannels;

    private IEnumerable<LayerBase<T>> SubLayers()
    {
        yield return _patchEmbedding;
        for (int i = 0; i < _numLayers; i++)
        {
            yield return _norm1[i]; yield return _qkv[i]; yield return _proj[i];
            yield return _norm2[i]; yield return _fc1[i]; yield return _fc2[i];
        }
        yield return _neckConv1;
        yield return _neckNorm1;
        yield return _neckConv2;
        yield return _neckNorm2;
    }

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank is 3 or 4
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_neckChannels)),
            new OutputAxisContract(TensorAxis.Height, AxisRelation.Fixed(_grid)),
            new OutputAxisContract(TensorAxis.Width, AxisRelation.Fixed(_grid)),
        }
        : null;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var image = input.Rank == 3 ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] }) : input;
        if (image.Rank != 4 || image.Shape[0] != 1 || image.Shape[1] != 3 || image.Shape[2] != _imageSize || image.Shape[3] != _imageSize)
            throw new ArgumentException(
                $"SamViTDetEncoderLayer expects one image of shape [3, {_imageSize}, {_imageSize}]; got shape [{string.Join(", ", input.Shape.ToArray())}].",
                nameof(input));
        var x = Tokens(_patchEmbedding.Forward(image));
        x = Engine.TensorAdd(x, _absolutePositions);
        for (int i = 0; i < _numLayers; i++)
        {
            x = Engine.TensorAdd(x, Attention(i, _norm1[i].Forward(x)));
            x = Engine.TensorAdd(x, _fc2[i].Forward(_fc1[i].Forward(_norm2[i].Forward(x))));
        }
        var map = Map(x, _dim);
        map = Map(_neckNorm1.Forward(Tokens(_neckConv1.Forward(map))), _neckChannels);
        map = Map(_neckNorm2.Forward(Tokens(_neckConv2.Forward(map))), _neckChannels);
        return Engine.Reshape(map, new[] { _neckChannels, _grid, _grid });
    }

    /// <summary>Block <paramref name="layer"/>'s attention over row-major tokens <c>[grid^2, dim]</c>.</summary>
    internal Tensor<T> Attention(int layer, Tensor<T> x)
    {
        bool global = IsGlobal(layer);
        int side = global ? _grid : _windowSize, perWindow = side * side, headDim = _dim / _numHeads;
        int windows = 1;
        int[]? inverse = null;
        var rows = x;
        if (!global)
        {
            var (gather, back, count) = WindowPartition.Indices(_grid, _grid, side);
            rows = WindowPartition.Partition(x, gather);
            inverse = back;
            windows = count;
        }
        int batch = windows * _numHeads;
        var qkv = Engine.Reshape(_qkv[layer].Forward(rows), new[] { windows, perWindow, 3, _numHeads, headDim });
        qkv = Engine.Reshape(Engine.TensorPermute(qkv, new[] { 2, 0, 3, 1, 4 }), new[] { 3, batch, perWindow, headDim });
        Tensor<T> Part(int i) => Engine.Reshape(
            Engine.TensorSlice(qkv, new[] { i, 0, 0, 0 }, new[] { 1, batch, perWindow, headDim }), new[] { batch, perWindow, headDim });
        var q = Part(0);
        var k = Part(1);
        var v = Part(2);
        var scores = Engine.TensorBatchMatMul<T>(
            Engine.TensorMultiplyScalar(q, NumOps.FromDouble(Math.Pow(headDim, -0.5))), Engine.TensorPermute(k, new[] { 0, 2, 1 }));
        scores = Engine.TensorAdd(scores, RelativeBias(layer, q, side, batch));
        var context = Engine.TensorBatchMatMul<T>(Engine.TensorSoftmax(scores, 2), v);              // [B', S^2, hd]
        context = Engine.TensorPermute(Engine.Reshape(context, new[] { windows, _numHeads, perWindow, headDim }), new[] { 0, 2, 1, 3 });
        var projected = _proj[layer].Forward(Engine.Reshape(context, new[] { windows * perWindow, _dim }));
        return inverse is null ? projected : WindowPartition.Merge(projected, inverse);
    }

    /// <summary>
    /// The decomposed relative-position bias <c>[B', S^2, S^2]</c> for queries <c>[B', S^2, hd]</c>. It is
    /// <c>q . Rh[qh - kh + S - 1]</c> summed over key columns plus <c>q . Rw[qw - kw + S - 1]</c> summed over key rows.
    /// </summary>
    internal Tensor<T> RelativeBias(int layer, Tensor<T> q, int side, int batch)
    {
        int s = side, hd = q.Shape[2];
        var index = new int[s * s];
        for (int a = 0; a < s; a++)
            for (int b = 0; b < s; b++) index[(a * s) + b] = a - b + s - 1;
        var rh = Engine.Reshape(CvTensorOps<T>.Select(RelativeTable(layer, true), index, 0), new[] { s, s, hd });   // [qh, kh, hd]
        var rw = Engine.Reshape(CvTensorOps<T>.Select(RelativeTable(layer, false), index, 0), new[] { s, s, hd });  // [qw, kw, hd]

        var grid = Engine.Reshape(q, new[] { batch, s, s, hd });                                                // [B', qh, qw, hd]
        // rel_h[b, qh, qw, kh] = q[b, qh, qw] . rh[qh, kh]: batch over qh.
        var byRow = Engine.Reshape(Engine.TensorPermute(grid, new[] { 1, 0, 2, 3 }), new[] { s, batch * s, hd });
        var relH = Engine.TensorBatchMatMul<T>(byRow, Engine.TensorPermute(rh, new[] { 0, 2, 1 }));            // [qh, B'*qw, kh]
        relH = Engine.TensorPermute(Engine.Reshape(relH, new[] { s, batch, s, s }), new[] { 1, 0, 2, 3 });      // [B', qh, qw, kh]
        // rel_w[b, qh, qw, kw] = q[b, qh, qw] . rw[qw, kw]: batch over qw.
        var byColumn = Engine.Reshape(Engine.TensorPermute(grid, new[] { 2, 0, 1, 3 }), new[] { s, batch * s, hd });
        var relW = Engine.TensorBatchMatMul<T>(byColumn, Engine.TensorPermute(rw, new[] { 0, 2, 1 }));         // [qw, B'*qh, kw]
        relW = Engine.TensorPermute(Engine.Reshape(relW, new[] { s, batch, s, s }), new[] { 1, 2, 0, 3 });      // [B', qh, qw, kw]

        // Spread each term over the other key axis with a 0/1 expansion: key k = kh * S + kw.
        var expandRow = new Tensor<T>(new[] { s, s * s });
        var expandColumn = new Tensor<T>(new[] { s, s * s });
        for (int kh = 0; kh < s; kh++)
            for (int kw = 0; kw < s; kw++)
            {
                expandRow[kh, (kh * s) + kw] = NumOps.One;
                expandColumn[kw, (kh * s) + kw] = NumOps.One;
            }
        var bias = Engine.TensorAdd(
            Engine.TensorMatMul(Engine.Reshape(relH, new[] { batch * s * s, s }), expandRow),
            Engine.TensorMatMul(Engine.Reshape(relW, new[] { batch * s * s, s }), expandColumn));
        return Engine.Reshape(bias, new[] { batch, s * s, s * s });
    }

    private Tensor<T> Tokens(Tensor<T> map)
    {
        int c = map.Shape[1], h = map.Shape[2], w = map.Shape[3];
        return Engine.TensorPermute(Engine.Reshape(map, new[] { c, h * w }), new[] { 1, 0 });
    }

    private Tensor<T> Map(Tensor<T> tokens, int channels) =>
        Engine.Reshape(Engine.TensorPermute(tokens, new[] { 1, 0 }), new[] { 1, channels, _grid, _grid });

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ImageSize"] = _imageSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["PatchSize"] = _patchSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Dim"] = _dim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumLayers"] = _numLayers.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MlpDim"] = _mlpDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["WindowSize"] = _windowSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["GlobalEvery"] = _globalEvery.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NeckChannels"] = _neckChannels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    public override void ResetState() { foreach (var layer in SubLayers()) layer.ResetState(); }
}

/// <summary>
/// GOT-OCR2's vision-to-language projector (HF <c>GotOcr2MultiModalProjector</c>). Two bias-free stride-2 3x3
/// convolutions (C to 2C, then 2C to <c>textDim</c>) cut the map 4x per side. The tokens are then flattened and
/// pass a linear layer. ViTDet-B's 64x64x256 neck output becomes 256 tokens.
/// </summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.Medium, TestInputShape = "4, 8, 8", TestConstructorArgs = "4, 6, 8")]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output, Note = "One token per position of the 4x-downsampled map.")]
[AutoParameters]
public partial class GotOcr2ProjectorLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _visionChannels;
    private readonly int _textDim;
    private readonly int _grid;

    private readonly ConvolutionalLayer<T> _down1;
    private readonly ConvolutionalLayer<T> _down2;
    [SubLayerInput("_textDim")] private readonly DenseLayer<T> _projection;

    public override bool SupportsTraining => true;

    public GotOcr2ProjectorLayer([LayerState] int visionChannels, [LayerState] int textDim, [LayerState] int grid)
        : base(new[] { visionChannels, grid, grid }, new[] { (grid / 4) * (grid / 4), textDim })
    {
        if (visionChannels <= 0) throw new ArgumentOutOfRangeException(nameof(visionChannels));
        if (textDim <= 0) throw new ArgumentOutOfRangeException(nameof(textDim));
        if (grid <= 0 || grid % 4 != 0)
            throw new ArgumentException($"grid ({grid}) must be a positive multiple of 4: the projector halves it twice.", nameof(grid));
        _visionChannels = visionChannels;
        _textDim = textDim;
        _grid = grid;
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        _down1 = new ConvolutionalLayer<T>(2 * visionChannels, 3, 2, 1, identity, biasMode: BiasMode.Never);
        _down2 = new ConvolutionalLayer<T>(textDim, 3, 2, 1, identity, biasMode: BiasMode.Never);
        _projection = new DenseLayer<T>(textDim, identity);
        RegisterSubLayer(_down1);
        RegisterSubLayer(_down2);
        RegisterSubLayer(_projection);
    }

    /// <summary>Number of image tokens produced.</summary>
    public int TokenCount => (_grid / 4) * (_grid / 4);

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank is 3 or 4
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Fixed(TokenCount)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_textDim)),
        }
        : null;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var map = input.Rank == 3 ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] }) : input;
        if (map.Rank != 4 || map.Shape[0] != 1 || map.Shape[1] != _visionChannels || map.Shape[2] != _grid || map.Shape[3] != _grid)
            throw new ArgumentException(
                $"GotOcr2ProjectorLayer expects a map of shape [{_visionChannels}, {_grid}, {_grid}]; got shape [{string.Join(", ", input.Shape.ToArray())}].",
                nameof(input));
        var reduced = _down2.Forward(_down1.Forward(map));                                                 // [1, D, g/4, g/4]
        int n = TokenCount;
        var tokens = Engine.TensorPermute(Engine.Reshape(reduced, new[] { _textDim, n }), new[] { 1, 0 }); // [N, D]
        return _projection.Forward(tokens);
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["VisionChannels"] = _visionChannels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["TextDim"] = _textDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Grid"] = _grid.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    public override void ResetState()
    {
        _down1.ResetState();
        _down2.ResetState();
        _projection.ResetState();
    }
}

/// <summary>
/// A Qwen-family causal decoder (Bai et al. 2023), the language model of GOT-OCR2: Qwen-0.5B. It reuses the
/// repository's LLaMA-style pre-RMSNorm blocks (<see cref="PreLNTransformerBlock{T}"/>), with rotary
/// grouped-query attention (biased q/k/v, as Qwen) and a SiLU-gated feed-forward. The head is tied to the
/// token embedding.
/// </summary>
/// <remarks>
/// Image embeddings can be spliced into the token sequence at a given offset. GOT-OCR2's prompt reserves
/// <c>&lt;imgpad&gt;</c> slots for them, and the reference fills them with <c>masked_scatter</c>.
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 1, Cost = ComputeCost.High, TestInputShape = "5", TestConstructorArgs = "20, 8, 2, 2, 16, 1, 16, 10000.0, 1e-6")]
[TensorPort("input", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices, Role = TensorPortRole.TokenIds, MaxExclusiveMember = "_vocabSize")]
[TensorPort("output", TensorPortDirection.Output, LayerInputDomainKind.Continuous, Role = TensorPortRole.Features)]
[TensorLayout(TensorAxis.Time, Direction = TensorLayoutDirection.Input, Note = "Token ids.")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output, Note = "Next-token logits.")]
[AutoParameters]
public partial class QwenDecoderLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _vocabSize;
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _numKeyValueHeads;
    private readonly int _ffnDim;
    private readonly int _numLayers;
    private readonly int _maxPositions;
    private readonly double _ropeTheta;
    private readonly double _rmsNormEpsilon;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _embedTokens;

    private readonly List<PreLNTransformerBlock<T>> _blocks = new();
    [SubLayerInput("_dim")] private readonly RMSNormalizationLayer<T> _finalNorm;

    public override bool SupportsTraining => true;

    public QwenDecoderLayer([LayerState] int vocabSize, [LayerState] int dim, [LayerState] int numHeads,
        [LayerState] int numKeyValueHeads, [LayerState] int ffnDim, [LayerState] int numLayers,
        [LayerState] int maxPositions, [LayerState] double ropeTheta, [LayerState] double rmsNormEpsilon)
        : base(new[] { -1 }, new[] { -1, vocabSize })
    {
        if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (dim <= 0 || numHeads <= 0 || dim % numHeads != 0 || (dim / numHeads) % 2 != 0)
            throw new ArgumentException($"dim ({dim}) must split into an even head width over numHeads ({numHeads}).", nameof(numHeads));
        if (numKeyValueHeads <= 0 || numHeads % numKeyValueHeads != 0)
            throw new ArgumentException($"numHeads ({numHeads}) must be a multiple of numKeyValueHeads ({numKeyValueHeads}).", nameof(numKeyValueHeads));
        if (ffnDim <= 0) throw new ArgumentOutOfRangeException(nameof(ffnDim));
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (maxPositions <= 0) throw new ArgumentOutOfRangeException(nameof(maxPositions));
        if (ropeTheta <= 0) throw new ArgumentOutOfRangeException(nameof(ropeTheta));
        if (rmsNormEpsilon <= 0) throw new ArgumentOutOfRangeException(nameof(rmsNormEpsilon));
        _vocabSize = vocabSize;
        _dim = dim;
        _numHeads = numHeads;
        _numKeyValueHeads = numKeyValueHeads;
        _ffnDim = ffnDim;
        _numLayers = numLayers;
        _maxPositions = maxPositions;
        _ropeTheta = ropeTheta;
        _rmsNormEpsilon = rmsNormEpsilon;

        // Qwen2PreTrainedModel._init_weights: N(0, initializer_range = 0.02).
        var random = LayerInitializationSeedScope.NextRandom();
        _embedTokens = NormalTensor(new[] { vocabSize, dim }, 0.02, random);
        RegisterTrainableParameter(_embedTokens, PersistentTensorRole.Weights);

        var qwen = AiDotNet.ModelLoading.Pretrained.DecoderOptions<T>.Qwen2;
        for (int i = 0; i < numLayers; i++)
        {
            var attention = new GroupedQueryAttentionLayer<T>(
                sequenceLength: maxPositions, embeddingDimension: dim, numHeads: numHeads, numKVHeads: numKeyValueHeads,
                useProjectionBias: qwen.UseAttentionQkvBias, useCausalMask: true);
            attention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, ropeTheta, maxPositions);
            _blocks.Add(new PreLNTransformerBlock<T>(hiddenSize: dim, ffnDim: ffnDim, attention: attention,
                ffnActivation: qwen.FfnActivation, gated: true));
        }
        _finalNorm = new RMSNormalizationLayer<T>(dim, rmsNormEpsilon);
        foreach (var block in _blocks) RegisterSubLayer(block);
        RegisterSubLayer(_finalNorm);
    }

    public int VocabSize => _vocabSize;

    public int Dim => _dim;

    /// <summary>Longest sequence the rotary tables and attention cover.</summary>
    public int MaxPositions => _maxPositions;

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 1
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_vocabSize)),
        }
        : null;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 1)
            throw new ArgumentException($"QwenDecoderLayer expects token ids of shape [S]; got rank {input.Rank}.", nameof(input));
        var ids = new int[input.Shape[0]];
        for (int i = 0; i < ids.Length; i++) ids[i] = (int)Math.Round(NumOps.ToDouble(input[i]));
        return Forward(ids, null, 0);
    }

    internal int ClampToken(int id) => Math.Min(Math.Max(id, 0), _vocabSize - 1);

    /// <summary>
    /// Logits <c>[S, vocab]</c> for <paramref name="ids"/>. When <paramref name="imageEmbeddings"/> is given,
    /// its rows replace the embeddings of positions <c>imageStart .. imageStart + K - 1</c>.
    /// </summary>
    internal Tensor<T> Forward(IReadOnlyList<int> ids, Tensor<T>? imageEmbeddings, int imageStart)
    {
        int s = ids.Count;
        if (s == 0 || s > _maxPositions)
            throw new ArgumentException($"The sequence ({s} tokens) must be within 1..{_maxPositions}.", nameof(ids));
        var x = CvTensorOps<T>.Select(_embedTokens, ids.Select(ClampToken).ToArray(), 0);              // [S, d]
        if (imageEmbeddings is not null)
        {
            int k = imageEmbeddings.Shape[0];
            if (imageStart < 0 || imageStart + k > s || imageEmbeddings.Shape[1] != _dim)
                throw new ArgumentException($"{k} image embeddings of width {imageEmbeddings.Shape[1]} do not fit at {imageStart} of a {s}-token, {_dim}-wide sequence.", nameof(imageEmbeddings));
            var parts = new List<Tensor<T>>();
            if (imageStart > 0) parts.Add(Engine.TensorSlice(x, new[] { 0, 0 }, new[] { imageStart, _dim }));
            parts.Add(imageEmbeddings);
            if (imageStart + k < s) parts.Add(Engine.TensorSlice(x, new[] { imageStart + k, 0 }, new[] { s - imageStart - k, _dim }));
            x = Engine.TensorConcatenate(parts.ToArray(), 0);
        }
        var h = Engine.Reshape(x, new[] { 1, s, _dim });
        foreach (var block in _blocks) h = block.Forward(h);
        h = _finalNorm.Forward(Engine.Reshape(h, new[] { s, _dim }));
        return Engine.TensorMatMul(h, Engine.TensorTranspose(_embedTokens));
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["VocabSize"] = _vocabSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Dim"] = _dim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumKeyValueHeads"] = _numKeyValueHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["FfnDim"] = _ffnDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumLayers"] = _numLayers.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MaxPositions"] = _maxPositions.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["RopeTheta"] = _ropeTheta.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        metadata["RmsNormEpsilon"] = _rmsNormEpsilon.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    public override void ResetState()
    {
        foreach (var block in _blocks) block.ResetState();
        _finalNorm.ResetState();
    }
}