using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.ComputerVision;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>How a <see cref="MagnetoDecoderLayer{T}"/> encodes token positions.</summary>
public enum KosmosPositionEncoding
{
    /// <summary>Fairseq sinusoidal absolute positions added to the embeddings (KOSMOS-2 released code).</summary>
    Sinusoidal,

    /// <summary>xPos relative positions inside every attention (Sun et al. 2022; the KOSMOS-1 paper).</summary>
    XPos
}

/// <summary>Multi-head attention over rank-2 <c>[S, d]</c> sequences, on the tape.</summary>
internal static class MultiHeadAttentionMath<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// softmax(q k^T / sqrt(dh) + mask) v over <paramref name="heads"/> heads. The causal mask blocks key j &gt; i.
    /// With <paramref name="xPos"/>, q and k are rotated and scaled by xPos before the product.
    /// </summary>
    public static Tensor<T> Attend(Tensor<T> q, Tensor<T> k, Tensor<T> v, int heads, bool causal, bool xPos)
    {
        var engine = AiDotNetEngine.Current;
        int sq = q.Shape[0], sk = k.Shape[0], d = q.Shape[1], dh = d / heads;
        Tensor<T> Split(Tensor<T> x, int s) => engine.TensorPermute(engine.Reshape(x, new[] { s, heads, dh }), new[] { 1, 0, 2 });
        var qh = Split(q, sq);
        var kh = Split(k, sk);
        var vh = Split(v, sk);
        if (xPos)
        {
            qh = XPos(qh, heads, sq, dh, downscale: false);
            kh = XPos(kh, heads, sk, dh, downscale: true);
        }
        var scores = engine.TensorMultiplyScalar(
            engine.TensorBatchMatMul<T>(qh, engine.TensorPermute(kh, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(dh)));
        if (causal)
        {
            var mask = new Tensor<T>(new[] { heads, sq, sk });
            T blocked = NumOps.FromDouble(-1e9);
            int shift = sk - sq; // queries are the LAST sq positions of the keys
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < sq; i++)
                    for (int j = i + shift + 1; j < sk; j++) mask[h, i, j] = blocked;
            scores = engine.TensorAdd(scores, mask);
        }
        var context = engine.TensorBatchMatMul<T>(engine.TensorSoftmax(scores, 2), vh);   // [H, Sq, dh]
        return engine.Reshape(engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { sq, d });
    }

    /// <summary>
    /// xPos (Sun et al. 2022, torchscale): interleaved rotary rotation by angle <c>n theta_i</c>, with
    /// <c>theta_i = 10000^(-2i/dh)</c>, scaled by <c>zeta_i^(+-n/512)</c>, where
    /// <c>zeta_i = (2i/dh + 0.4) / 1.4</c>. Queries are scaled up and keys down, so the product decays with
    /// <c>|n - m|</c>.
    /// </summary>
    private static Tensor<T> XPos(Tensor<T> x, int heads, int s, int dh, bool downscale)
    {
        var engine = AiDotNetEngine.Current;
        var cos = new Tensor<T>(new[] { 1, s, dh });
        var sin = new Tensor<T>(new[] { 1, s, dh });
        for (int n = 0; n < s; n++)
            for (int i = 0; i < dh / 2; i++)
            {
                double theta = Math.Pow(10000, -2.0 * i / dh);
                double zeta = ((2.0 * i / dh) + 0.4) / 1.4;
                double scale = Math.Pow(zeta, (downscale ? -1.0 : 1.0) * n / 512.0);
                double c = Math.Cos(n * theta) * scale, sn = Math.Sin(n * theta) * scale;
                cos[0, n, 2 * i] = NumOps.FromDouble(c);
                cos[0, n, (2 * i) + 1] = NumOps.FromDouble(c);
                sin[0, n, 2 * i] = NumOps.FromDouble(sn);
                sin[0, n, (2 * i) + 1] = NumOps.FromDouble(sn);
            }
        // rotate_every_two: (x0, x1) -> (-x1, x0), as a constant [dh, dh] map.
        var rotate = new Tensor<T>(new[] { dh, dh });
        for (int i = 0; i < dh / 2; i++)
        {
            rotate[(2 * i) + 1, 2 * i] = NumOps.FromDouble(-1);
            rotate[2 * i, (2 * i) + 1] = NumOps.One;
        }
        var rotated = engine.Reshape(engine.TensorMatMul(engine.Reshape(x, new[] { heads * s, dh }), rotate), new[] { heads, s, dh });
        var shape = new[] { heads, s, dh };
        return engine.TensorAdd(
            engine.TensorMultiply(x, engine.TensorBroadcastTo(cos, shape)),
            engine.TensorMultiply(rotated, engine.TensorBroadcastTo(sin, shape)));
    }

    /// <summary><c>x / sqrt(sum x^2 + 1e-12)</c> per row, on the tape (torch F.normalize).</summary>
    public static Tensor<T> L2NormalizeRows(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var norm = engine.TensorSqrt(engine.TensorAddScalar(
            engine.ReduceSum(engine.TensorMultiply(x, x), new[] { 1 }, keepDims: true), NumOps.FromDouble(1e-12)));
        return engine.TensorDivide(x, engine.TensorBroadcastTo(norm, x._shape));
    }

    /// <summary>QuickGELU, CLIP's activation: <c>x * sigmoid(1.702 x)</c>.</summary>
    public static Tensor<T> QuickGelu(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorMultiply(x, engine.Sigmoid(engine.TensorMultiplyScalar(x, NumOps.FromDouble(1.702))));
    }
}

/// <summary>
/// CLIP's vision transformer (Radford et al. 2021) as used by KOSMOS (reference Kosmos2VisionTransformer):
/// <list type="bullet">
/// <item>A <c>patch x patch</c> stride-<c>patch</c> convolution, plus a class token and learned position
/// embeddings.</item>
/// <item>pre_layrnorm, then pre-LN encoder blocks: attention and a QuickGELU MLP at 4x width, each with a
/// residual.</item>
/// <item>post_layernorm over ALL tokens, then L2 normalization per token. KOSMOS feeds every token, not
/// only the pooled class token, to its resampler.</item>
/// </list>
/// </summary>
/// <remarks>
/// <para>Input: one image <c>[3, H, W]</c> (or <c>[1, 3, H, W]</c>) with H = W = <c>imageSize</c>. Output:
/// <c>[1 + (imageSize / patch)^2, hidden]</c>. The patch convolution here carries a bias; CLIP's has none.</para>
/// <para><b>For Beginners:</b> This cuts the image into small squares, turns each into a vector, and lets them
/// exchange information, so each vector describes its square in the context of the whole image.</para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "3, 8, 8", TestConstructorArgs = "8, 1, 2, 4, 8")]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output, Note = "Class token then one row per patch.")]
[AutoParameters]
public partial class ClipVisionTransformerLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _hidden;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _patchSize;
    private readonly int _imageSize;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _classEmbedding;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _positionEmbedding;

    private readonly ConvolutionalLayer<T> _patchEmbedding;
    [SubLayerInput("_hidden")]
    private readonly LayerNormalizationLayer<T> _preNorm;
    [SubLayerInput("_hidden")]
    private readonly List<LayerNormalizationLayer<T>> _norm1 = new();
    [SubLayerInput("_hidden")]
    private readonly List<DenseLayer<T>> _query = new();
    [SubLayerInput("_hidden")]
    private readonly List<DenseLayer<T>> _key = new();
    [SubLayerInput("_hidden")]
    private readonly List<DenseLayer<T>> _value = new();
    [SubLayerInput("_hidden")]
    private readonly List<DenseLayer<T>> _output = new();
    [SubLayerInput("_hidden")]
    private readonly List<LayerNormalizationLayer<T>> _norm2 = new();
    [SubLayerInput("_hidden")]
    private readonly List<DenseLayer<T>> _fc1 = new();
    [SubLayerInput("_mlpDim")]
    private readonly List<DenseLayer<T>> _fc2 = new();
    [SubLayerInput("_hidden")]
    private readonly LayerNormalizationLayer<T> _postNorm;
    private readonly int _mlpDim;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>Creates the encoder. CLIP ViT-L/14 (KOSMOS): hidden 1024, 24 layers, 16 heads, patch 14, 224 px.</summary>
    public ClipVisionTransformerLayer([LayerState] int hidden, [LayerState] int numLayers, [LayerState] int numHeads,
        [LayerState] int patchSize, [LayerState] int imageSize)
        : base(new[] { 3, imageSize, imageSize }, new[] { 1 + ((imageSize / Math.Max(1, patchSize)) * (imageSize / Math.Max(1, patchSize))), hidden })
    {
        if (hidden <= 0 || numHeads <= 0 || hidden % numHeads != 0)
            throw new ArgumentException($"hidden ({hidden}) must be a positive multiple of numHeads ({numHeads}).", nameof(numHeads));
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (patchSize <= 0 || imageSize <= 0 || imageSize % patchSize != 0)
            throw new ArgumentException($"imageSize ({imageSize}) must be a positive multiple of patchSize ({patchSize}).", nameof(patchSize));
        _hidden = hidden;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _patchSize = patchSize;
        _imageSize = imageSize;
        _mlpDim = 4 * hidden;
        int tokens = 1 + ((imageSize / patchSize) * (imageSize / patchSize));

        // Kosmos2 _init_weights: class embedding N(0, d^-0.5); position embedding N(0, 0.02).
        var random = LayerInitializationSeedScope.NextRandom();
        _classEmbedding = Normal(new[] { 1, hidden }, Math.Pow(hidden, -0.5), random);
        _positionEmbedding = Normal(new[] { tokens, hidden }, 0.02, random);
        RegisterTrainableParameter(_classEmbedding, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_positionEmbedding, PersistentTensorRole.Weights);

        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        _patchEmbedding = new ConvolutionalLayer<T>(hidden, patchSize, patchSize, 0, identity);
        _preNorm = new LayerNormalizationLayer<T>();
        for (int i = 0; i < numLayers; i++)
        {
            _norm1.Add(new LayerNormalizationLayer<T>());
            _query.Add(new DenseLayer<T>(hidden, identity));
            _key.Add(new DenseLayer<T>(hidden, identity));
            _value.Add(new DenseLayer<T>(hidden, identity));
            _output.Add(new DenseLayer<T>(hidden, identity));
            _norm2.Add(new LayerNormalizationLayer<T>());
            _fc1.Add(new DenseLayer<T>(_mlpDim, identity));
            _fc2.Add(new DenseLayer<T>(hidden, identity));
        }
        _postNorm = new LayerNormalizationLayer<T>();
        foreach (var layer in SubLayers()) RegisterSubLayer(layer);
    }

    private IEnumerable<LayerBase<T>> SubLayers()
    {
        yield return _patchEmbedding;
        yield return _preNorm;
        for (int i = 0; i < _numLayers; i++)
        {
            yield return _norm1[i]; yield return _query[i]; yield return _key[i]; yield return _value[i]; yield return _output[i];
            yield return _norm2[i]; yield return _fc1[i]; yield return _fc2[i];
        }
        yield return _postNorm;
    }

    private Tensor<T> Normal(int[] shape, double std, Random random)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            tensor[i] = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * std);
        }
        return tensor;
    }

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => (inputRank is 3 or 4)
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Fixed(1 + ((_imageSize / _patchSize) * (_imageSize / _patchSize)))),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_hidden)),
        }
        : null;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var image = input.Rank == 3 ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] }) : input;
        if (image.Rank != 4 || image.Shape[0] != 1 || image.Shape[1] != 3 || image.Shape[2] != _imageSize || image.Shape[3] != _imageSize)
            throw new ArgumentException(
                $"ClipVisionTransformerLayer expects one image of shape [3, {_imageSize}, {_imageSize}]; got shape [{string.Join(", ", input.Shape.ToArray())}].",
                nameof(input));
        int grid = _imageSize / _patchSize, patches = grid * grid;
        var patchMap = _patchEmbedding.Forward(image);                                                  // [1, d, g, g]
        var patchTokens = Engine.TensorPermute(Engine.Reshape(patchMap, new[] { _hidden, patches }), new[] { 1, 0 });
        var x = Engine.TensorAdd(Engine.TensorConcatenate(new[] { _classEmbedding, patchTokens }, 0), _positionEmbedding);
        x = _preNorm.Forward(x);
        for (int i = 0; i < _numLayers; i++)
        {
            var h = _norm1[i].Forward(x);
            var attended = MultiHeadAttentionMath<T>.Attend(_query[i].Forward(h), _key[i].Forward(h), _value[i].Forward(h), _numHeads, causal: false, xPos: false);
            x = Engine.TensorAdd(x, _output[i].Forward(attended));
            var m = _fc2[i].Forward(MultiHeadAttentionMath<T>.QuickGelu(_fc1[i].Forward(_norm2[i].Forward(x))));
            x = Engine.TensorAdd(x, m);
        }
        return MultiHeadAttentionMath<T>.L2NormalizeRows(_postNorm.Forward(x));
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Hidden"] = _hidden.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumLayers"] = _numLayers.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["PatchSize"] = _patchSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["ImageSize"] = _imageSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var layer in SubLayers()) layer.ResetState(); }
}

/// <summary>
/// Turns image tokens into a fixed number of image embeddings for the language model.
/// </summary>
/// <remarks>
/// <para>
/// There are two forms:
/// <list type="bullet">
/// <item>KOSMOS-2 (<c>perceiver = false</c>; reference <c>Kosmos2ImageToTextProjection</c>): project the
/// tokens to the text width. Then <c>numLatents</c> learned queries attend once, with no residual and no
/// LayerNorm, to <c>[image tokens; queries]</c>.</item>
/// <item>KOSMOS-1 (<c>perceiver = true</c>): a Flamingo perceiver resampler (Alayrac et al. 2022). Each of
/// <c>depth</c> blocks runs <c>latents += Attn(LN(latents), LN([x; latents]))</c>, then
/// <c>latents += FFN(LN(latents))</c> with a 4x GELU MLP. A final LayerNorm follows.</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> A fixed set of learned "questions" reads the image and summarizes it into the
/// same small number of vectors every time, so an image always takes the same room in the text.</para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 2, Cost = ComputeCost.Medium, TestInputShape = "5, 8", TestConstructorArgs = "8, 8, 4, 2, 1, true")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input, Note = "Image tokens.")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output, Note = "numLatents image embeddings.")]
[AutoParameters]
public partial class KosmosImageResamplerLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _visionDim;
    private readonly int _dim;
    private readonly int _numLatents;
    private readonly int _numHeads;
    private readonly int _depth;
    private readonly bool _perceiver;
    private readonly int _ffnDim;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _latents;

    [SubLayerInput("_visionDim")]
    private readonly DenseLayer<T> _projection;
    [SubLayerInput("_dim")]
    private readonly List<DenseLayer<T>> _query = new();
    [SubLayerInput("_dim")]
    private readonly List<DenseLayer<T>> _key = new();
    [SubLayerInput("_dim")]
    private readonly List<DenseLayer<T>> _value = new();
    [SubLayerInput("_dim")]
    private readonly List<DenseLayer<T>> _output = new();
    [SubLayerInput("_dim")]
    private readonly List<LayerNormalizationLayer<T>> _mediaNorm = new();
    [SubLayerInput("_dim")]
    private readonly List<LayerNormalizationLayer<T>> _latentNorm = new();
    [SubLayerInput("_dim")]
    private readonly List<LayerNormalizationLayer<T>> _ffnNorm = new();
    [SubLayerInput("_dim")]
    private readonly List<DenseLayer<T>> _fc1 = new();
    [SubLayerInput("_ffnDim")]
    private readonly List<DenseLayer<T>> _fc2 = new();
    [SubLayerInput("_dim")]
    private readonly List<LayerNormalizationLayer<T>> _finalNorm = new();

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>Creates the resampler. KOSMOS-2: vision 1024 -> text 2048, 64 latents, 32 heads, one attention.</summary>
    public KosmosImageResamplerLayer([LayerState] int visionDim, [LayerState] int dim, [LayerState] int numLatents,
        [LayerState] int numHeads, [LayerState] int depth, [LayerState] bool perceiver)
        : base(new[] { -1, visionDim }, new[] { numLatents, dim })
    {
        if (visionDim <= 0) throw new ArgumentOutOfRangeException(nameof(visionDim));
        if (dim <= 0 || numHeads <= 0 || dim % numHeads != 0)
            throw new ArgumentException($"dim ({dim}) must be a positive multiple of numHeads ({numHeads}).", nameof(numHeads));
        if (numLatents <= 0) throw new ArgumentOutOfRangeException(nameof(numLatents));
        if (depth <= 0) throw new ArgumentOutOfRangeException(nameof(depth));
        _visionDim = visionDim;
        _dim = dim;
        _numLatents = numLatents;
        _numHeads = numHeads;
        _depth = perceiver ? depth : 1;
        _perceiver = perceiver;
        _ffnDim = 4 * dim;

        // latent_query = nn.Parameter(torch.randn(numLatents, dim)).
        var random = LayerInitializationSeedScope.NextRandom();
        _latents = new Tensor<T>(new[] { numLatents, dim });
        for (int i = 0; i < _latents.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            _latents[i] = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        RegisterTrainableParameter(_latents, PersistentTensorRole.Weights);

        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        _projection = new DenseLayer<T>(dim, identity);
        for (int i = 0; i < _depth; i++)
        {
            _query.Add(new DenseLayer<T>(dim, identity));
            _key.Add(new DenseLayer<T>(dim, identity));
            _value.Add(new DenseLayer<T>(dim, identity));
            _output.Add(new DenseLayer<T>(dim, identity));
            if (perceiver)
            {
                _mediaNorm.Add(new LayerNormalizationLayer<T>());
                _latentNorm.Add(new LayerNormalizationLayer<T>());
                _ffnNorm.Add(new LayerNormalizationLayer<T>());
                _fc1.Add(new DenseLayer<T>(_ffnDim, (IActivationFunction<T>)new GELUActivation<T>()));
                _fc2.Add(new DenseLayer<T>(dim, identity));
            }
        }
        if (perceiver) _finalNorm.Add(new LayerNormalizationLayer<T>());
        foreach (var layer in SubLayers()) RegisterSubLayer(layer);
    }

    private IEnumerable<LayerBase<T>> SubLayers()
    {
        yield return _projection;
        for (int i = 0; i < _depth; i++)
        {
            yield return _query[i]; yield return _key[i]; yield return _value[i]; yield return _output[i];
            if (_perceiver)
            {
                yield return _mediaNorm[i]; yield return _latentNorm[i]; yield return _ffnNorm[i]; yield return _fc1[i]; yield return _fc2[i];
            }
        }
        foreach (var norm in _finalNorm) yield return norm;
    }

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 2
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Fixed(_numLatents)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_dim)),
        }
        : null;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 2 || input.Shape[1] != _visionDim)
            throw new ArgumentException(
                $"KosmosImageResamplerLayer expects image tokens of shape [N, {_visionDim}]; got shape [{string.Join(", ", input.Shape.ToArray())}].", nameof(input));
        var media = _projection.Forward(input);
        var latents = _latents;
        for (int i = 0; i < _depth; i++)
        {
            if (!_perceiver)
            {
                var keys = Engine.TensorConcatenate(new[] { media, latents }, 0);
                return _output[i].Forward(MultiHeadAttentionMath<T>.Attend(
                    _query[i].Forward(latents), _key[i].Forward(keys), _value[i].Forward(keys), _numHeads, causal: false, xPos: false));
            }

            var q = _latentNorm[i].Forward(latents);
            var kv = Engine.TensorConcatenate(new[] { _mediaNorm[i].Forward(media), q }, 0);
            latents = Engine.TensorAdd(latents, _output[i].Forward(MultiHeadAttentionMath<T>.Attend(
                _query[i].Forward(q), _key[i].Forward(kv), _value[i].Forward(kv), _numHeads, causal: false, xPos: false)));
            latents = Engine.TensorAdd(latents, _fc2[i].Forward(_fc1[i].Forward(_ffnNorm[i].Forward(latents))));
        }
        return _finalNorm[0].Forward(latents);
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["VisionDim"] = _visionDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Dim"] = _dim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumLatents"] = _numLatents.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Depth"] = _depth.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Perceiver"] = _perceiver.ToString();
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var layer in SubLayers()) layer.ResetState(); }
}

/// <summary>
/// KOSMOS's MAGNETO causal language model (reference Kosmos2TextTransformer + tied lm_head). It runs these
/// steps:
/// <list type="bullet">
/// <item>Token embeddings are looked up, image embeddings are written into their slots, and everything is
/// scaled by <c>sqrt(d)</c>.</item>
/// <item>Positions: fairseq sinusoidal positions starting at 2, or xPos inside attention.</item>
/// <item>Pre-LN blocks, where attention carries a Sub-LN on its output before <c>out_proj</c>, and the FFN
/// is fc1, GELU, a Sub-LN, then fc2.</item>
/// <item>A final LayerNorm, then logits from the tied embedding matrix.</item>
/// </list>
/// </summary>
/// <remarks>
/// <para>The single-input forward takes token ids <c>[S]</c> and returns next-token logits <c>[S, vocab]</c>.</para>
/// <para><b>For Beginners:</b> This is the language model that writes text one token at a time, reading the
/// image's embeddings as if they were words at the start of the sentence.</para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 1, Cost = ComputeCost.High, TestInputShape = "5", TestConstructorArgs = "20, 8, 1, 2, 16, 32, 0")]
[TensorPort("input", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices, Role = TensorPortRole.TokenIds, MaxExclusiveMember = "_vocabSize")]
[TensorPort("output", TensorPortDirection.Output, LayerInputDomainKind.Continuous, Role = TensorPortRole.Features)]
[TensorLayout(TensorAxis.Time, Direction = TensorLayoutDirection.Input, Note = "Token ids.")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output, Note = "Next-token logits.")]
[AutoParameters]
public partial class MagnetoDecoderLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _vocabSize;
    private readonly int _dim;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _ffnDim;
    private readonly int _maxPositions;
    private readonly KosmosPositionEncoding _positions;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _embedTokens;

    [SubLayerInput("_dim")]
    private readonly List<LayerNormalizationLayer<T>> _attnNorm = new();
    [SubLayerInput("_dim")]
    private readonly List<DenseLayer<T>> _query = new();
    [SubLayerInput("_dim")]
    private readonly List<DenseLayer<T>> _key = new();
    [SubLayerInput("_dim")]
    private readonly List<DenseLayer<T>> _value = new();
    [SubLayerInput("_dim")]
    private readonly List<LayerNormalizationLayer<T>> _innerAttnNorm = new();
    [SubLayerInput("_dim")]
    private readonly List<DenseLayer<T>> _output = new();
    [SubLayerInput("_dim")]
    private readonly List<LayerNormalizationLayer<T>> _ffnPreNorm = new();
    [SubLayerInput("_dim")]
    private readonly List<DenseLayer<T>> _fc1 = new();
    [SubLayerInput("_ffnDim")]
    private readonly List<LayerNormalizationLayer<T>> _ffnInnerNorm = new();
    [SubLayerInput("_ffnDim")]
    private readonly List<DenseLayer<T>> _fc2 = new();
    [SubLayerInput("_dim")]
    private readonly LayerNormalizationLayer<T> _finalNorm;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>Creates the decoder. KOSMOS: 24 layers, 2048 wide, 32 heads, FFN 8192, 2048 positions.</summary>
    public MagnetoDecoderLayer([LayerState] int vocabSize, [LayerState] int dim, [LayerState] int numLayers,
        [LayerState] int numHeads, [LayerState] int ffnDim, [LayerState] int maxPositions, [LayerState] KosmosPositionEncoding positions)
        : base(new[] { -1 }, new[] { -1, vocabSize })
    {
        if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (dim <= 0 || numHeads <= 0 || dim % numHeads != 0 || (dim / numHeads) % 2 != 0)
            throw new ArgumentException($"dim ({dim}) must split into an even head width over numHeads ({numHeads}).", nameof(numHeads));
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (ffnDim <= 0) throw new ArgumentOutOfRangeException(nameof(ffnDim));
        if (maxPositions <= 0) throw new ArgumentOutOfRangeException(nameof(maxPositions));
        _vocabSize = vocabSize;
        _dim = dim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _ffnDim = ffnDim;
        _maxPositions = maxPositions;
        _positions = positions;

        // Text init_std 0.02; the padding row (pad_token_id 1) of nn.Embedding is zero.
        var random = LayerInitializationSeedScope.NextRandom();
        _embedTokens = new Tensor<T>(new[] { vocabSize, dim });
        for (int i = 0; i < _embedTokens.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            _embedTokens[i] = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * 0.02);
        }
        if (vocabSize > 1) for (int c = 0; c < dim; c++) _embedTokens[1, c] = NumOps.Zero;
        RegisterTrainableParameter(_embedTokens, PersistentTensorRole.Weights);

        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        for (int i = 0; i < numLayers; i++)
        {
            _attnNorm.Add(new LayerNormalizationLayer<T>());
            _query.Add(new DenseLayer<T>(dim, identity));
            _key.Add(new DenseLayer<T>(dim, identity));
            _value.Add(new DenseLayer<T>(dim, identity));
            _innerAttnNorm.Add(new LayerNormalizationLayer<T>());
            _output.Add(new DenseLayer<T>(dim, identity));
            _ffnPreNorm.Add(new LayerNormalizationLayer<T>());
            _fc1.Add(new DenseLayer<T>(ffnDim, (IActivationFunction<T>)new GELUActivation<T>()));
            _ffnInnerNorm.Add(new LayerNormalizationLayer<T>());
            _fc2.Add(new DenseLayer<T>(dim, identity));
        }
        _finalNorm = new LayerNormalizationLayer<T>();
        foreach (var layer in SubLayers()) RegisterSubLayer(layer);
    }

    private IEnumerable<LayerBase<T>> SubLayers()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            yield return _attnNorm[i]; yield return _query[i]; yield return _key[i]; yield return _value[i];
            yield return _innerAttnNorm[i]; yield return _output[i];
            yield return _ffnPreNorm[i]; yield return _fc1[i]; yield return _ffnInnerNorm[i]; yield return _fc2[i];
        }
        yield return _finalNorm;
    }

    /// <summary>Vocabulary size (logit width).</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Model width.</summary>
    public int Dim => _dim;

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 1
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_vocabSize)),
        }
        : null;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 1)
            throw new ArgumentException($"MagnetoDecoderLayer expects token ids of shape [S]; got rank {input.Rank}.", nameof(input));
        var ids = new int[input.Shape[0]];
        for (int i = 0; i < ids.Length; i++)
            ids[i] = Math.Min(Math.Max((int)Math.Round(NumOps.ToDouble(input[i])), 0), _vocabSize - 1);
        return Forward(ids, null, 0);
    }

    /// <summary>
    /// Next-token logits <c>[S, vocab]</c> for <paramref name="ids"/>. When <paramref name="imageEmbeddings"/>
    /// (<c>[K, d]</c>) is given, it replaces the token embeddings at positions <c>imageStart .. imageStart + K - 1</c>
    /// (reference image_embeds_position_mask).
    /// </summary>
    internal Tensor<T> Forward(int[] ids, Tensor<T>? imageEmbeddings, int imageStart)
    {
        int s = ids.Length;
        if (s == 0) throw new ArgumentException("MagnetoDecoderLayer needs at least one token.", nameof(ids));
        if (s > _maxPositions) throw new ArgumentException($"The sequence ({s}) exceeds the {_maxPositions} positions.", nameof(ids));
        var x = CvTensorOps<T>.Select(_embedTokens, ids, 0);                                             // [S, d]
        if (imageEmbeddings is not null)
        {
            int k = imageEmbeddings.Shape[0];
            if (imageStart < 0 || imageStart + k > s)
                throw new ArgumentException($"The {k} image slots starting at {imageStart} do not fit a {s}-token sequence.", nameof(imageStart));
            var parts = new List<Tensor<T>>();
            if (imageStart > 0) parts.Add(Engine.TensorSlice(x, new[] { 0, 0 }, new[] { imageStart, _dim }));
            parts.Add(imageEmbeddings);
            if (imageStart + k < s) parts.Add(Engine.TensorSlice(x, new[] { imageStart + k, 0 }, new[] { s - imageStart - k, _dim }));
            x = Engine.TensorConcatenate(parts.ToArray(), 0);
        }
        x = Engine.TensorMultiplyScalar(x, NumOps.FromDouble(Math.Sqrt(_dim)));
        if (_positions == KosmosPositionEncoding.Sinusoidal) x = Engine.TensorAdd(x, Sinusoid(s));

        bool xPos = _positions == KosmosPositionEncoding.XPos;
        for (int i = 0; i < _numLayers; i++)
        {
            var h = _attnNorm[i].Forward(x);
            var attended = MultiHeadAttentionMath<T>.Attend(_query[i].Forward(h), _key[i].Forward(h), _value[i].Forward(h), _numHeads, causal: true, xPos);
            x = Engine.TensorAdd(x, _output[i].Forward(_innerAttnNorm[i].Forward(attended)));
            var f = _fc2[i].Forward(_ffnInnerNorm[i].Forward(_fc1[i].Forward(_ffnPreNorm[i].Forward(x))));
            x = Engine.TensorAdd(x, f);
        }
        // lm_head is tied to embed_tokens: logits = h E^T.
        return Engine.TensorMatMul(_finalNorm.Forward(x), Engine.TensorTranspose(_embedTokens));
    }

    /// <summary>
    /// Kosmos2TextSinusoidalPositionalEmbedding for unpadded positions <c>2 .. S + 1</c>:
    /// <c>[sin(p f_j), cos(p f_j)]</c> with <c>f_j = exp(-j log(10000) / (d/2 - 1))</c>.
    /// </summary>
    private Tensor<T> Sinusoid(int s)
    {
        int half = _dim / 2;
        double step = Math.Log(10000) / Math.Max(1, half - 1);
        var pe = new Tensor<T>(new[] { s, _dim });
        for (int i = 0; i < s; i++)
        {
            double p = i + 2;
            for (int j = 0; j < half; j++)
            {
                double angle = p * Math.Exp(-j * step);
                pe[i, j] = NumOps.FromDouble(Math.Sin(angle));
                pe[i, half + j] = NumOps.FromDouble(Math.Cos(angle));
            }
        }
        return pe;
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["VocabSize"] = _vocabSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Dim"] = _dim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumLayers"] = _numLayers.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["FfnDim"] = _ffnDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MaxPositions"] = _maxPositions.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Positions"] = _positions.ToString();
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var layer in SubLayers()) layer.ResetState(); }
}
