using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.ComputerVision;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// T5 relative-position bucketing (Raffel et al. 2020; Mesh TensorFlow <c>_relative_position_bucket</c>).
/// </summary>
/// <remarks>
/// Half of the buckets hold exact small distances. The rest grow logarithmically up to <c>maxDistance</c>, and every
/// larger distance shares the last bucket. Bidirectional bucketing gives positive and negative distances separate
/// halves. Unidirectional (causal) bucketing maps every future position to bucket 0, which the causal mask then hides.
/// </remarks>
internal static class T5RelativePositionBuckets
{
    public static int Bucket(long relativePosition, bool bidirectional, int numBuckets, int maxDistance)
    {
        int bucket = 0;
        int buckets = numBuckets;
        long distance;
        if (bidirectional)
        {
            buckets /= 2;
            if (relativePosition > 0) bucket += buckets;
            distance = Math.Abs(relativePosition);
        }
        else
        {
            distance = -Math.Min(relativePosition, 0);
        }

        int maxExact = buckets / 2;
        if (distance < maxExact) return bucket + (int)distance;
        // torch's .to(torch.long) truncates toward zero; the value is non-negative here.
        int large = maxExact + (int)(Math.Log((double)distance / maxExact) / Math.Log((double)maxDistance / maxExact) * (buckets - maxExact));
        return bucket + Math.Min(large, buckets - 1);
    }

    /// <summary>Looks each bucket up in a <c>[numBuckets, heads]</c> table and returns the bias as <c>[heads, Sq, Sk]</c>.</summary>
    public static Tensor<T> Bias<T>(Tensor<T> table, int[] buckets, int queries, int keys)
    {
        var engine = AiDotNetEngine.Current;
        int heads = table.Shape[1];
        var rows = CvTensorOps<T>.Select(table, buckets, 0);                                   // [Sq*Sk, H]
        return engine.TensorPermute(engine.Reshape(rows, new[] { queries, keys, heads }), new[] { 2, 0, 1 });
    }
}

/// <summary>
/// T5 multi-head attention: bias-free projections, no <c>1/sqrt(d)</c> score scaling, and an additive position bias.
/// </summary>
/// <remarks>
/// T5 folds the score scaling into its initialisation: queries are drawn from N(0, (d·d_kv)^-1/2) rather than
/// being divided at run time. The position bias is supplied by the caller. It is computed once per stack and shared
/// by every block, as in the reference.
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = false, ExpectedInputRank = 2, Cost = ComputeCost.Medium, TestInputShape = "4, 8", TestConstructorArgs = "8, 2, 4")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class T5AttentionLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _keyValueDim;
    private readonly int _inner;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _query;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _key;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _value;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _output;

    public override bool SupportsTraining => true;

    public T5AttentionLayer([LayerState] int dim, [LayerState] int numHeads, [LayerState] int keyValueDim)
        : base(new[] { -1, dim }, new[] { -1, dim })
    {
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (keyValueDim <= 0) throw new ArgumentOutOfRangeException(nameof(keyValueDim));
        _dim = dim;
        _numHeads = numHeads;
        _keyValueDim = keyValueDim;
        _inner = numHeads * keyValueDim;

        // UdopPreTrainedModel._init_weights (factor 1.0).
        var random = LayerInitializationSeedScope.NextRandom();
        _query = Normal(new[] { dim, _inner }, Math.Pow(dim * keyValueDim, -0.5), random);
        _key = Normal(new[] { dim, _inner }, Math.Pow(dim, -0.5), random);
        _value = Normal(new[] { dim, _inner }, Math.Pow(dim, -0.5), random);
        _output = Normal(new[] { _inner, dim }, Math.Pow(_inner, -0.5), random);
        RegisterTrainableParameter(_query, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_key, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_value, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_output, PersistentTensorRole.Weights);
    }

    public int NumHeads => _numHeads;

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 2
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_dim)),
        }
        : null;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 2 || input.Shape[1] != _dim)
            throw new ArgumentException(
                $"T5AttentionLayer expects features of shape [S, {_dim}]; got shape [{string.Join(", ", input.Shape.ToArray())}].", nameof(input));
        return Forward(input, input, null, causal: false);
    }

    /// <summary>Attends from <paramref name="x"/> to <paramref name="memory"/> (the same tensor for self-attention).</summary>
    internal Tensor<T> Forward(Tensor<T> x, Tensor<T> memory, Tensor<T>? positionBias, bool causal)
    {
        int sq = x.Shape[0], sk = memory.Shape[0];
        Tensor<T> Heads(Tensor<T> projected, int s) =>
            Engine.TensorPermute(Engine.Reshape(projected, new[] { s, _numHeads, _keyValueDim }), new[] { 1, 0, 2 });
        var q = Heads(Engine.TensorMatMul(x, _query), sq);
        var k = Heads(Engine.TensorMatMul(memory, _key), sk);
        var v = Heads(Engine.TensorMatMul(memory, _value), sk);
        var scores = Engine.TensorBatchMatMul<T>(q, Engine.TensorPermute(k, new[] { 0, 2, 1 }));   // [H, Sq, Sk]
        if (positionBias is not null) scores = Engine.TensorAdd(scores, positionBias);
        if (causal)
        {
            var mask = new Tensor<T>(new[] { _numHeads, sq, sk });
            T blocked = NumOps.FromDouble(-1e9);
            int shift = sk - sq;
            for (int h = 0; h < _numHeads; h++)
                for (int i = 0; i < sq; i++)
                    for (int j = i + shift + 1; j < sk; j++) mask[h, i, j] = blocked;
            scores = Engine.TensorAdd(scores, mask);
        }
        var context = Engine.TensorBatchMatMul<T>(Engine.TensorSoftmax(scores, 2), v);             // [H, Sq, dkv]
        var merged = Engine.Reshape(Engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { sq, _inner });
        return Engine.TensorMatMul(merged, _output);
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

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Dim"] = _dim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["KeyValueDim"] = _keyValueDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    public override void ResetState() { }
}

/// <summary>T5 feed-forward block (<c>DenseReluDense</c>): bias-free <c>wo(relu(wi(x)))</c>.</summary>
[LayerCategory(LayerCategory.Dense)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = false, ExpectedInputRank = 2, Cost = ComputeCost.Medium, TestInputShape = "4, 8", TestConstructorArgs = "8, 16")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class T5FeedForwardLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _dim;
    private readonly int _ffDim;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _wi;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _wo;

    public override bool SupportsTraining => true;

    public T5FeedForwardLayer([LayerState] int dim, [LayerState] int ffDim)
        : base(new[] { -1, dim }, new[] { -1, dim })
    {
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (ffDim <= 0) throw new ArgumentOutOfRangeException(nameof(ffDim));
        _dim = dim;
        _ffDim = ffDim;
        var random = LayerInitializationSeedScope.NextRandom();
        _wi = Normal(new[] { dim, ffDim }, Math.Pow(dim, -0.5), random);
        _wo = Normal(new[] { ffDim, dim }, Math.Pow(ffDim, -0.5), random);
        RegisterTrainableParameter(_wi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_wo, PersistentTensorRole.Weights);
    }

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 2
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_dim)),
        }
        : null;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 2 || input.Shape[1] != _dim)
            throw new ArgumentException(
                $"T5FeedForwardLayer expects features of shape [S, {_dim}]; got shape [{string.Join(", ", input.Shape.ToArray())}].", nameof(input));
        return Engine.TensorMatMul(Engine.ReLU(Engine.TensorMatMul(input, _wi)), _wo);
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

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Dim"] = _dim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["FfDim"] = _ffDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    public override void ResetState() { }
}

/// <summary>
/// UDOP's unified vision-text-layout transformer (Tang et al. 2023; HF <c>UdopForConditionalGeneration</c>): a T5
/// encoder-decoder whose encoder sees text, image patches and 2-D layout together.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item><b>Layout-induced fusion.</b> Each token's embedding gets the embedding of the 16x16 image patch that
/// contains its box centre. Tokens whose box mean is exactly 0 or 1, such as task prompts, are skipped. Patches
/// that no token claimed are appended after the text, each with its grid-cell box.</item>
/// <item><b>2-D cell embedding.</b> Left/right corners index an x table and top/bottom corners index a y table.
/// The four lookups are summed onto the encoder input.</item>
/// <item><b>Encoder position bias.</b> Three bucketed biases are summed: T5 1-D over sequence order, plus horizontal
/// and vertical biases over box-centre distances scaled by 100. The bias is computed once and shared by every
/// block.</item>
/// <item><b>Decoder.</b> A T5 causal 1-D bias from block 0, cross-attention without bias, and a head tied to the
/// shared embedding with the <c>d^-1/2</c> rescale.</item>
/// </list>
/// All blocks are pre-RMSNorm residual blocks with bias-free projections and a ReLU feed-forward.
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 2, Cost = ComputeCost.High, TestInputShape = "4, 5",
    TestConstructorArgs = "20, 8, 2, 4, 16, 1, 1, 16, 8, 8, 16, 10, 16")]
[TensorPort("input", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices, Role = TensorPortRole.TokenIds, MaxExclusiveMember = "_packedLimit")]
[TensorPort("output", TensorPortDirection.Output, LayerInputDomainKind.Continuous, Role = TensorPortRole.Features)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input, Note = "Packed (token, x0, y0, x1, y1) rows, boxes on the 0-1000 grid.")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output, Note = "Next-token logits for the decoder start token.")]
[AutoParameters]
public partial class UdopTransformerLayer<T> : LayerBase<T>, IShapeContract
{
    /// <summary>Boxes arrive on the 0-1000 grid the UDOP processor emits; the model works on 0-1.</summary>
    public const int CoordinateGrid = 1000;

    /// <summary>Scale the reference applies to 0-1 box distances before bucketing them.</summary>
    private const double LayoutScale = 100;

    private readonly int _vocabSize;
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _keyValueDim;
    private readonly int _ffDim;
    private readonly int _numEncoderLayers;
    private readonly int _numDecoderLayers;
    private readonly int _imageSize;
    private readonly int _patchSize;
    private readonly int _numBuckets;
    private readonly int _maxDistance;
    private readonly int _maxDistance2D;
    private readonly int _max2DPositions;
    private readonly int _packedLimit;
    private readonly double _epsilon;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _shared;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _encoderBias1D;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _encoderBiasHorizontal;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _encoderBiasVertical;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _decoderBias;

    private readonly ConvolutionalLayer<T> _patchEmbedding;
    private readonly EmbeddingLayer<T> _cellX;
    private readonly EmbeddingLayer<T> _cellY;

    [SubLayerInput("_dim")] private readonly List<RMSNormalizationLayer<T>> _encoderSelfNorm = new();
    [SubLayerInput("_dim")] private readonly List<T5AttentionLayer<T>> _encoderSelfAttention = new();
    [SubLayerInput("_dim")] private readonly List<RMSNormalizationLayer<T>> _encoderFeedForwardNorm = new();
    [SubLayerInput("_dim")] private readonly List<T5FeedForwardLayer<T>> _encoderFeedForward = new();
    [SubLayerInput("_dim")] private readonly RMSNormalizationLayer<T> _encoderFinalNorm;
    [SubLayerInput("_dim")] private readonly List<RMSNormalizationLayer<T>> _decoderSelfNorm = new();
    [SubLayerInput("_dim")] private readonly List<T5AttentionLayer<T>> _decoderSelfAttention = new();
    [SubLayerInput("_dim")] private readonly List<RMSNormalizationLayer<T>> _decoderCrossNorm = new();
    [SubLayerInput("_dim")] private readonly List<T5AttentionLayer<T>> _decoderCrossAttention = new();
    [SubLayerInput("_dim")] private readonly List<RMSNormalizationLayer<T>> _decoderFeedForwardNorm = new();
    [SubLayerInput("_dim")] private readonly List<T5FeedForwardLayer<T>> _decoderFeedForward = new();
    [SubLayerInput("_dim")] private readonly RMSNormalizationLayer<T> _decoderFinalNorm;

    public override bool SupportsTraining => true;

    public UdopTransformerLayer([LayerState] int vocabSize, [LayerState] int dim, [LayerState] int numHeads,
        [LayerState] int keyValueDim, [LayerState] int ffDim, [LayerState] int numEncoderLayers,
        [LayerState] int numDecoderLayers, [LayerState] int imageSize, [LayerState] int patchSize,
        [LayerState] int numBuckets, [LayerState] int maxDistance, [LayerState] int maxDistance2D,
        [LayerState] int max2DPositions, [LayerState] double epsilon = 1e-6)
        : base(new[] { -1, 5 }, new[] { 1, vocabSize })
    {
        if (vocabSize < 2) throw new ArgumentOutOfRangeException(nameof(vocabSize), "UDOP needs at least the pad (0) and EOS (1) tokens.");
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (keyValueDim <= 0) throw new ArgumentOutOfRangeException(nameof(keyValueDim));
        if (ffDim <= 0) throw new ArgumentOutOfRangeException(nameof(ffDim));
        if (numEncoderLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numEncoderLayers));
        if (numDecoderLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numDecoderLayers));
        if (patchSize <= 0 || imageSize <= 0 || imageSize % patchSize != 0)
            throw new ArgumentException($"imageSize ({imageSize}) must be a positive multiple of patchSize ({patchSize}).", nameof(patchSize));
        if (numBuckets < 4) throw new ArgumentOutOfRangeException(nameof(numBuckets), "Bidirectional bucketing needs at least four buckets.");
        if (maxDistance <= numBuckets / 4) throw new ArgumentOutOfRangeException(nameof(maxDistance), "maxDistance must exceed the exact-bucket range.");
        if (maxDistance2D <= numBuckets / 4) throw new ArgumentOutOfRangeException(nameof(maxDistance2D), "maxDistance2D must exceed the exact-bucket range.");
        if (max2DPositions < 2) throw new ArgumentOutOfRangeException(nameof(max2DPositions));
        _vocabSize = vocabSize;
        _dim = dim;
        _numHeads = numHeads;
        _keyValueDim = keyValueDim;
        _ffDim = ffDim;
        _numEncoderLayers = numEncoderLayers;
        _numDecoderLayers = numDecoderLayers;
        _imageSize = imageSize;
        _patchSize = patchSize;
        _numBuckets = numBuckets;
        _maxDistance = maxDistance;
        _maxDistance2D = maxDistance2D;
        _max2DPositions = max2DPositions;
        _packedLimit = Math.Max(vocabSize, CoordinateGrid + 1);
        _epsilon = epsilon;

        // UdopPreTrainedModel._init_weights (factor 1.0): shared N(0, 1); relative biases N(0, d^-1/2).
        var random = LayerInitializationSeedScope.NextRandom();
        _shared = Normal(new[] { vocabSize, dim }, 1.0, random);
        _encoderBias1D = Normal(new[] { numBuckets, numHeads }, Math.Pow(dim, -0.5), random);
        _encoderBiasHorizontal = Normal(new[] { numBuckets, numHeads }, Math.Pow(dim, -0.5), random);
        _encoderBiasVertical = Normal(new[] { numBuckets, numHeads }, Math.Pow(dim, -0.5), random);
        _decoderBias = Normal(new[] { numBuckets, numHeads }, Math.Pow(dim, -0.5), random);
        RegisterTrainableParameter(_shared, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_encoderBias1D, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_encoderBiasHorizontal, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_encoderBiasVertical, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_decoderBias, PersistentTensorRole.Weights);

        _patchEmbedding = new ConvolutionalLayer<T>(dim, patchSize, patchSize, 0, (IActivationFunction<T>)new IdentityActivation<T>());
        // The cell tables index quantised box corners, not the patch features built above.
        _cellX = LayerGraphContract.FromDerivedInput(new EmbeddingLayer<T>(max2DPositions, dim), "boxes");
        _cellY = LayerGraphContract.FromDerivedInput(new EmbeddingLayer<T>(max2DPositions, dim), "boxes");
        for (int i = 0; i < numEncoderLayers; i++)
        {
            _encoderSelfNorm.Add(new RMSNormalizationLayer<T>(dim, epsilon));
            _encoderSelfAttention.Add(new T5AttentionLayer<T>(dim, numHeads, keyValueDim));
            _encoderFeedForwardNorm.Add(new RMSNormalizationLayer<T>(dim, epsilon));
            _encoderFeedForward.Add(new T5FeedForwardLayer<T>(dim, ffDim));
        }
        _encoderFinalNorm = new RMSNormalizationLayer<T>(dim, epsilon);
        for (int i = 0; i < numDecoderLayers; i++)
        {
            _decoderSelfNorm.Add(new RMSNormalizationLayer<T>(dim, epsilon));
            _decoderSelfAttention.Add(new T5AttentionLayer<T>(dim, numHeads, keyValueDim));
            _decoderCrossNorm.Add(new RMSNormalizationLayer<T>(dim, epsilon));
            _decoderCrossAttention.Add(new T5AttentionLayer<T>(dim, numHeads, keyValueDim));
            _decoderFeedForwardNorm.Add(new RMSNormalizationLayer<T>(dim, epsilon));
            _decoderFeedForward.Add(new T5FeedForwardLayer<T>(dim, ffDim));
        }
        _decoderFinalNorm = new RMSNormalizationLayer<T>(dim, epsilon);
        foreach (var layer in SubLayers()) RegisterSubLayer(layer);
        // Registered apart from SubLayers(): the cell tables are parallel lookups on the boxes, not a stage of the chain.
        RegisterSubLayer(_cellX);
        RegisterSubLayer(_cellY);
    }

    private IEnumerable<LayerBase<T>> SubLayers()
    {
        yield return _patchEmbedding;
        for (int i = 0; i < _numEncoderLayers; i++)
        {
            yield return _encoderSelfNorm[i]; yield return _encoderSelfAttention[i];
            yield return _encoderFeedForwardNorm[i]; yield return _encoderFeedForward[i];
        }
        yield return _encoderFinalNorm;
        for (int i = 0; i < _numDecoderLayers; i++)
        {
            yield return _decoderSelfNorm[i]; yield return _decoderSelfAttention[i];
            yield return _decoderCrossNorm[i]; yield return _decoderCrossAttention[i];
            yield return _decoderFeedForwardNorm[i]; yield return _decoderFeedForward[i];
        }
        yield return _decoderFinalNorm;
    }

    public int VocabSize => _vocabSize;

    /// <summary>Side of the square page image the patch embedding expects.</summary>
    public int ImageSize => _imageSize;

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 2
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Fixed(1)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_vocabSize)),
        }
        : null;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 2 || input.Shape[1] != 5)
            throw new ArgumentException(
                $"UdopTransformerLayer expects packed rows of shape [S, 5] (token, x0, y0, x1, y1); got shape [{string.Join(", ", input.Shape.ToArray())}].",
                nameof(input));
        var (tokens, boxes) = Unpack(input);
        return Decode(new[] { 0 }, Encode(tokens, boxes, null));
    }

    /// <summary>Splits packed <c>[S, 5]</c> rows into clamped token ids and 0-1 boxes.</summary>
    internal (int[] Tokens, double[][] Boxes) Unpack(Tensor<T> packed)
    {
        int s = packed.Shape[0];
        var tokens = new int[s];
        var boxes = new double[s][];
        for (int i = 0; i < s; i++)
        {
            tokens[i] = ClampToken((int)Math.Round(NumOps.ToDouble(packed[i, 0])));
            boxes[i] = new double[4];
            for (int c = 0; c < 4; c++)
                boxes[i][c] = Math.Min(Math.Max(NumOps.ToDouble(packed[i, c + 1]), 0), CoordinateGrid) / CoordinateGrid;
        }
        return (tokens, boxes);
    }

    internal int ClampToken(int id) => Math.Min(Math.Max(id, 0), _vocabSize - 1);

    /// <summary>
    /// Runs the encoder. <paramref name="boxes"/> are 0-1 corners, one per token. <paramref name="image"/> is an
    /// optional normalised <c>[1, 3, size, size]</c> page.
    /// </summary>
    internal Tensor<T> Encode(int[] tokens, double[][] boxes, Tensor<T>? image)
    {
        if (tokens.Length != boxes.Length) throw new ArgumentException("Every token needs a box.", nameof(boxes));
        if (tokens.Length == 0 && image is null) throw new ArgumentException("UDOP needs at least one token or a page image.", nameof(tokens));
        var sequenceBoxes = new List<double[]>(boxes);
        Tensor<T>? x = tokens.Length > 0 ? CvTensorOps<T>.Select(_shared, tokens, 0) : null;             // [S, d]

        if (image is not null)
        {
            if (image.Rank != 4 || image.Shape[0] != 1 || image.Shape[1] != 3 || image.Shape[2] != _imageSize || image.Shape[3] != _imageSize)
                throw new ArgumentException(
                    $"UDOP expects one page of shape [1, 3, {_imageSize}, {_imageSize}]; got shape [{string.Join(", ", image.Shape.ToArray())}].", nameof(image));
            int grid = _imageSize / _patchSize, patches = grid * grid;
            var map = _patchEmbedding.Forward(image);                                                          // [1, d, g, g]
            var patchRows = Engine.TensorPermute(Engine.Reshape(map, new[] { _dim, patches }), new[] { 1, 0 });  // [P, d]

            var claimed = new bool[patches];
            if (x is not null)
            {
                var points = new int[tokens.Length];
                var keep = new Tensor<T>(new[] { tokens.Length, _dim });
                for (int i = 0; i < tokens.Length; i++)
                {
                    var b = boxes[i];
                    int px = Math.Min(Math.Max((int)Math.Floor((b[0] + b[2]) / 2.0 * grid), 0), grid - 1);
                    int py = Math.Min(Math.Max((int)Math.Floor((b[1] + b[3]) / 2.0 * grid), 0), grid - 1);
                    points[i] = px + (py * grid);
                    claimed[points[i]] = true;
                    double mean = (b[0] + b[1] + b[2] + b[3]) / 4.0;
                    if (mean != 0.0 && mean != 1.0)
                        for (int d = 0; d < _dim; d++) keep[i, d] = NumOps.One;
                }
                x = Engine.TensorAdd(x, Engine.TensorMultiply(CvTensorOps<T>.Select(patchRows, points, 0), keep));
            }

            var unclaimed = Enumerable.Range(0, patches).Where(p => !claimed[p]).ToArray();
            if (unclaimed.Length > 0)
            {
                var rest = CvTensorOps<T>.Select(patchRows, unclaimed, 0);
                x = x is null ? rest : Engine.TensorConcatenate(new[] { x, rest }, 0);
                foreach (int p in unclaimed)
                {
                    int column = p % grid, row = p / grid;
                    sequenceBoxes.Add(new[] { (double)column / grid, (double)row / grid, (double)(column + 1) / grid, (double)(row + 1) / grid });
                }
            }
        }
        if (x is null) throw new InvalidOperationException("Every image patch was claimed and there are no tokens.");

        x = Engine.TensorAdd(x, CellEmbedding(sequenceBoxes));
        var bias = EncoderBias(sequenceBoxes);
        for (int i = 0; i < _numEncoderLayers; i++)
        {
            var h = _encoderSelfNorm[i].Forward(x);
            x = Engine.TensorAdd(x, _encoderSelfAttention[i].Forward(h, h, bias, causal: false));
            x = Engine.TensorAdd(x, _encoderFeedForward[i].Forward(_encoderFeedForwardNorm[i].Forward(x)));
        }
        return _encoderFinalNorm.Forward(x);
    }

    private Tensor<T> CellEmbedding(IReadOnlyList<double[]> boxes)
    {
        int s = boxes.Count;
        Tensor<T> Ids(int corner)
        {
            var ids = new Tensor<T>(new[] { s });
            for (int i = 0; i < s; i++)
            {
                double value = Math.Min(Math.Max(boxes[i][corner], 0.0), 1.0);
                ids[i] = NumOps.FromDouble(Math.Min((long)(value * (_max2DPositions - 1)), _max2DPositions - 1));
            }
            return ids;
        }
        return Engine.TensorAdd(
            Engine.TensorAdd(_cellX.Forward(Ids(0)), _cellY.Forward(Ids(1))),
            Engine.TensorAdd(_cellX.Forward(Ids(2)), _cellY.Forward(Ids(3))));
    }

    private Tensor<T> EncoderBias(IReadOnlyList<double[]> boxes)
    {
        int s = boxes.Count;
        var order = new int[s * s];
        var horizontal = new int[s * s];
        var vertical = new int[s * s];
        for (int q = 0; q < s; q++)
        {
            double qx = (boxes[q][0] + boxes[q][2]) / 2.0, qy = (boxes[q][1] + boxes[q][3]) / 2.0;
            for (int k = 0; k < s; k++)
            {
                double kx = (boxes[k][0] + boxes[k][2]) / 2.0, ky = (boxes[k][1] + boxes[k][3]) / 2.0;
                int at = (q * s) + k;
                order[at] = T5RelativePositionBuckets.Bucket(k - q, true, _numBuckets, _maxDistance);
                horizontal[at] = T5RelativePositionBuckets.Bucket((long)((kx - qx) * LayoutScale), true, _numBuckets, _maxDistance2D);
                vertical[at] = T5RelativePositionBuckets.Bucket((long)((ky - qy) * LayoutScale), true, _numBuckets, _maxDistance2D);
            }
        }
        return Engine.TensorAdd(
            T5RelativePositionBuckets.Bias(_encoderBias1D, order, s, s),
            Engine.TensorAdd(
                T5RelativePositionBuckets.Bias(_encoderBiasHorizontal, horizontal, s, s),
                T5RelativePositionBuckets.Bias(_encoderBiasVertical, vertical, s, s)));
    }

    /// <summary>Runs the decoder over <paramref name="decoderIds"/> and returns next-token logits <c>[T, vocab]</c>.</summary>
    internal Tensor<T> Decode(int[] decoderIds, Tensor<T> memory)
    {
        int t = decoderIds.Length;
        if (t == 0) throw new ArgumentException("The decoder needs at least the start token.", nameof(decoderIds));
        var y = CvTensorOps<T>.Select(_shared, decoderIds.Select(ClampToken).ToArray(), 0);
        var buckets = new int[t * t];
        for (int q = 0; q < t; q++)
            for (int k = 0; k < t; k++)
                buckets[(q * t) + k] = T5RelativePositionBuckets.Bucket(k - q, false, _numBuckets, _maxDistance);
        var bias = T5RelativePositionBuckets.Bias(_decoderBias, buckets, t, t);
        for (int i = 0; i < _numDecoderLayers; i++)
        {
            var h = _decoderSelfNorm[i].Forward(y);
            y = Engine.TensorAdd(y, _decoderSelfAttention[i].Forward(h, h, bias, causal: true));
            y = Engine.TensorAdd(y, _decoderCrossAttention[i].Forward(_decoderCrossNorm[i].Forward(y), memory, null, causal: false));
            y = Engine.TensorAdd(y, _decoderFeedForward[i].Forward(_decoderFeedForwardNorm[i].Forward(y)));
        }
        var scaled = Engine.TensorMultiplyScalar(_decoderFinalNorm.Forward(y), NumOps.FromDouble(Math.Pow(_dim, -0.5)));
        return Engine.TensorMatMul(scaled, Engine.TensorTranspose(_shared));
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

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["VocabSize"] = _vocabSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Dim"] = _dim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["KeyValueDim"] = _keyValueDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["FfDim"] = _ffDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumEncoderLayers"] = _numEncoderLayers.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumDecoderLayers"] = _numDecoderLayers.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["ImageSize"] = _imageSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["PatchSize"] = _patchSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumBuckets"] = _numBuckets.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MaxDistance"] = _maxDistance.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MaxDistance2D"] = _maxDistance2D.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Max2DPositions"] = _max2DPositions.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Epsilon"] = _epsilon.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    public override void ResetState()
    {
        foreach (var layer in SubLayers()) layer.ResetState();
        _cellX.ResetState();
        _cellY.ResetState();
    }
}
