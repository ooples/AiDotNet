using AiDotNet.Attributes;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// The Point Transformer V3 backbone (Wu et al., CVPR 2024), shared by <see cref="PointTransformerV3{T}"/>
/// and <see cref="Sonata{T}"/>: serialization, a sparse-convolution or linear stem, staged blocks of
/// serialized patch attention with xCPE, grid pooling between stages and, unless it is encoder-only,
/// a U-Net decoder of unpooling plus blocks.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>Input</b>: <c>[points, width]</c> for one cloud or <c>[..., points, width]</c> for a batch of
/// equally sized clouds. The first three channels are the xyz coordinates in metres; the features
/// are the remaining channels, or every channel when coordinates are also features (Sonata feeds
/// coordinate, colour and normal). Grid coordinates are <c>floor((xyz - min) / gridSize)</c> per
/// cloud, as Pointcept's <c>GridSample</c> produces; the input is expected to be grid-sampled
/// already (one point per voxel), as in every PTv3 and Sonata configuration.
/// </para>
/// <para>
/// <b>Output</b>: one feature vector per input point, <c>[..., points, OutputChannels]</c>: the
/// decoder's finest stage, or, encoder-only, Sonata's feature up-casting of every stage concatenated
/// back to full resolution.
/// </para>
/// <para>
/// Only the features flow through <c>T</c>; grid coordinates, codes, orders, neighbour tables and
/// cluster maps are <see cref="int"/>/<see cref="long"/> and coordinates are <see cref="double"/>,
/// so a half-precision network never rounds an index.
/// </para>
/// <para>
/// <b>For Beginners:</b> This layer turns a point cloud into a feature vector for every point.
/// It sorts the points along space-filling curves so that nearby points sit next to each other,
/// lets each small group of neighbours attend to one another, and repeatedly merges neighbouring
/// points into coarser clouds to see larger regions.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Transformer)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, HasTrainingMode = true, Cost = ComputeCost.High,
    TestInputShape = "1, 24, 9",
    TestConstructorArgs = "6, false, 0.05, new[] { 2 }, new[] { 1, 1 }, new[] { 8, 16 }, new[] { 2, 2 }, new[] { 16, 16 }, "
        + "new[] { 1 }, new[] { 8 }, new[] { 2 }, new[] { 16 }")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Length, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Length, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public sealed partial class PointTransformerV3Encoder<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inputChannels;
    private readonly bool _coordinatesAsFeatures;
    private readonly double _gridSize;
    private readonly int[] _strides;
    private readonly int[] _encoderDepths;
    private readonly int[] _encoderChannels;
    private readonly int[] _encoderNumHeads;
    private readonly int[] _encoderPatchSizes;
    private readonly int[] _decoderDepths;
    private readonly int[] _decoderChannels;
    private readonly int[] _decoderNumHeads;
    private readonly int[] _decoderPatchSizes;
    private readonly double _mlpRatio;
    private readonly double _dropPathRate;
    private readonly bool _enableConditionalPositionalEncoding;
    private readonly bool _preNorm;
    private readonly bool _shuffleOrders;
    private readonly string[] _serializationOrders;
    private readonly PointTransformerV3Revision _revision;
    private readonly bool _encoderOnly;
    private readonly bool _maskToken;
    private readonly int? _seed;

    private readonly PtContext<T> _ctx;
    private readonly ISpaceFillingCurve[] _curves;
    private readonly PtSubmanifoldConv<T>? _stemConv;
    private readonly PtLinear<T>? _stemLinear;
    private readonly PtNorm<T> _stemNorm;
    private readonly int _maskTokenSlot = -1;
    private readonly PointTransformerV3Pooling<T>[] _pooling;
    private readonly PointTransformerV3Block<T>[][] _encoderBlocks;
    private readonly PointTransformerV3Unpooling<T>[] _unpooling;
    private readonly PointTransformerV3Block<T>[][] _decoderBlocks;

    /// <summary>Creates a PTv3 backbone. Every argument left null takes the PTv3 m1 base value.</summary>
    /// <param name="inputChannels">Feature channels per point (6 in PTv3: colour and normal).</param>
    /// <param name="coordinatesAsFeatures">Whether the xyz channels are features too (Sonata: true).</param>
    /// <param name="gridSize">Voxel edge in metres (0.02).</param>
    /// <param name="strides">Pooling stride between stages (2, 2, 2, 2).</param>
    /// <param name="encoderDepths">Blocks per encoder stage (2, 2, 2, 6, 2).</param>
    /// <param name="encoderChannels">Channels per encoder stage (32, 64, 128, 256, 512).</param>
    /// <param name="encoderNumHeads">Heads per encoder stage (2, 4, 8, 16, 32).</param>
    /// <param name="encoderPatchSizes">Patch size per encoder stage (1024 each).</param>
    /// <param name="decoderDepths">Blocks per decoder stage, finest first (2, 2, 2, 2).</param>
    /// <param name="decoderChannels">Channels per decoder stage, finest first (64, 64, 128, 256).</param>
    /// <param name="decoderNumHeads">Heads per decoder stage (4, 4, 8, 16).</param>
    /// <param name="decoderPatchSizes">Patch size per decoder stage (1024 each).</param>
    /// <param name="mlpRatio">MLP hidden width over channels (4).</param>
    /// <param name="dropPathRate">Largest stochastic-depth rate, reached linearly by the last block (0.3).</param>
    /// <param name="enableConditionalPositionalEncoding">Whether each block starts with xCPE (true).</param>
    /// <param name="preNorm">Pre-norm blocks (true).</param>
    /// <param name="shuffleOrders">Whether training permutes the serialization orders after every pooling (true).</param>
    /// <param name="serializationOrders">Registered <see cref="ISpaceFillingCurve"/> names (z, z-trans, hilbert, hilbert-trans).</param>
    /// <param name="revision">Which Pointcept backbone to build.</param>
    /// <param name="encoderOnly">Drop the decoder and up-cast the encoder features instead (Sonata).</param>
    /// <param name="maskToken">Learn a mask token for Sonata's masked views (m2 only).</param>
    /// <param name="seed">Initialisation seed, or null for a secure random one.</param>
    public PointTransformerV3Encoder(
        [LayerState] int inputChannels = 6,
        [LayerState] bool coordinatesAsFeatures = false,
        [LayerState] double gridSize = 0.02,
        [LayerState] int[]? strides = null,
        [LayerState] int[]? encoderDepths = null,
        [LayerState] int[]? encoderChannels = null,
        [LayerState] int[]? encoderNumHeads = null,
        [LayerState] int[]? encoderPatchSizes = null,
        [LayerState] int[]? decoderDepths = null,
        [LayerState] int[]? decoderChannels = null,
        [LayerState] int[]? decoderNumHeads = null,
        [LayerState] int[]? decoderPatchSizes = null,
        [LayerState] double mlpRatio = 4.0,
        [LayerState] double dropPathRate = 0.3,
        [LayerState] bool enableConditionalPositionalEncoding = true,
        [LayerState] bool preNorm = true,
        [LayerState] bool shuffleOrders = true,
        [LayerState] string[]? serializationOrders = null,
        [LayerState] PointTransformerV3Revision revision = PointTransformerV3Revision.M1Base,
        [LayerState] bool encoderOnly = false,
        [LayerState] bool maskToken = false,
        [LayerState] int? seed = null)
        : base(new[] { InputWidth(inputChannels, coordinatesAsFeatures) },
            new[] { OutputWidth(encoderOnly, encoderChannels ?? DefaultEncoderChannels, decoderChannels ?? DefaultDecoderChannels) })
    {
        _inputChannels = inputChannels;
        _coordinatesAsFeatures = coordinatesAsFeatures;
        _gridSize = gridSize;
        _strides = strides ?? new[] { 2, 2, 2, 2 };
        _encoderDepths = encoderDepths ?? new[] { 2, 2, 2, 6, 2 };
        _encoderChannels = encoderChannels ?? DefaultEncoderChannels;
        _encoderNumHeads = encoderNumHeads ?? new[] { 2, 4, 8, 16, 32 };
        _encoderPatchSizes = encoderPatchSizes ?? Repeat(1024, _encoderDepths.Length);
        _decoderDepths = decoderDepths ?? new[] { 2, 2, 2, 2 };
        _decoderChannels = decoderChannels ?? DefaultDecoderChannels;
        _decoderNumHeads = decoderNumHeads ?? new[] { 4, 4, 8, 16 };
        _decoderPatchSizes = decoderPatchSizes ?? Repeat(1024, _decoderDepths.Length);
        _mlpRatio = mlpRatio;
        _dropPathRate = dropPathRate;
        _enableConditionalPositionalEncoding = enableConditionalPositionalEncoding;
        _preNorm = preNorm;
        _shuffleOrders = shuffleOrders;
        _serializationOrders = serializationOrders ?? new[] { "z", "z-trans", "hilbert", "hilbert-trans" };
        _revision = revision;
        _encoderOnly = encoderOnly;
        _maskToken = maskToken;
        _seed = seed;
        Validate();

        _curves = new ISpaceFillingCurve[_serializationOrders.Length];
        for (int i = 0; i < _curves.Length; i++) _curves[i] = SpaceFillingCurves.Resolve(_serializationOrders[i]);

        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        bool m1 = revision == PointTransformerV3Revision.M1Base;
        _ctx = new PtContext<T>(random, truncatedNormalInit: !m1);

        int stages = _encoderDepths.Length;
        int c0 = _encoderChannels[0];
        if (m1)
        {
            // Embedding: SubMConv3d(in, C0, kernel_size=5, bias=False) -> BatchNorm1d -> GELU.
            _stemConv = new PtSubmanifoldConv<T>(_ctx, inputChannels, c0, 5, bias: false);
            _stemNorm = new PtBatchNorm<T>(_ctx, c0, "embedding.norm");
        }
        else
        {
            // Embedding: Linear(in, C0) -> LayerNorm -> GELU, then an optional mask token.
            _stemLinear = new PtLinear<T>(_ctx, inputChannels, c0);
            _stemNorm = new PtLayerNorm<T>(_ctx, c0);
            if (maskToken) _maskTokenSlot = _ctx.Add(new Tensor<T>(new[] { 1, c0 }), PersistentTensorRole.Embeddings);
        }

        var encoderDrop = Linspace(0.0, dropPathRate, Sum(_encoderDepths));
        _pooling = new PointTransformerV3Pooling<T>[stages - 1];
        _encoderBlocks = new PointTransformerV3Block<T>[stages][];
        int blockIndex = 0;
        for (int s = 0; s < stages; s++)
        {
            if (s > 0)
                _pooling[s - 1] = new PointTransformerV3Pooling<T>(_ctx, _encoderChannels[s - 1], _encoderChannels[s],
                    _strides[s - 1], revision, $"enc{s}.down");
            _encoderBlocks[s] = new PointTransformerV3Block<T>[_encoderDepths[s]];
            for (int i = 0; i < _encoderDepths[s]; i++)
            {
                _encoderBlocks[s][i] = new PointTransformerV3Block<T>(_ctx, _encoderChannels[s], _encoderNumHeads[s],
                    _encoderPatchSizes[s], mlpRatio, encoderDrop[blockIndex++], preNorm, i % _curves.Length,
                    enableConditionalPositionalEncoding);
            }
        }

        _unpooling = new PointTransformerV3Unpooling<T>[encoderOnly ? 0 : stages - 1];
        _decoderBlocks = new PointTransformerV3Block<T>[encoderOnly ? 0 : stages - 1][];
        if (!encoderOnly)
        {
            var decoderDrop = Linspace(0.0, dropPathRate, Sum(_decoderDepths));
            var widths = new int[stages];
            Array.Copy(_decoderChannels, widths, stages - 1);
            widths[stages - 1] = _encoderChannels[stages - 1];
            for (int s = stages - 2; s >= 0; s--)
            {
                int start = Sum(_decoderDepths, s);
                _unpooling[s] = new PointTransformerV3Unpooling<T>(_ctx, widths[s + 1], _encoderChannels[s], widths[s],
                    revision, $"dec{s}.up");
                _decoderBlocks[s] = new PointTransformerV3Block<T>[_decoderDepths[s]];
                for (int i = 0; i < _decoderDepths[s]; i++)
                {
                    // Pointcept reverses each decoder stage's slice of the drop-path schedule.
                    double rate = decoderDrop[start + _decoderDepths[s] - 1 - i];
                    _decoderBlocks[s][i] = new PointTransformerV3Block<T>(_ctx, widths[s], _decoderNumHeads[s],
                        _decoderPatchSizes[s], mlpRatio, rate, preNorm, i % _curves.Length,
                        enableConditionalPositionalEncoding);
                }
            }
        }

        for (int i = 0; i < _ctx.Created.Count; i++) RegisterTrainableParameter(_ctx.Created[i], _ctx.Roles[i]);
        foreach (var (tensor, name) in _ctx.Buffers) RegisterBuffer(tensor, name);
        _ctx.Live = GetTrainableParameters;
    }

    private static readonly int[] DefaultEncoderChannels = { 32, 64, 128, 256, 512 };
    private static readonly int[] DefaultDecoderChannels = { 64, 64, 128, 256 };

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Feature channels per point the layer expects after the coordinates.</summary>
    public int InputChannels => _inputChannels;

    /// <summary>Width of the input's last axis: three coordinates plus the features, or just the features when they include the coordinates.</summary>
    public int InputWidthChannels => InputWidth(_inputChannels, _coordinatesAsFeatures);

    /// <summary>Feature channels per output point.</summary>
    public int OutputChannels => OutputWidth(_encoderOnly, _encoderChannels, _decoderChannels);

    /// <summary>Whether this backbone has no decoder (Sonata).</summary>
    public bool IsEncoderOnly => _encoderOnly;

    internal IReadOnlyList<int> EncoderChannels => _encoderChannels;

    internal int StageCount => _encoderDepths.Length;

    internal PointTransformerV3Revision Revision => _revision;

    internal PtContext<T> Context => _ctx;

    internal IReadOnlyList<ISpaceFillingCurve> Curves => _curves;

    internal PointTransformerV3Block<T> EncoderBlock(int stage, int index) => _encoderBlocks[stage][index];

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(OutputChannels));
        var points = new OutputAxisContract(TensorAxis.Length, AxisRelation.Same(TensorAxis.Length));
        return inputRank switch
        {
            2 => new[] { points, features },
            3 => new[] { new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)), points, features },
            _ => null
        };
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var (level, leading, points) = PrepareInput(input, IsTrainingMode);
        var output = ForwardLevel(level, IsTrainingMode);
        var shape = new int[leading.Length + 2];
        Array.Copy(leading, shape, leading.Length);
        shape[leading.Length] = points;
        shape[leading.Length + 1] = OutputChannels;
        return Engine.Reshape(output, shape);
    }

    /// <summary>Runs the whole backbone on a prepared level and returns full-resolution features.</summary>
    internal Tensor<T> ForwardLevel(PointCloudLevel<T> level, bool training)
    {
        var levels = Encode(level, training);
        if (_encoderOnly) return UpCast(levels, levels.Count - 1).Features;
        return Decode(levels, training);
    }

    /// <summary>
    /// Splits <c>[..., points, width]</c> into the flattened features and the geometry of every cloud,
    /// grid-samples coordinates and serializes them.
    /// </summary>
    internal (PointCloudLevel<T> Level, int[] Leading, int Points) PrepareInput(Tensor<T> input, bool training)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        int width = InputWidthChannels;
        if (input.Rank < 2 || input.Shape[input.Rank - 1] != width)
            throw new ArgumentException(
                $"Expected [..., points, {width}] (xyz first), got [{string.Join(", ", input.Shape.ToArray())}].", nameof(input));
        int points = input.Shape[input.Rank - 2];
        var leading = new int[input.Rank - 2];
        int clouds = 1;
        for (int i = 0; i < leading.Length; i++)
        {
            leading[i] = input.Shape[i];
            clouds *= leading[i];
        }

        int total = clouds * points;
        var flat = Engine.Reshape(input, new[] { total, width });
        var values = new float[total * width];
        NumOps.ToFloatSpan(flat.ToArray(), values);
        var coordinates = new double[total * 3];
        for (int i = 0; i < total; i++)
        {
            coordinates[i * 3] = values[i * width];
            coordinates[i * 3 + 1] = values[i * width + 1];
            coordinates[i * 3 + 2] = values[i * width + 2];
        }

        var features = _coordinatesAsFeatures
            ? flat
            : Engine.TensorSlice(flat, new[] { 0, 3 }, new[] { total, _inputChannels });
        var batch = new int[total];
        for (int i = 0; i < total; i++) batch[i] = i / Math.Max(1, points);
        return (BuildLevel(features, coordinates, coordinates, batch, clouds, null, training), leading, points);
    }

    /// <summary>Builds the full-resolution level: grid sampling coordinates and serializing them.</summary>
    internal PointCloudLevel<T> BuildLevel(Tensor<T> features, double[] coordinates, double[] originCoordinates,
        int[] batch, int batchCount, Tensor<T>? mask, bool training)
    {
        int count = batch.Length;
        var min = new double[batchCount * 3];
        for (int i = 0; i < min.Length; i++) min[i] = double.PositiveInfinity;
        for (int i = 0; i < count; i++)
        {
            for (int a = 0; a < 3; a++)
                min[batch[i] * 3 + a] = Math.Min(min[batch[i] * 3 + a], coordinates[i * 3 + a]);
        }

        var grid = new int[count * 3];
        for (int i = 0; i < count; i++)
        {
            for (int a = 0; a < 3; a++)
            {
                double cell = Math.Floor((coordinates[i * 3 + a] - min[batch[i] * 3 + a]) / _gridSize);
                if (cell > 65535)
                    throw new ArgumentException(
                        $"A cloud spans more than 65536 voxels of {_gridSize} m along one axis; use a larger grid size.");
                grid[i * 3 + a] = (int)cell;
            }
        }

        var serialization = PointSerialization.Serialize(grid, batch, PointSerialization.DepthFor(grid), _curves);
        if (_shuffleOrders && training) serialization = serialization.Permute(RandomPermutation(_ctx.Random, _curves.Length));
        return new PointCloudLevel<T>(features, coordinates, originCoordinates, grid, batch, batchCount, serialization)
        {
            Mask = mask
        };
    }

    /// <summary>The embedding and every encoder stage; returns the level of each stage, finest first.</summary>
    internal List<PointCloudLevel<T>> Encode(PointCloudLevel<T> level, bool training)
    {
        var engine = Engine;
        Tensor<T> x = _stemConv is not null
            ? _stemConv.Forward(level, level.Features)
            : (_stemLinear ?? throw new InvalidOperationException("The PTv3 stem was not built.")).Forward(level.Features);
        x = engine.TensorGELU(_stemNorm.Forward(x, training));
        if (_maskTokenSlot >= 0 && level.Mask is not null)
        {
            // torch.where(mask, mask_token, feat), written as a blend so the token receives its gradient.
            var shape = x.Shape.ToArray();
            var mask = engine.TensorBroadcastTo(level.Mask, shape);
            var ones = new Tensor<T>(level.Mask.Shape.ToArray());
            ones.Fill(NumOps.One);
            var keep = engine.TensorBroadcastTo(engine.TensorSubtract(ones, level.Mask), shape);
            var token = engine.TensorBroadcastTo(_ctx[_maskTokenSlot], shape);
            x = engine.TensorAdd(engine.TensorMultiply(x, keep), engine.TensorMultiply(token, mask));
        }

        level.Features = x;
        var levels = new List<PointCloudLevel<T>> { level };
        var current = level;
        for (int s = 0; s < _encoderBlocks.Length; s++)
        {
            if (s > 0)
            {
                current = _pooling[s - 1].Forward(current, _curves, _shuffleOrders, training);
                levels.Add(current);
            }

            foreach (var block in _encoderBlocks[s])
                current.Features = block.Forward(current, current.Features, training);
        }

        return levels;
    }

    /// <summary>The decoder: unpool into each finer stage and run its blocks.</summary>
    internal Tensor<T> Decode(List<PointCloudLevel<T>> levels, bool training)
    {
        for (int s = levels.Count - 2; s >= 0; s--)
        {
            var target = levels[s];
            target.Features = _unpooling[s].Forward(levels[s + 1], target, training);
            foreach (var block in _decoderBlocks[s])
                target.Features = block.Forward(target, target.Features, training);
        }

        return levels[0].Features;
    }

    /// <summary>
    /// Sonata's feature up-casting: walks <paramref name="steps"/> poolings back up, concatenating
    /// each parent's features with its cell's coarse features (<c>cat([parent.feat, point.feat[inverse]])</c>).
    /// </summary>
    internal PointCloudLevel<T> UpCast(List<PointCloudLevel<T>> levels, int steps)
    {
        var current = levels[levels.Count - 1];
        for (int i = 0; i < steps; i++)
        {
            var parent = current.Parent ?? throw new InvalidOperationException("Cannot up-cast past full resolution.");
            var inverse = current.PoolingInverse ?? throw new InvalidOperationException("Missing pooling cluster map.");
            parent.Features = Engine.TensorConcatenate(
                new[] { parent.Features, Engine.TensorIndexSelect(current.Features, PtOps.Indices(inverse), 0) }, 1);
            current = parent;
        }

        return current;
    }

    internal static int[] RandomPermutation(Random random, int count)
    {
        var permutation = new int[count];
        for (int i = 0; i < count; i++) permutation[i] = i;
        for (int i = count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (permutation[i], permutation[j]) = (permutation[j], permutation[i]);
        }

        return permutation;
    }

    private void Validate()
    {
        int stages = _encoderDepths.Length;
        if (stages < 1) throw new ArgumentException("At least one encoder stage is required.");
        if (_inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(InputChannels), "Input channels must be positive.");
        if (_coordinatesAsFeatures && _inputChannels < 3)
            throw new ArgumentException("When coordinates are features the feature width must be at least 3.");
        if (!(_gridSize > 0) || double.IsInfinity(_gridSize)) throw new ArgumentException("Grid size must be positive and finite.");
        if (_encoderChannels.Length != stages || _encoderNumHeads.Length != stages || _encoderPatchSizes.Length != stages)
            throw new ArgumentException("Encoder depths, channels, heads and patch sizes must all have one entry per stage.");
        if (_strides.Length != stages - 1) throw new ArgumentException("There must be one pooling stride between each pair of stages.");
        foreach (int stride in _strides)
        {
            if (stride < 1) throw new ArgumentException("Pooling strides must be positive.");
            if (_revision == PointTransformerV3Revision.M1Base && (stride & (stride - 1)) != 0)
                throw new ArgumentException("PTv3 m1 pools by serialized code, so its strides must be powers of two.");
        }

        if (!_encoderOnly
            && (_decoderDepths.Length != stages - 1 || _decoderChannels.Length != stages - 1
                || _decoderNumHeads.Length != stages - 1 || _decoderPatchSizes.Length != stages - 1))
            throw new ArgumentException("Decoder depths, channels, heads and patch sizes need one entry per stage but the last.");
        if (_serializationOrders.Length == 0) throw new ArgumentException("At least one serialization order is required.");
        if (!(_mlpRatio > 0)) throw new ArgumentException("The MLP ratio must be positive.");
        if (_dropPathRate < 0 || _dropPathRate >= 1) throw new ArgumentException("The drop-path rate must be in [0, 1).");
        if (_maskToken && _revision != PointTransformerV3Revision.M2Sonata)
            throw new ArgumentException("The mask token belongs to the Sonata (m2) embedding.");
    }

    private static int InputWidth(int inputChannels, bool coordinatesAsFeatures)
        => coordinatesAsFeatures ? inputChannels : inputChannels + 3;

    private static int OutputWidth(bool encoderOnly, int[] encoderChannels, int[] decoderChannels)
        => encoderOnly ? Sum(encoderChannels) : (decoderChannels.Length > 0 ? decoderChannels[0] : encoderChannels[0]);

    private static int Sum(int[] values, int count = -1)
    {
        int n = count < 0 ? values.Length : count, total = 0;
        for (int i = 0; i < n; i++) total += values[i];
        return total;
    }

    private static int[] Repeat(int value, int count)
    {
        var result = new int[count];
        for (int i = 0; i < count; i++) result[i] = value;
        return result;
    }

    /// <summary><c>torch.linspace(start, end, count)</c>, computed in double.</summary>
    internal static double[] Linspace(double start, double end, int count)
    {
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = count == 1 ? start : start + (end - start) * i / (count - 1);
        return values;
    }
}
