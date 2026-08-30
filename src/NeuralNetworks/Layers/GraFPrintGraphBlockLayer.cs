using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements one GraFPrint Grapher + residual feed-forward block.
/// </summary>
/// <remarks>
/// This follows the official encoder: 1x1 projection and batch normalization, a dynamic
/// dilated k-NN max-relative graph convolution, a 1x1 projection back to the residual width,
/// then a 4x expansion feed-forward network. Both the graph branch and FFN are residual.
/// </remarks>
[LayerCategory(LayerCategory.Graph)]
[LayerTask(LayerTask.GraphProcessing)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = false, ExpectedInputRank = 3,
    Cost = ComputeCost.High, TestInputShape = "8, 4, 4",
    TestConstructorArgs = "8, 2, 1, 0.0")]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class GraFPrintGraphBlockLayer<T> : LayerBase<T>, ILayerSerializationExtras<T>, IShapeContract
{
    private readonly int _channels;
    private readonly int _k;
    private readonly int _dilation;
    private readonly double _dropPathRate;

    private readonly ConvolutionalLayer<T> _fc1;
    private readonly BatchNormalizationLayer<T> _fc1Bn;
    private readonly ConvolutionalLayer<T> _maxRelativeProjection;
    private readonly BatchNormalizationLayer<T> _maxRelativeBn;
    private readonly ConvolutionalLayer<T> _fc2;
    private readonly BatchNormalizationLayer<T> _fc2Bn;
    private readonly ConvolutionalLayer<T> _ffn1;
    private readonly BatchNormalizationLayer<T> _ffn1Bn;
    private readonly ConvolutionalLayer<T> _ffn2;
    private readonly BatchNormalizationLayer<T> _ffn2Bn;

    [Scratch]
    private int[]? _toNodeIndices;

    [Scratch]
    private int[]? _toSpatialIndices;

    [Scratch]
    private int[]? _cachedPermutationShape;

    [Scratch]
    private Vector<T>? _pendingExtraParameters;

    private long _dropPathCounter;

    /// <summary>
    /// Gets the most recently constructed dynamic neighbor graph as [batch,node,neighbor].
    /// This is diagnostic state and is intentionally excluded from serialization.
    /// </summary>
    internal int[,,]? LastNeighborIndices { get; private set; }

    /// <summary>Gets the graph neighborhood size.</summary>
    public int K => _k;

    /// <summary>Gets the residual feature width.</summary>
    public int Channels => _channels;

    /// <summary>Gets the dilated-neighborhood factor.</summary>
    public int Dilation => _dilation;

    /// <summary>Gets the residual stochastic-depth probability.</summary>
    public double DropPathRate => _dropPathRate;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates a paper-faithful GraFPrint graph block.</summary>
    public GraFPrintGraphBlockLayer(
        [LayerState] int channels,
        [LayerState] int k = 3,
        [LayerState] int dilation = 1,
        [LayerState] double dropPathRate = 0.0)
        : base([channels, -1, -1], [channels, -1, -1])
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (dilation <= 0) throw new ArgumentOutOfRangeException(nameof(dilation));
        if (dropPathRate < 0.0 || dropPathRate >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(dropPathRate));

        _channels = channels;
        _k = k;
        _dilation = dilation;
        _dropPathRate = dropPathRate;

        var identity = new IdentityActivation<T>();
        var relu = new ReLUActivation<T>();

        // Grapher fc1/fc2 and MRConv use bias=True in the official implementation,
        // even though each projection is followed by BatchNorm.
        _fc1 = new ConvolutionalLayer<T>(channels, 1, 1, 0, identity,
            nonlinearityForInit: relu, biasMode: BiasMode.Always);
        _fc1Bn = new BatchNormalizationLayer<T>();
        _maxRelativeProjection = new ConvolutionalLayer<T>(channels * 2, 1, 1, 0, identity,
            nonlinearityForInit: relu, biasMode: BiasMode.Always);
        _maxRelativeBn = new BatchNormalizationLayer<T>();
        _fc2 = new ConvolutionalLayer<T>(channels, 1, 1, 0, identity,
            nonlinearityForInit: relu, biasMode: BiasMode.Always);
        _fc2Bn = new BatchNormalizationLayer<T>();

        // The reference FFN explicitly disables convolution biases because BN supplies beta.
        _ffn1 = new ConvolutionalLayer<T>(channels * 4, 1, 1, 0, identity,
            nonlinearityForInit: relu, biasMode: BiasMode.Never);
        _ffn1Bn = new BatchNormalizationLayer<T>();
        _ffn2 = new ConvolutionalLayer<T>(channels, 1, 1, 0, identity,
            nonlinearityForInit: relu, biasMode: BiasMode.Never);
        _ffn2Bn = new BatchNormalizationLayer<T>();

        RegisterSubLayer(_fc1);
        RegisterSubLayer(_fc1Bn);
        RegisterSubLayer(_maxRelativeProjection);
        RegisterSubLayer(_maxRelativeBn);
        RegisterSubLayer(_fc2);
        RegisterSubLayer(_fc2Bn);
        RegisterSubLayer(_ffn1);
        RegisterSubLayer(_ffn1Bn);
        RegisterSubLayer(_ffn2);
        RegisterSubLayer(_ffn2Bn);

        // All trainable dimensions depend only on channels. Resolve at 1x1 now so clone and
        // deserialization know the exact parameter count before the first real spatial forward.
        ResolveSubLayers(height: 1, width: 1);
    }

    /// <inheritdoc />
    protected override void OnFirstForward(Tensor<T> input)
    {
        GetDimensions(input, out _, out int channels, out int height, out int width);
        if (channels != _channels)
            throw new ArgumentException(
                $"GraFPrint graph block was configured for {_channels} channels but received {channels}.",
                nameof(input));

        ResolveSubLayers(height, width);
        ResolveShapes([channels, height, width], [channels, height, width]);

        if (_pendingExtraParameters is not null)
        {
            var pending = _pendingExtraParameters;
            _pendingExtraParameters = null;
            ApplyExtraParameters(pending);
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Rank == 3;
        Tensor<T> x = unbatched
            ? Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2]])
            : input;

        GetDimensions(x, out int batch, out int channels, out int height, out int width);
        if (channels != _channels)
            throw new ArgumentException(
                $"GraFPrint graph block was configured for {_channels} channels but received {channels}.",
                nameof(input));

        // Grapher: fc1 -> dynamic max-relative graph convolution -> fc2 -> DropPath + skip.
        var projected = _fc1Bn.Forward(_fc1.Forward(x));
        var nodes = SpatialToNodes(projected, batch, channels, height, width);
        var maxRelative = ComputeMaxRelative(nodes, batch, height * width, channels);
        var maxRelativeSpatial = NodesToSpatial(maxRelative, batch, channels, height, width);
        var graphFeatures = Engine.Concat(new[] { projected, maxRelativeSpatial }, axis: 1);
        var graphUpdated = _maxRelativeProjection.Forward(graphFeatures);
        graphUpdated = _maxRelativeBn.Forward(graphUpdated);
        graphUpdated = Engine.ReLU(graphUpdated);
        var graphBranch = _fc2Bn.Forward(_fc2.Forward(graphUpdated));
        var graphResidual = Engine.TensorAdd(x, ApplyDropPath(graphBranch));

        // FFN: 1x1 expand 4x -> BN -> activation -> 1x1 contract -> BN -> DropPath + skip.
        var ffn = _ffn1Bn.Forward(_ffn1.Forward(graphResidual));
        ffn = Engine.ReLU(ffn);
        ffn = _ffn2Bn.Forward(_ffn2.Forward(ffn));
        var output = Engine.TensorAdd(graphResidual, ApplyDropPath(ffn));

        return unbatched
            ? Engine.Reshape(output, [channels, height, width])
            : output;
    }

    private Tensor<T> ComputeMaxRelative(
        Tensor<T> nodes, int batch, int nodeCount, int channels)
    {
        if (nodeCount == 1)
        {
            LastNeighborIndices = new int[batch, 1, 1];
            return Engine.TensorMultiplyScalar(nodes, NumOps.Zero);
        }

        // The official DenseDilatedKnnGraph L2-normalizes every node before measuring
        // Euclidean distance and leaves the diagonal in the candidate set. Consequently,
        // the node itself is normally the first of k neighbors; excluding it changes both
        // the max-relative aggregation and the meaning of k.
        int effectiveK = Math.Min(_k, nodeCount);
        int candidateCount = Math.Min(nodeCount, checked(effectiveK * _dilation));
        LastNeighborIndices = new int[batch, nodeCount, effectiveK];
        var batchResults = new List<Tensor<T>>(batch);

        for (int b = 0; b < batch; b++)
        {
            var sample = nodes.Slice(b); // [N,C]
            // Neighbor selection is discrete and detached in the reference implementation.
            // Build a value-only normalized copy so graph selection cannot accidentally add
            // a gradient path while the gathered node values below remain tape-connected.
            var normalizedSample = NormalizeNodesForKnn(sample, nodeCount, channels);
            var distances = Engine.PairwiseDistanceSquared(normalizedSample, normalizedSample);

            var (_, nearest) = Engine.TopK(
                distances, candidateCount, axis: 1, largest: false);
            var selfIndices = new int[nodeCount * effectiveK];
            var neighborIndices = new int[nodeCount * effectiveK];
            for (int node = 0; node < nodeCount; node++)
            {
                for (int neighbor = 0; neighbor < effectiveK; neighbor++)
                {
                    int candidate = Math.Min(neighbor * _dilation, candidateCount - 1);
                    int selected = nearest[node, candidate];
                    int flat = node * effectiveK + neighbor;
                    selfIndices[flat] = node;
                    neighborIndices[flat] = selected;
                    LastNeighborIndices[b, node, neighbor] = selected;
                }
            }

            var xi = Engine.TensorGather(
                sample, new Tensor<int>(selfIndices, [selfIndices.Length]), axis: 0);
            var xj = Engine.TensorGather(
                sample, new Tensor<int>(neighborIndices, [neighborIndices.Length]), axis: 0);
            var differences = Engine.TensorSubtract(xj, xi);
            var grouped = Engine.Reshape(
                differences, [nodeCount, effectiveK, channels]);

            // ReduceMax currently reports the correct winner but its rank-3/axis-1 backward
            // routing is not reliable. Use its discrete argmax only to build a constant mask,
            // then express the selected maximum as multiply + sum on the tape.
            _ = Engine.ReduceMax(grouped, new[] { 1 }, keepDims: false, out var argMax);
            var mask = new Tensor<T>([nodeCount, effectiveK, channels]);
            for (int node = 0; node < nodeCount; node++)
            {
                for (int channel = 0; channel < channels; channel++)
                {
                    int sourceFlatIndex = argMax[node * channels + channel];
                    int selectedNeighbor = (sourceFlatIndex / channels) % effectiveK;
                    if ((uint)selectedNeighbor < (uint)effectiveK)
                        mask[node, selectedNeighbor, channel] = NumOps.One;
                }
            }

            batchResults.Add(Engine.ReduceSum(
                Engine.TensorMultiply(grouped, mask), new[] { 1 }, keepDims: false));
        }

        var combined = batchResults.Count == 1
            ? batchResults[0]
            : Engine.TensorConcatenate(batchResults.ToArray(), axis: 0);
        return Engine.Reshape(combined, [batch, nodeCount, channels]);
    }

    private Tensor<T> NormalizeNodesForKnn(Tensor<T> sample, int nodeCount, int channels)
    {
        var normalized = new Tensor<T>([nodeCount, channels]);
        T epsilon = NumOps.FromDouble(1e-12);
        for (int node = 0; node < nodeCount; node++)
        {
            T sumSquares = NumOps.Zero;
            for (int channel = 0; channel < channels; channel++)
            {
                T value = sample[node, channel];
                sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(value, value));
            }

            T norm = NumOps.Sqrt(sumSquares);
            T denominator = NumOps.GreaterThan(norm, epsilon) ? norm : epsilon;
            for (int channel = 0; channel < channels; channel++)
                normalized[node, channel] = NumOps.Divide(sample[node, channel], denominator);
        }

        return normalized;
    }

    private Tensor<T> SpatialToNodes(
        Tensor<T> input, int batch, int channels, int height, int width)
    {
        EnsurePermutationIndices(batch, channels, height, width);
        var flat = Engine.Reshape(input, [input.Length]);
        var reordered = Engine.TensorGather(
            flat, new Tensor<int>(_toNodeIndices!, [_toNodeIndices!.Length]), axis: 0);
        return Engine.Reshape(reordered, [batch, height * width, channels]);
    }

    private Tensor<T> NodesToSpatial(
        Tensor<T> nodes, int batch, int channels, int height, int width)
    {
        EnsurePermutationIndices(batch, channels, height, width);
        var flat = Engine.Reshape(nodes, [nodes.Length]);
        var reordered = Engine.TensorGather(
            flat, new Tensor<int>(_toSpatialIndices!, [_toSpatialIndices!.Length]), axis: 0);
        return Engine.Reshape(reordered, [batch, channels, height, width]);
    }

    private void EnsurePermutationIndices(int batch, int channels, int height, int width)
    {
        int[] shape = [batch, channels, height, width];
        if (_cachedPermutationShape is not null &&
            _cachedPermutationShape.AsSpan().SequenceEqual(shape)) return;

        int length = checked(batch * channels * height * width);
        _toNodeIndices = new int[length];
        _toSpatialIndices = new int[length];
        int nodeOrder = 0;
        int spatialOrder = 0;
        int nodeCount = height * width;

        // Gather NCHW storage into [B,N,C].
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                    for (int c = 0; c < channels; c++)
                        _toNodeIndices[nodeOrder++] = ((b * channels + c) * height + h) * width + w;

        // Gather [B,N,C] storage back into NCHW.
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        _toSpatialIndices[spatialOrder++] = (b * nodeCount + h * width + w) * channels + c;

        _cachedPermutationShape = shape;
    }

    private Tensor<T> ApplyDropPath(Tensor<T> branch)
    {
        if (!IsTrainingMode || _dropPathRate <= 0.0) return branch;

        int batch = branch.Rank == 4 ? branch.Shape[0] : 1;
        T rate = NumOps.FromDouble(_dropPathRate);
        T scale = NumOps.FromDouble(1.0 / (1.0 - _dropPathRate));
        long counter = System.Threading.Interlocked.Increment(ref _dropPathCounter);
        int? seed = RandomSeed.HasValue
            ? unchecked((int)(((uint)RandomSeed.Value * 2654435761u) ^ (uint)counter))
            : null;
        var mask = Engine.TensorDropoutMask<T>([batch, 1, 1, 1], rate, scale, seed);
        return Engine.TensorMultiply(branch, mask);
    }

    private void ResolveSubLayers(int height, int width)
    {
        _fc1.ResolveFromShape([_channels, height, width]);
        _fc1Bn.ResolveFromShape([1, _channels, height, width]);
        _maxRelativeProjection.ResolveFromShape([_channels * 2, height, width]);
        _maxRelativeBn.ResolveFromShape([1, _channels * 2, height, width]);
        _fc2.ResolveFromShape([_channels * 2, height, width]);
        _fc2Bn.ResolveFromShape([1, _channels, height, width]);
        _ffn1.ResolveFromShape([_channels, height, width]);
        _ffn1Bn.ResolveFromShape([1, _channels * 4, height, width]);
        _ffn2.ResolveFromShape([_channels * 4, height, width]);
        _ffn2Bn.ResolveFromShape([1, _channels, height, width]);
    }

    /// <inheritdoc />
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var culture = System.Globalization.CultureInfo.InvariantCulture;
        metadata["Channels"] = _channels.ToString(culture);
        metadata["K"] = _k.ToString(culture);
        metadata["Dilation"] = _dilation.ToString(culture);
        metadata["DropPathRate"] = _dropPathRate.ToString("R", culture);
        return metadata;
    }

    int ILayerSerializationExtras<T>.ExtraParameterCount
    {
        get
        {
            int count = 0;
            foreach (var bn in BatchNormLayers())
                if (bn is ILayerSerializationExtras<T> extras)
                    count += extras.ExtraParameterCount;
            return count;
        }
    }

    Vector<T> ILayerSerializationExtras<T>.GetExtraParameters()
    {
        var values = new List<T>();
        foreach (var bn in BatchNormLayers())
            if (bn is ILayerSerializationExtras<T> extras)
                values.AddRange(extras.GetExtraParameters().ToArray());
        return new Vector<T>(values.ToArray());
    }

    void ILayerSerializationExtras<T>.SetExtraParameters(Vector<T> extraParameters)
    {
        if (!IsShapeResolved)
        {
            _pendingExtraParameters = extraParameters;
            return;
        }
        ApplyExtraParameters(extraParameters);
    }

    private void ApplyExtraParameters(Vector<T> extraParameters)
    {
        int offset = 0;
        foreach (var bn in BatchNormLayers())
        {
            if (bn is not ILayerSerializationExtras<T> extras) continue;
            int count = extras.ExtraParameterCount;
            if (offset + count > extraParameters.Length)
                throw new ArgumentException(
                    $"Truncated GraFPrint graph-block state: need {offset + count} values but received {extraParameters.Length}.",
                    nameof(extraParameters));
            if (count > 0) extras.SetExtraParameters(extraParameters.SubVector(offset, count));
            offset += count;
        }
        if (offset != extraParameters.Length)
            throw new ArgumentException(
                $"GraFPrint graph-block state has {extraParameters.Length - offset} trailing values.",
                nameof(extraParameters));
    }

    private IEnumerable<BatchNormalizationLayer<T>> BatchNormLayers()
    {
        yield return _fc1Bn;
        yield return _maxRelativeBn;
        yield return _fc2Bn;
        yield return _ffn1Bn;
        yield return _ffn2Bn;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var layer in GetSubLayers()) layer.ResetState();
        LastNeighborIndices = null;
        _toNodeIndices = null;
        _toSpatialIndices = null;
        _cachedPermutationShape = null;
        _dropPathCounter = 0;
    }

    private static void GetDimensions(
        Tensor<T> input, out int batch, out int channels, out int height, out int width)
    {
        if (input.Rank == 3)
        {
            batch = 1;
            channels = input.Shape[0];
            height = input.Shape[1];
            width = input.Shape[2];
            return;
        }
        if (input.Rank == 4)
        {
            batch = input.Shape[0];
            channels = input.Shape[1];
            height = input.Shape[2];
            width = input.Shape[3];
            return;
        }
        throw new ArgumentException(
            $"GraFPrint graph block requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; received rank {input.Rank}.",
            nameof(input));
    }
}
