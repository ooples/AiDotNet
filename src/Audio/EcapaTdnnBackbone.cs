using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio;

/// <summary>
/// The ECAPA-TDNN encoder (Desplanques et al. 2020, arXiv:2005.07143), built from real 1-D
/// convolutions over time and shared by every model that uses it.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The layout follows the paper's Figure 2 and the SpeechBrain reference (<c>ECAPA_TDNN</c>), whose
/// hyper-parameter lists this class takes verbatim: <c>channels</c>, <c>kernelSizes</c> and
/// <c>dilations</c> each hold one entry per stage, where stage 0 is the frame-level TDNN block, the
/// middle entries are SE-Res2Blocks and the last entry is the multi-layer feature aggregation (MFA)
/// convolution. The paper's C=1024 model is <c>[1024, 1024, 1024, 1024, 3072]</c>,
/// <c>[5, 3, 3, 3, 1]</c>, <c>[1, 2, 3, 4, 1]</c>.
/// </para>
/// <list type="bullet">
/// <item>TDNN block: Conv1d, ReLU, BatchNorm1d.</item>
/// <item>SE-Res2Block: TDNN 1x1, a Res2Net block of <c>scale - 1</c> dilated TDNN blocks over channel
/// splits (y1 = x1, yi = TDNN(xi + y(i-1))), TDNN 1x1, squeeze-excitation over the time mean, and a
/// residual connection (with a 1x1 shortcut when the width changes).</item>
/// <item>MFA: the SE-Res2Block outputs concatenated on the channel axis, then a TDNN block.</item>
/// <item>Channel- and context-dependent attentive statistics pooling: the frame features are joined
/// with their global mean and standard deviation, scored by TDNN, tanh and a 1x1 convolution, softmaxed
/// over time, and reduced to a weighted mean and standard deviation.</item>
/// <item>BatchNorm1d over the pooled statistics and a linear projection to the embedding.</item>
/// </list>
/// <para>
/// Every step is an engine operation, so the gradient tape differentiates the whole encoder. The
/// layers are created once and exposed through <see cref="Layers"/>; the owning model publishes those
/// exact instances so training, serialization and clone walk the same weights the forward uses.
/// </para>
/// </remarks>
internal sealed class EcapaTdnnBackbone<T>
{
    private const double StatisticsEpsilon = 1e-12;

    private readonly int _res2NetScale;
    private readonly int[] _channels;
    private readonly int[] _kernelSizes;
    private readonly int[] _dilations;
    private readonly int _seChannels;
    private readonly int _attentionChannels;
    private readonly int _embeddingDimension;
    private readonly List<ILayer<T>> _layers = new();

    // Roles over _layers, assigned by Build.
    private TdnnBlock _frameBlock;
    private readonly List<SeRes2Block> _blocks = new();
    private TdnnBlock _mfaBlock;
    private TdnnBlock _attentionTdnn;
    private Conv1DLayer<T> _attentionScore;
    private BatchNormalizationLayer<T> _poolingNorm;
    private DenseLayer<T> _embedding;

    /// <summary>Creates the encoder.</summary>
    /// <param name="channels">Output width of each stage; the last is the MFA width.</param>
    /// <param name="kernelSizes">Kernel width of each stage (the SE-Res2Block entry is its Res2Net kernel).</param>
    /// <param name="dilations">Dilation of each stage.</param>
    /// <param name="res2NetScale">Number of Res2Net channel splits (8 in the paper).</param>
    /// <param name="seChannels">Squeeze-excitation bottleneck width (128 in the paper).</param>
    /// <param name="attentionChannels">Attentive-pooling bottleneck width (128 in the paper).</param>
    /// <param name="embeddingDimension">Width of the output embedding (192 in the paper).</param>
    public EcapaTdnnBackbone(
        int[] channels,
        int[] kernelSizes,
        int[] dilations,
        int res2NetScale,
        int seChannels,
        int attentionChannels,
        int embeddingDimension)
    {
        if (channels is null) throw new ArgumentNullException(nameof(channels));
        if (kernelSizes is null) throw new ArgumentNullException(nameof(kernelSizes));
        if (dilations is null) throw new ArgumentNullException(nameof(dilations));
        if (channels.Length < 3)
            throw new ArgumentException(
                "ECAPA-TDNN needs at least three stages: the frame-level TDNN block, one SE-Res2Block and the MFA convolution.",
                nameof(channels));
        if (kernelSizes.Length != channels.Length || dilations.Length != channels.Length)
            throw new ArgumentException(
                $"channels ({channels.Length}), kernelSizes ({kernelSizes.Length}) and dilations ({dilations.Length}) " +
                "must have one entry per stage.", nameof(kernelSizes));
        for (int i = 0; i < channels.Length; i++)
        {
            if (channels[i] <= 0) throw new ArgumentOutOfRangeException(nameof(channels), $"channels[{i}] must be positive.");
            if (kernelSizes[i] <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSizes), $"kernelSizes[{i}] must be positive.");
            if (dilations[i] <= 0) throw new ArgumentOutOfRangeException(nameof(dilations), $"dilations[{i}] must be positive.");
        }
        if (res2NetScale <= 0) throw new ArgumentOutOfRangeException(nameof(res2NetScale));
        if (seChannels <= 0) throw new ArgumentOutOfRangeException(nameof(seChannels));
        if (attentionChannels <= 0) throw new ArgumentOutOfRangeException(nameof(attentionChannels));
        if (embeddingDimension <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDimension));
        for (int i = 1; i < channels.Length - 1; i++)
        {
            if (channels[i] % res2NetScale != 0)
                throw new ArgumentException(
                    $"SE-Res2Block width channels[{i}] ({channels[i]}) must be divisible by the Res2Net scale ({res2NetScale}).",
                    nameof(res2NetScale));
        }

        _res2NetScale = res2NetScale;
        _channels = (int[])channels.Clone();
        _kernelSizes = (int[])kernelSizes.Clone();
        _dilations = (int[])dilations.Clone();
        _seChannels = seChannels;
        _attentionChannels = attentionChannels;
        _embeddingDimension = embeddingDimension;

        Build();
    }

    /// <summary>The encoder's layers, in a fixed order, for the owning model to publish.</summary>
    public IReadOnlyList<ILayer<T>> Layers => _layers;

    /// <summary>
    /// Points every role at the layers of <paramref name="layers"/>, position for position.
    /// </summary>
    /// <remarks>
    /// A deserialize or eager clone replaces the owning model's layer instances. The model rebinds its
    /// own layer-typed members, but not ones held inside this object, so the model calls this before a
    /// forward; it does nothing while the graph is still the one this encoder built or last bound.
    /// </remarks>
    /// <param name="layers">The model's current layers; the encoder's are its first entries.</param>
    public void BindTo(IReadOnlyList<ILayer<T>> layers)
    {
        if (layers is null) throw new ArgumentNullException(nameof(layers));
        bool same = layers.Count >= _layers.Count;
        for (int i = 0; same && i < _layers.Count; i++)
        {
            same = ReferenceEquals(layers[i], _layers[i]);
        }

        if (same) return;

        _bindSource = layers;
        try
        {
            _layers.Clear();
            _blocks.Clear();
            Build();
        }
        finally
        {
            _bindSource = null;
        }
    }

    [System.Diagnostics.CodeAnalysis.MemberNotNull(
        nameof(_frameBlock), nameof(_mfaBlock), nameof(_attentionTdnn),
        nameof(_attentionScore), nameof(_poolingNorm), nameof(_embedding))]
    private void Build()
    {
        _frameBlock = AddTdnn(_channels[0], _kernelSizes[0], _dilations[0]);

        int previousWidth = _channels[0];
        for (int i = 1; i < _channels.Length - 1; i++)
        {
            _blocks.Add(AddSeRes2Block(previousWidth, _channels[i], _kernelSizes[i], _dilations[i], _seChannels));
            previousWidth = _channels[i];
        }

        int mfaWidth = _channels[_channels.Length - 1];
        _mfaBlock = AddTdnn(mfaWidth, _kernelSizes[_kernelSizes.Length - 1], _dilations[_dilations.Length - 1]);

        _attentionTdnn = AddTdnn(_attentionChannels, 1, 1);
        _attentionScore = Add(new Conv1DLayer<T>(mfaWidth, 1));
        _poolingNorm = Add(new BatchNormalizationLayer<T>());
        _embedding = Add(new DenseLayer<T>(_embeddingDimension, (IActivationFunction<T>)new IdentityActivation<T>()));
    }

    /// <summary>
    /// Encodes channel-first features <c>[B, F, T]</c> into embeddings <c>[B, embeddingDimension]</c>.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> features)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (features.Shape.Length != 3)
            throw new ArgumentException(
                $"ECAPA-TDNN expects channel-first features [B, F, T]; got rank {features.Shape.Length}.",
                nameof(features));

        var engine = AiDotNetEngine.Current;
        var x = _frameBlock.Forward(engine, features);

        var blockOutputs = new Tensor<T>[_blocks.Count];
        for (int i = 0; i < _blocks.Count; i++)
        {
            x = _blocks[i].Forward(engine, x, _res2NetScale);
            blockOutputs[i] = x;
        }

        var aggregated = _mfaBlock.Forward(engine, engine.TensorConcatenate(blockOutputs, axis: 1));
        var pooled = AttentiveStatisticsPooling(engine, aggregated);
        return _embedding.Forward(_poolingNorm.Forward(pooled));
    }

    /// <summary>
    /// Converts a time-major feature matrix <c>[T, F]</c> (or a batch <c>[B, T, F]</c>) to the
    /// channel-first <c>[B, F, T]</c> layout the convolutions take.
    /// </summary>
    public static Tensor<T> ToChannelFirst(Tensor<T> timeMajorFeatures)
    {
        var engine = AiDotNetEngine.Current;
        var batched = timeMajorFeatures.Shape.Length switch
        {
            2 => engine.Reshape(timeMajorFeatures, new[] { 1, timeMajorFeatures.Shape[0], timeMajorFeatures.Shape[1] }),
            3 => timeMajorFeatures,
            _ => throw new ArgumentException(
                $"ECAPA-TDNN features must be time-major [T, F] or [B, T, F]; got rank {timeMajorFeatures.Shape.Length}.",
                nameof(timeMajorFeatures))
        };
        return engine.TensorPermute(batched, new[] { 0, 2, 1 });
    }

    private Tensor<T> AttentiveStatisticsPooling(IEngine engine, Tensor<T> x)
    {
        // Context: the frames joined with their global (uniform-weight) mean and standard deviation.
        int batch = x.Shape[0], width = x.Shape[1], frames = x.Shape[2];
        var (globalMean, globalStd) = WeightedStatistics(engine, x, weights: null);
        var full = new[] { batch, width, frames };
        var context = engine.TensorConcatenate(
            new[]
            {
                x,
                engine.TensorBroadcastTo(engine.Reshape(globalMean, new[] { batch, width, 1 }), full),
                engine.TensorBroadcastTo(engine.Reshape(globalStd, new[] { batch, width, 1 }), full)
            },
            axis: 1);

        // Per-channel, per-frame attention weights, normalised over time.
        var scores = _attentionScore.Forward(engine.Tanh(_attentionTdnn.Forward(engine, context)));
        var weights = engine.TensorSoftmax(scores, axis: 2);

        var (mean, std) = WeightedStatistics(engine, x, weights);
        return engine.TensorConcatenate(new[] { mean, std }, axis: 1);
    }

    private static (Tensor<T> Mean, Tensor<T> Std) WeightedStatistics(IEngine engine, Tensor<T> x, Tensor<T>? weights)
    {
        Tensor<T> mean;
        Tensor<T> variance;
        if (weights is null)
        {
            mean = engine.ReduceMean(x, new[] { 2 }, keepDims: true);
            var centered = engine.TensorSubtract(x, mean);
            variance = engine.ReduceMean(engine.TensorMultiply(centered, centered), new[] { 2 }, keepDims: false);
        }
        else
        {
            mean = engine.ReduceSum(engine.TensorMultiply(weights, x), new[] { 2 }, keepDims: true);
            var centered = engine.TensorSubtract(x, mean);
            variance = engine.ReduceSum(
                engine.TensorMultiply(weights, engine.TensorMultiply(centered, centered)), new[] { 2 }, keepDims: false);
        }

        // Both variance forms are weighted sums of squares, so they are non-negative; the epsilon only
        // keeps the square root's gradient finite where a channel is constant over time.
        var numOps = MathHelper.GetNumericOperations<T>();
        var std = engine.TensorSqrt(engine.TensorAddScalar(variance, numOps.FromDouble(StatisticsEpsilon)));
        return (engine.Reshape(mean, new[] { mean.Shape[0], mean.Shape[1] }), std);
    }

    // During BindTo, each construction step takes the layer at the same position in the model's
    // current graph instead of the one it just built, so the roles point at the live instances.
    private IReadOnlyList<ILayer<T>>? _bindSource;

    private TLayer Add<TLayer>(TLayer layer) where TLayer : class, ILayer<T>
    {
        if (_bindSource is not null)
        {
            int index = _layers.Count;
            if (index >= _bindSource.Count || _bindSource[index] is not TLayer existing)
            {
                throw new InvalidOperationException(
                    $"The layer graph does not match the ECAPA-TDNN layout at position {index}: expected " +
                    $"{typeof(TLayer).Name}, found {(index < _bindSource.Count ? _bindSource[index].GetType().Name : "the end")}.");
            }

            layer = existing;
        }

        _layers.Add(layer);
        return layer;
    }

    private TdnnBlock AddTdnn(int outputChannels, int kernelSize, int dilation)
        => new(
            Add(new Conv1DLayer<T>(outputChannels, kernelSize, dilation: dilation, activation: new ReLUActivation<T>())),
            Add(new BatchNormalizationLayer<T>()));

    private SeRes2Block AddSeRes2Block(int inputWidth, int width, int kernelSize, int dilation, int seChannels)
    {
        var reduce = AddTdnn(width, 1, 1);
        var branches = new TdnnBlock[_res2NetScale - 1];
        for (int i = 0; i < branches.Length; i++)
        {
            branches[i] = AddTdnn(width / _res2NetScale, kernelSize, dilation);
        }

        var expand = AddTdnn(width, 1, 1);
        var squeeze = Add(new DenseLayer<T>(seChannels, (IActivationFunction<T>)new ReLUActivation<T>()));
        var excite = Add(new DenseLayer<T>(width, (IActivationFunction<T>)new SigmoidActivation<T>()));
        var shortcut = inputWidth == width ? null : Add(new Conv1DLayer<T>(width, 1));
        return new SeRes2Block(reduce, branches, expand, squeeze, excite, shortcut);
    }

    /// <summary>Conv1d, ReLU (fused into the convolution) and BatchNorm1d.</summary>
    private sealed class TdnnBlock
    {
        private readonly Conv1DLayer<T> _conv;
        private readonly BatchNormalizationLayer<T> _norm;

        public TdnnBlock(Conv1DLayer<T> conv, BatchNormalizationLayer<T> norm)
        {
            _conv = conv;
            _norm = norm;
        }

        public Tensor<T> Forward(IEngine engine, Tensor<T> x) => BatchNorm1d(engine, _norm, _conv.Forward(x));
    }

    private sealed class SeRes2Block
    {
        private readonly TdnnBlock _reduce;
        private readonly TdnnBlock[] _branches;
        private readonly TdnnBlock _expand;
        private readonly DenseLayer<T> _squeeze;
        private readonly DenseLayer<T> _excite;
        private readonly Conv1DLayer<T>? _shortcut;

        public SeRes2Block(
            TdnnBlock reduce, TdnnBlock[] branches, TdnnBlock expand,
            DenseLayer<T> squeeze, DenseLayer<T> excite, Conv1DLayer<T>? shortcut)
        {
            _reduce = reduce;
            _branches = branches;
            _expand = expand;
            _squeeze = squeeze;
            _excite = excite;
            _shortcut = shortcut;
        }

        public Tensor<T> Forward(IEngine engine, Tensor<T> x, int scale)
        {
            var residual = _shortcut is null ? x : _shortcut.Forward(x);
            var h = _reduce.Forward(engine, x);

            // Res2Net: the first split passes through; each later split adds the previous output
            // before its own dilated TDNN block, so the receptive field grows split by split.
            int splitWidth = h.Shape[1] / scale;
            var splits = new Tensor<T>[scale];
            Tensor<T>? previous = null;
            for (int i = 0; i < scale; i++)
            {
                var split = engine.TensorNarrow(h, dim: 1, start: i * splitWidth, length: splitWidth);
                if (i == 0)
                {
                    splits[i] = split;
                    continue;
                }

                var branchInput = previous is null ? split : engine.TensorAdd(split, previous);
                previous = _branches[i - 1].Forward(engine, branchInput);
                splits[i] = previous;
            }

            h = _expand.Forward(engine, engine.TensorConcatenate(splits, axis: 1));

            // Squeeze-excitation: a per-channel gate from the time-averaged activations.
            var squeezed = engine.ReduceMean(h, new[] { 2 }, keepDims: false);
            var gate = _excite.Forward(_squeeze.Forward(squeezed));
            h = engine.TensorMultiply(h, engine.Reshape(gate, new[] { gate.Shape[0], gate.Shape[1], 1 }));

            return engine.TensorAdd(h, residual);
        }
    }

    /// <summary>
    /// BatchNorm1d over <c>[B, C, T]</c>: statistics per channel across every batch item and frame.
    /// BatchNormalizationLayer reads a rank-3 tensor as an unbatched image <c>[C, H, W]</c>, so the
    /// activations are presented to it as <c>[B * T, C]</c> rows.
    /// </summary>
    private static Tensor<T> BatchNorm1d(IEngine engine, BatchNormalizationLayer<T> norm, Tensor<T> x)
    {
        int batch = x.Shape[0], channels = x.Shape[1], frames = x.Shape[2];
        var rows = engine.Reshape(engine.TensorPermute(x, new[] { 0, 2, 1 }), new[] { batch * frames, channels });
        var normalized = engine.Reshape(norm.Forward(rows), new[] { batch, frames, channels });

        // Materialize the permuted view (Reshape of a non-contiguous tensor copies it, on the tape).
        // The next convolution would copy it anyway, and the engine's strided elementwise paths are
        // not safe for every op: its strided Tanh returned NaN above |x| ~44 in float
        // (ooples/AiDotNet.Tensors#1087), which is exactly what the attentive pooling's tanh fed it.
        return engine.Reshape(engine.TensorPermute(normalized, new[] { 0, 2, 1 }), new[] { batch, channels, frames });
    }
}
