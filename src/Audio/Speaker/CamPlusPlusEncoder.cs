using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Audio.Speaker;

/// <summary>The sizes of a CAM++ speaker encoder (3D-Speaker <c>CAMPPlus</c>).</summary>
/// <param name="FeatureDim">Filterbank bands; a multiple of 8, as the FCM head halves the frequency axis three times.</param>
/// <param name="EmbeddingSize">The speaker embedding's width (192 in the released models).</param>
/// <param name="GrowthRate">Channels each D-TDNN layer adds (32).</param>
/// <param name="BottleneckMultiplier">A D-TDNN layer's bottleneck is this times the growth rate (<c>bn_size</c>, 4).</param>
/// <param name="InitialChannels">The TDNN layer's output channels (128).</param>
internal sealed record CamPlusPlusConfiguration(int FeatureDim = 80, int EmbeddingSize = 192, int GrowthRate = 32,
    int BottleneckMultiplier = 4, int InitialChannels = 128)
{
    /// <summary>The FCM head's channels (<c>m_channels</c>, fixed at 32 in the reference).</summary>
    public const int HeadChannels = 32;

    /// <summary>Layers, kernel and dilation of the three D-TDNN blocks.</summary>
    public static readonly (int Layers, int Kernel, int Dilation)[] BlockShapes = { (12, 3, 1), (24, 3, 2), (16, 3, 2) };

    /// <summary>Frames the context-aware masking pools over per segment (<c>seg_len</c>).</summary>
    public const int SegmentLength = 100;
}

/// <summary>
/// CAM++ (Wang et al. 2023): a speaker encoder of a 2-D convolutional front-end module (FCM: residual blocks over the
/// time-frequency plane), a TDNN layer, three densely connected TDNN blocks whose layers gate their output with
/// context-aware masking, statistics pooling, and a linear embedding with a non-affine BatchNorm.
/// </summary>
/// <remarks>
/// <para>Ported from modelscope/3D-Speaker (<c>speakerlab/models/campplus</c>, Apache-2.0); state-dict names follow it
/// (<c>head.*</c>, <c>xvector.*</c>). Input is a batch of filterbank frames <c>[B, T, FeatureDim]</c>.</para>
/// <para>Context-aware masking (§2.2): each layer's local output is scaled by
/// <c>σ(W₂ ReLU(W₁ (mean_T(x) + segment_mean(x))))</c>, where the segment mean averages 100-frame segments (the last one
/// shorter) and spreads each back over its frames.</para>
/// <para>The embedding's BatchNorm has no affine parameters (<c>batchnorm_</c>); it is computed here, with running
/// statistics in <see cref="EmbeddingRunningMean"/> and <see cref="EmbeddingRunningVariance"/> (momentum 0.1, the
/// unbiased batch variance), since <see cref="BatchNormalizationLayer{T}"/> always trains a scale and shift.</para>
/// </remarks>
internal sealed class CamPlusPlusEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const double Epsilon = 1e-5;
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _owned = new();

    internal sealed record ResBlock(BiasFreeLinearLayer<T> Conv1, BatchNormalizationLayer<T> Norm1, BiasFreeLinearLayer<T> Conv2,
        BatchNormalizationLayer<T> Norm2, BiasFreeLinearLayer<T>? ShortcutConv, BatchNormalizationLayer<T>? ShortcutNorm, int Stride, int Inputs);

    internal sealed record DenseLayer(BatchNormalizationLayer<T> Norm1, NormedConv1DLayer<T> Bottleneck, BatchNormalizationLayer<T> Norm2,
        NormedConv1DLayer<T> Local, NormedConv1DLayer<T> Context1, NormedConv1DLayer<T> Context2);

    internal sealed record Transit(BatchNormalizationLayer<T> Norm, NormedConv1DLayer<T> Conv);

    public CamPlusPlusEncoder(IEngine engine, List<LayerBase<T>> layers, CamPlusPlusConfiguration configuration, Random initialization)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.FeatureDim % 8 != 0) throw new ArgumentException("The feature dimension must be a multiple of 8.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); _owned.Add(layer); return layer; }
        double Gaussian() => Math.Sqrt(-2.0 * Math.Log(1.0 - initialization.NextDouble())) * Math.Cos(2.0 * Math.PI * initialization.NextDouble());
        // nn.Conv2d's default initialization (Kaiming uniform, a = √5): U(±1/√fan_in).
        BiasFreeLinearLayer<T> Kernel(int outputs, int inputs, int size)
        {
            int count = outputs * inputs * size * size;
            var holder = Add(new BiasFreeLinearLayer<T>(1, count));
            TorchParameters.Linear(holder, 1, count, Enumerable.Range(0, count)
                .Select(_ => (2 * initialization.NextDouble() - 1) / Math.Sqrt(inputs * size * size)).ToArray());
            return holder;
        }
        BatchNormalizationLayer<T> Norm(int channels, bool sequence) =>
            Add(new BatchNormalizationLayer<T>(channels, Epsilon, 0.9) { Layout = sequence ? BatchNormDataLayout.ChannelsFirst : BatchNormDataLayout.Infer });
        // The reference re-initializes every Conv1d with kaiming_normal_ (fan_in, gain √2) and zero biases.
        NormedConv1DLayer<T> Conv1d(int inputs, int outputs, int kernel, int stride, int dilation, int padding, bool bias)
        {
            var conv = Add(new NormedConv1DLayer<T>(inputs, outputs, kernel, stride, dilation, 1, padding, false,
                ConvolutionNormalization.None, useBias: bias));
            double std = Math.Sqrt(2.0 / (inputs * kernel));
            conv.Reinitialize(() => std * Gaussian());
            return conv;
        }

        int m = CamPlusPlusConfiguration.HeadChannels;
        HeadConv1 = Kernel(m, 1, 3);
        HeadNorm1 = Norm(m, false);
        foreach (var stage in new[] { HeadStage1, HeadStage2 })
            for (int i = 0; i < 2; i++)
            {
                int stride = i == 0 ? 2 : 1;
                stage.Add(new ResBlock(Kernel(m, m, 3), Norm(m, false), Kernel(m, m, 3), Norm(m, false),
                    stride != 1 ? Kernel(m, m, 1) : null, stride != 1 ? Norm(m, false) : null, stride, m));
            }
        HeadConv2 = Kernel(m, m, 3);
        HeadNorm2 = Norm(m, false);

        int channels = m * (c.FeatureDim / 8);
        Tdnn = Conv1d(channels, c.InitialChannels, 5, 2, 1, 2, false);
        TdnnNorm = Norm(c.InitialChannels, true);
        channels = c.InitialChannels;
        int bottleneck = c.BottleneckMultiplier * c.GrowthRate;
        foreach (var (count, kernel, dilation) in CamPlusPlusConfiguration.BlockShapes)
        {
            var block = new List<DenseLayer>();
            for (int i = 0; i < count; i++)
            {
                int inputs = channels + i * c.GrowthRate;
                block.Add(new DenseLayer(Norm(inputs, true), Conv1d(inputs, bottleneck, 1, 1, 1, 0, false), Norm(bottleneck, true),
                    Conv1d(bottleneck, c.GrowthRate, kernel, 1, dilation, (kernel - 1) / 2 * dilation, false),
                    Conv1d(bottleneck, bottleneck / 2, 1, 1, 1, 0, true), Conv1d(bottleneck / 2, c.GrowthRate, 1, 1, 1, 0, true)));
            }
            Blocks.Add(block);
            channels += count * c.GrowthRate;
            Transits.Add(new Transit(Norm(channels, true), Conv1d(channels, channels / 2, 1, 1, 1, 0, false)));
            channels /= 2;
        }
        OutputNorm = Norm(channels, true);
        Dense = Conv1d(2 * channels, c.EmbeddingSize, 1, 1, 1, 0, false);
        EmbeddingRunningMean = new Tensor<T>(new[] { c.EmbeddingSize });
        EmbeddingRunningVariance = new Tensor<T>(new[] { c.EmbeddingSize }, new Vector<T>(Enumerable.Repeat(NumOps.One, c.EmbeddingSize).ToArray()));
    }

    public CamPlusPlusConfiguration Configuration { get; }
    public BiasFreeLinearLayer<T> HeadConv1 { get; }
    public BatchNormalizationLayer<T> HeadNorm1 { get; }
    public List<ResBlock> HeadStage1 { get; } = new();
    public List<ResBlock> HeadStage2 { get; } = new();
    public BiasFreeLinearLayer<T> HeadConv2 { get; }
    public BatchNormalizationLayer<T> HeadNorm2 { get; }
    public NormedConv1DLayer<T> Tdnn { get; }
    public BatchNormalizationLayer<T> TdnnNorm { get; }
    public List<List<DenseLayer>> Blocks { get; } = new();
    public List<Transit> Transits { get; } = new();
    public BatchNormalizationLayer<T> OutputNorm { get; }
    public NormedConv1DLayer<T> Dense { get; }

    /// <summary>The non-affine embedding BatchNorm's running mean.</summary>
    public Tensor<T> EmbeddingRunningMean { get; private set; }

    /// <summary>The non-affine embedding BatchNorm's running variance.</summary>
    public Tensor<T> EmbeddingRunningVariance { get; private set; }

    /// <summary>Puts every layer in training (batch statistics) or evaluation (running statistics) mode.</summary>
    public void SetTrainingMode(bool training)
    {
        foreach (var layer in _owned) layer.SetTrainingMode(training);
        Training = training;
    }

    /// <summary>Whether the encoder normalizes with batch statistics (and updates the running ones).</summary>
    public bool Training { get; private set; } = true;

    private Tensor<T> Weight(BiasFreeLinearLayer<T> holder, int outputs, int inputs, int size) =>
        _engine.Reshape(holder.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One }))), new[] { outputs, inputs, size, size });

    private Tensor<T> Conv2d(Tensor<T> x, BiasFreeLinearLayer<T> kernel, int inputs, int size, int strideHeight)
    {
        int m = CamPlusPlusConfiguration.HeadChannels, pad = size / 2;
        return _engine.Conv2D(x, Weight(kernel, m, inputs, size), new[] { strideHeight, 1 }, new[] { pad, pad }, new[] { 1, 1 });
    }

    /// <summary>The FCM head: <c>[B, F, T]</c> to <c>[B, 32·F/8, T]</c>.</summary>
    public Tensor<T> Head(Tensor<T> features)
    {
        int b = features.Shape[0], f = features.Shape[1], t = features.Shape[2];
        var x = _engine.ReLU(HeadNorm1.Forward(Conv2d(_engine.Reshape(features, new[] { b, 1, f, t }), HeadConv1, 1, 3, 1)));
        foreach (var block in HeadStage1.Concat(HeadStage2))
        {
            var y = _engine.ReLU(block.Norm1.Forward(Conv2d(x, block.Conv1, block.Inputs, 3, block.Stride)));
            y = block.Norm2.Forward(Conv2d(y, block.Conv2, CamPlusPlusConfiguration.HeadChannels, 3, 1));
            var shortcut = block.ShortcutConv is not null && block.ShortcutNorm is not null
                ? block.ShortcutNorm.Forward(Conv2d(x, block.ShortcutConv, block.Inputs, 1, block.Stride))
                : x;
            x = _engine.ReLU(_engine.TensorAdd(y, shortcut));
        }
        x = _engine.ReLU(HeadNorm2.Forward(Conv2d(x, HeadConv2, CamPlusPlusConfiguration.HeadChannels, 3, 2)));
        return _engine.Reshape(x, new[] { b, x.Shape[1] * x.Shape[2], x.Shape[3] });
    }

    /// <summary>The context <c>mean_T(x) + segment_mean(x)</c> of <c>[B, C, T]</c>, as <c>[B, C, T]</c>.</summary>
    private Tensor<T> Context(Tensor<T> x)
    {
        int b = x.Shape[0], c = x.Shape[1], t = x.Shape[2], seg = CamPlusPlusConfiguration.SegmentLength;
        int segments = (t + seg - 1) / seg;
        // avg_pool1d(kernel = stride = 100, ceil_mode) then each segment's mean repeated over its frames; the global mean
        // is added in the same matrix: averaging [T, S] then spreading [S, T].
        var average = new Tensor<T>(new[] { t, segments + 1 });
        var spread = new Tensor<T>(new[] { segments + 1, t });
        for (int i = 0; i < t; i++)
        {
            int s = i / seg, length = Math.Min(seg, t - s * seg);
            average[i, s] = NumOps.FromDouble(1.0 / length);
            average[i, segments] = NumOps.FromDouble(1.0 / t);
            spread[s, i] = NumOps.One;
            spread[segments, i] = NumOps.One;
        }
        var pooled = _engine.TensorMatMul(_engine.TensorMatMul(_engine.Reshape(x, new[] { b * c, t }), average), spread);
        return _engine.Reshape(pooled, new[] { b, c, t });
    }

    /// <summary>The x-vector trunk from the head's output to the pooled statistics <c>[B, 2C]</c>.</summary>
    public Tensor<T> Pool(Tensor<T> head)
    {
        var x = _engine.ReLU(TdnnNorm.Forward(Tdnn.Forward(head)));
        for (int i = 0; i < Blocks.Count; i++)
        {
            foreach (var layer in Blocks[i])
            {
                var bottleneck = _engine.ReLU(layer.Norm2.Forward(layer.Bottleneck.Forward(_engine.ReLU(layer.Norm1.Forward(x)))));
                var local = layer.Local.Forward(bottleneck);
                var gate = _engine.Sigmoid(layer.Context2.Forward(_engine.ReLU(layer.Context1.Forward(Context(bottleneck)))));
                x = _engine.TensorConcatenate(new[] { x, _engine.TensorMultiply(local, gate) }, 1);
            }
            x = Transits[i].Conv.Forward(_engine.ReLU(Transits[i].Norm.Forward(x)));
        }
        x = _engine.ReLU(OutputNorm.Forward(x));
        int frames = x.Shape[2];
        var mean = _engine.ReduceMean(x, new[] { 2 }, true);
        var centred = _engine.TensorSubtract(x, _engine.TensorBroadcastTo(mean, x.Shape.ToArray()));
        // The unbiased standard deviation (torch.std's default).
        var variance = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(centred, centred), new[] { 2 }, false),
            NumOps.FromDouble(1.0 / (frames - 1)));
        return _engine.TensorConcatenate(new[] { _engine.ReduceMean(x, new[] { 2 }, false), _engine.TensorSqrt(variance) }, 1);
    }

    /// <summary>Speaker embeddings <c>[B, EmbeddingSize]</c> of filterbank frames <c>[B, T, FeatureDim]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> features)
    {
        var stats = Pool(Head(_engine.TensorPermute(features, new[] { 0, 2, 1 })));
        int b = stats.Shape[0], width = stats.Shape[1], e = Configuration.EmbeddingSize;
        var x = _engine.Reshape(Dense.Forward(_engine.Reshape(stats, new[] { b, width, 1 })), new[] { b, e });
        return NormalizeEmbedding(x);
    }

    private Tensor<T> NormalizeEmbedding(Tensor<T> x)
    {
        int b = x.Shape[0], e = x.Shape[1];
        var shape = new[] { b, e };
        if (!Training)
        {
            var scale = new Tensor<T>(new[] { 1, e });
            for (int j = 0; j < e; j++) scale[0, j] = NumOps.FromDouble(1.0 / Math.Sqrt(NumOps.ToDouble(EmbeddingRunningVariance[j]) + Epsilon));
            var centredRunning = _engine.TensorSubtract(x, _engine.TensorBroadcastTo(_engine.Reshape(EmbeddingRunningMean, new[] { 1, e }), shape));
            return _engine.TensorMultiply(centredRunning, _engine.TensorBroadcastTo(scale, shape));
        }
        if (b < 2) throw new ArgumentException("Training-mode BatchNorm needs more than one embedding per batch.", nameof(x));
        var mean = _engine.ReduceMean(x, new[] { 0 }, true);
        var centred = _engine.TensorSubtract(x, _engine.TensorBroadcastTo(mean, shape));
        var variance = _engine.ReduceMean(_engine.TensorMultiply(centred, centred), new[] { 0 }, true);
        var output = _engine.TensorDivide(centred,
            _engine.TensorBroadcastTo(_engine.TensorSqrt(_engine.TensorAddScalar(variance, NumOps.FromDouble(Epsilon))), shape));
        // running ← 0.9 running + 0.1 batch, the variance unbiased; off the tape.
        var updatedMean = new Tensor<T>(new[] { e });
        var updatedVariance = new Tensor<T>(new[] { e });
        for (int j = 0; j < e; j++)
        {
            updatedMean[j] = NumOps.FromDouble(0.9 * NumOps.ToDouble(EmbeddingRunningMean[j]) + 0.1 * NumOps.ToDouble(mean[0, j]));
            updatedVariance[j] = NumOps.FromDouble(0.9 * NumOps.ToDouble(EmbeddingRunningVariance[j])
                + 0.1 * NumOps.ToDouble(variance[0, j]) * b / (b - 1));
        }
        EmbeddingRunningMean = updatedMean;
        EmbeddingRunningVariance = updatedVariance;
        return output;
    }

    /// <summary>Loads a 3D-Speaker <c>CAMPPlus</c> state dict (<c>num_batches_tracked</c> is read and not used).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int m = CamPlusPlusConfiguration.HeadChannels;
        void Kernel(BiasFreeLinearLayer<T> holder, string name, int outputs, int inputs, int size)
        {
            int count = outputs * inputs * size * size;
            TorchParameters.Linear(holder, 1, count, read(name, new[] { outputs, inputs, size, size }));
        }
        void Norm(BatchNormalizationLayer<T> norm, string name, int channels)
        {
            var shape = new[] { channels };
            TorchParameters.BatchNorm(norm, read(name + ".weight", shape), read(name + ".bias", shape), read(name + ".running_mean", shape),
                read(name + ".running_var", shape));
            read(name + ".num_batches_tracked", Array.Empty<int>());
        }
        void Conv(NormedConv1DLayer<T> conv, string name, int inputs, int kernel, bool bias) =>
            conv.LoadTorchWeights(read(name + ".weight", new[] { conv.OutputChannels, inputs, kernel }), null,
                bias ? read(name + ".bias", new[] { conv.OutputChannels }) : new double[conv.OutputChannels]);

        Kernel(HeadConv1, "head.conv1.weight", m, 1, 3);
        Norm(HeadNorm1, "head.bn1", m);
        foreach (var (stage, name) in new[] { (HeadStage1, "head.layer1"), (HeadStage2, "head.layer2") })
            for (int i = 0; i < stage.Count; i++)
            {
                var b = stage[i];
                string p = $"{name}.{i}";
                Kernel(b.Conv1, p + ".conv1.weight", m, m, 3);
                Norm(b.Norm1, p + ".bn1", m);
                Kernel(b.Conv2, p + ".conv2.weight", m, m, 3);
                Norm(b.Norm2, p + ".bn2", m);
                if (b.ShortcutConv is not null && b.ShortcutNorm is not null)
                {
                    Kernel(b.ShortcutConv, p + ".shortcut.0.weight", m, m, 1);
                    Norm(b.ShortcutNorm, p + ".shortcut.1", m);
                }
            }
        Kernel(HeadConv2, "head.conv2.weight", m, m, 3);
        Norm(HeadNorm2, "head.bn2", m);

        int channels = m * (c.FeatureDim / 8);
        Conv(Tdnn, "xvector.tdnn.linear", channels, 5, false);
        Norm(TdnnNorm, "xvector.tdnn.nonlinear.batchnorm", c.InitialChannels);
        channels = c.InitialChannels;
        int bottleneck = c.BottleneckMultiplier * c.GrowthRate;
        for (int i = 0; i < Blocks.Count; i++)
        {
            var (_, kernel, _) = CamPlusPlusConfiguration.BlockShapes[i];
            for (int j = 0; j < Blocks[i].Count; j++)
            {
                var l = Blocks[i][j];
                string p = $"xvector.block{i + 1}.tdnnd{j + 1}";
                int inputs = channels + j * c.GrowthRate;
                Norm(l.Norm1, p + ".nonlinear1.batchnorm", inputs);
                Conv(l.Bottleneck, p + ".linear1", inputs, 1, false);
                Norm(l.Norm2, p + ".nonlinear2.batchnorm", bottleneck);
                Conv(l.Local, p + ".cam_layer.linear_local", bottleneck, kernel, false);
                Conv(l.Context1, p + ".cam_layer.linear1", bottleneck, 1, true);
                Conv(l.Context2, p + ".cam_layer.linear2", bottleneck / 2, 1, true);
            }
            channels += Blocks[i].Count * c.GrowthRate;
            Norm(Transits[i].Norm, $"xvector.transit{i + 1}.nonlinear.batchnorm", channels);
            Conv(Transits[i].Conv, $"xvector.transit{i + 1}.linear", channels, 1, false);
            channels /= 2;
        }
        Norm(OutputNorm, "xvector.out_nonlinear.batchnorm", channels);
        Conv(Dense, "xvector.dense.linear", 2 * channels, 1, false);
        var e = new[] { c.EmbeddingSize };
        EmbeddingRunningMean = new Tensor<T>(e, new Vector<T>(read("xvector.dense.nonlinear.batchnorm.running_mean", e).Select(NumOps.FromDouble).ToArray()));
        EmbeddingRunningVariance = new Tensor<T>(e, new Vector<T>(read("xvector.dense.nonlinear.batchnorm.running_var", e).Select(NumOps.FromDouble).ToArray()));
        read("xvector.dense.nonlinear.batchnorm.num_batches_tracked", Array.Empty<int>());
    }
}
