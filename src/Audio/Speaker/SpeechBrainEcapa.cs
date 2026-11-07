using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.Audio.Speaker;

/// <summary>The sizes of speechbrain's ECAPA-TDNN extractor (<c>speechbrain/spkrec-ecapa-voxceleb</c>'s hyperparameters).</summary>
internal sealed record SpeechBrainEcapaConfiguration
{
    public int SampleRate { get; init; } = 16000;
    public int Mels { get; init; } = 80;
    public int[] Channels { get; init; } = { 1024, 1024, 1024, 1024, 3072 };
    public int[] Kernels { get; init; } = { 5, 3, 3, 3, 1 };
    public int[] Dilations { get; init; } = { 1, 2, 3, 4, 1 };
    public int AttentionChannels { get; init; } = 128;
    public int Res2NetScale { get; init; } = 8;
    public int SeChannels { get; init; } = 128;
    public int EmbeddingSize { get; init; } = 192;
}

/// <summary>
/// speechbrain 0.5's ECAPA-TDNN speaker embedding (Desplanques et al. 2020) as <c>spkrec-ecapa-voxceleb</c>'s
/// <c>encode_batch</c> computes it, the embedding WhisperSpeech conditions on: the 80-mel <c>Fbank</c> with the sentence
/// mean removed; a TDNN block (convolution with reflect "same" padding, ReLU, BatchNorm); SE-Res2Net blocks (1×1 TDNN,
/// Res2Net over channel chunks, 1×1 TDNN, squeeze-excitation over the time mean, residual through a 1×1 convolution when
/// the width changes); the SE-Res2Net outputs concatenated through the MFA TDNN; attentive statistics pooling with global
/// context (the variance clamped at 1e-12 before the square root); BatchNorm; a 1×1 convolution to the embedding.
/// </summary>
internal sealed class SpeechBrainEcapa<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly SpeechBrainEcapaConfiguration _c;
    private readonly double[,] _filters;

    internal sealed class Tdnn
    {
        public NormedConv1DLayer<T> Conv = null!;
        public BatchNormalizationLayer<T> Norm = null!;
        public int Inputs, Outputs, Kernel, Dilation;
    }

    internal sealed class SeRes2Block
    {
        public Tdnn Reduce = null!, Expand = null!;
        public List<Tdnn> Branches = new();
        public NormedConv1DLayer<T> Squeeze = null!, Excite = null!;
        public NormedConv1DLayer<T>? Shortcut;
    }

    public SpeechBrainEcapa(IEngine engine, List<LayerBase<T>> layers, SpeechBrainEcapaConfiguration configuration, Random random)
    {
        _engine = engine;
        _c = configuration;
        var c = configuration;
        int stages = c.Channels.Length;
        if (stages < 3 || c.Kernels.Length != stages || c.Dilations.Length != stages)
            throw new ArgumentException("ECAPA-TDNN needs a TDNN block, at least one SE-Res2Net block and the MFA block, each with a kernel and dilation.", nameof(configuration));
        NormedConv1DLayer<T> Conv(int inputs, int outputs, int kernel, int dilation)
        {
            var conv = new NormedConv1DLayer<T>(inputs, outputs, kernel, 1, dilation, 1, 0, false, ConvolutionNormalization.None);
            layers.Add(conv);
            conv.LoadTorchWeights(IndexTtsOps.Uniform(random, outputs * inputs * kernel, inputs * kernel), null, IndexTtsOps.Uniform(random, outputs, inputs * kernel));
            return conv;
        }
        BatchNormalizationLayer<T> Norm(int channels)
        {
            var norm = new BatchNormalizationLayer<T>(channels, epsilon: 1e-5, momentum: 0.9);
            layers.Add(norm);
            return norm;
        }
        Tdnn NewTdnn(int inputs, int outputs, int kernel, int dilation) => new()
        {
            Conv = Conv(inputs, outputs, kernel, dilation), Norm = Norm(outputs), Inputs = inputs, Outputs = outputs, Kernel = kernel, Dilation = dilation,
        };
        First = NewTdnn(c.Mels, c.Channels[0], c.Kernels[0], c.Dilations[0]);
        for (int i = 1; i < stages - 1; i++)
        {
            int inputs = c.Channels[i - 1], width = c.Channels[i];
            if (width % c.Res2NetScale != 0) throw new ArgumentException("The Res2Net scale must divide each block's width.", nameof(configuration));
            var block = new SeRes2Block { Reduce = NewTdnn(inputs, width, 1, 1) };
            for (int j = 0; j < c.Res2NetScale - 1; j++)
                block.Branches.Add(NewTdnn(width / c.Res2NetScale, width / c.Res2NetScale, c.Kernels[i], c.Dilations[i]));
            block.Expand = NewTdnn(width, width, 1, 1);
            block.Squeeze = Conv(width, c.SeChannels, 1, 1);
            block.Excite = Conv(c.SeChannels, width, 1, 1);
            if (inputs != width) block.Shortcut = Conv(inputs, width, 1, 1);
            Blocks.Add(block);
        }
        int mfa = c.Channels[stages - 1];
        Mfa = NewTdnn(c.Channels[stages - 2] * (stages - 2), mfa, c.Kernels[stages - 1], c.Dilations[stages - 1]);
        AttentionTdnn = NewTdnn(3 * mfa, c.AttentionChannels, 1, 1);
        AttentionScore = Conv(c.AttentionChannels, mfa, 1, 1);
        PoolingNorm = Norm(2 * mfa);
        Embedding = Conv(2 * mfa, c.EmbeddingSize, 1, 1);
        _filters = SpeechBrainFbank.Filters(c.Mels, c.SampleRate);
        foreach (var norm in layers.OfType<BatchNormalizationLayer<T>>()) norm.SetTrainingMode(false);
    }

    public SpeechBrainEcapaConfiguration Configuration => _c;
    public Tdnn First { get; }
    public List<SeRes2Block> Blocks { get; } = new();
    public Tdnn Mfa { get; }
    public Tdnn AttentionTdnn { get; }
    public NormedConv1DLayer<T> AttentionScore { get; }
    public BatchNormalizationLayer<T> PoolingNorm { get; }
    public NormedConv1DLayer<T> Embedding { get; }

    /// <summary>speechbrain's <c>Fbank</c> of a 16 kHz waveform, the sentence mean removed: <c>[frames, mels]</c>.</summary>
    public double[,] Features(IReadOnlyList<double> wave) => SpeechBrainFbank.Features(wave, _filters);

    // speechbrain's BatchNorm1d over [1, C, L] (evaluation: the running statistics).
    private Tensor<T> Normalize(BatchNormalizationLayer<T> norm, Tensor<T> x)
    {
        int channels = x.Shape[1], length = x.Shape[2];
        return _engine.Reshape(norm.Forward(_engine.Reshape(x, new[] { 1, channels, length, 1 })), new[] { 1, channels, length });
    }

    // Conv1d with "same" reflect padding (the dilated kernel's reach split evenly).
    private Tensor<T> Convolve(NormedConv1DLayer<T> conv, Tensor<T> x, int kernel, int dilation)
    {
        int pad = dilation * (kernel - 1) / 2;
        return conv.Forward(pad > 0 ? VocoderOps.ReflectPad(_engine, x, pad, pad) : x);
    }

    private Tensor<T> Run(Tdnn block, Tensor<T> x) =>
        Normalize(block.Norm, _engine.ReLU(Convolve(block.Conv, x, block.Kernel, block.Dilation)));

    private Tensor<T> Run(SeRes2Block block, Tensor<T> x)
    {
        var residual = block.Shortcut is null ? x : block.Shortcut.Forward(x);
        x = Run(block.Reduce, x);
        int scale = block.Branches.Count + 1, width = x.Shape[1], chunk = width / scale, length = x.Shape[2];
        var outputs = new Tensor<T>[scale];
        for (int i = 0; i < scale; i++)
        {
            var part = _engine.TensorSlice(x, new[] { 0, i * chunk, 0 }, new[] { 1, chunk, length });
            outputs[i] = i == 0 ? part : Run(block.Branches[i - 1], i == 1 ? part : _engine.TensorAdd(part, outputs[i - 1]));
        }
        x = Run(block.Expand, _engine.TensorConcatenate(outputs, 1));
        var s = _engine.ReduceMean(x, new[] { 2 }, true);
        s = _engine.Sigmoid(block.Excite.Forward(_engine.ReLU(block.Squeeze.Forward(s))));
        x = _engine.TensorMultiply(x, _engine.TensorBroadcastTo(s, x._shape));
        return _engine.TensorAdd(x, residual);
    }

    // _compute_statistics: the weighted mean and sqrt(clamp(Σ w (x − mean)², 1e-12)) over time.
    private (Tensor<T> Mean, Tensor<T> Std) Statistics(Tensor<T> x, Tensor<T> weights)
    {
        var mean = _engine.ReduceSum(_engine.TensorMultiply(weights, x), new[] { 2 }, true);
        var centered = _engine.TensorSubtract(x, _engine.TensorBroadcastTo(mean, x._shape));
        var variance = _engine.ReduceSum(_engine.TensorMultiply(weights, _engine.TensorMultiply(centered, centered)), new[] { 2 }, true);
        var floor = new Tensor<T>(variance._shape);
        for (int i = 0; i < floor.Length; i++) floor[i] = NumOps.FromDouble(Math.Max(1e-12 - NumOps.ToDouble(variance[i]), 0));
        return (mean, _engine.TensorSqrt(_engine.TensorAdd(variance, floor)));
    }

    /// <summary>The embedding <c>[EmbeddingSize]</c> of a 16 kHz waveform (<c>encode_batch</c> with every frame valid).</summary>
    public Tensor<T> Embed(IReadOnlyList<double> wave)
    {
        var features = Features(wave);
        int frames = features.GetLength(0);
        var x = new Tensor<T>(new[] { 1, _c.Mels, frames });
        for (int f = 0; f < frames; f++)
            for (int m = 0; m < _c.Mels; m++) x[0, m, f] = NumOps.FromDouble(features[f, m]);
        x = Run(First, x);
        var outputs = new List<Tensor<T>>();
        foreach (var block in Blocks)
        {
            x = Run(block, x);
            outputs.Add(x);
        }
        x = Run(Mfa, _engine.TensorConcatenate(outputs.ToArray(), 1));
        int channels = x.Shape[1];
        var uniform = new Tensor<T>(new[] { 1, 1, frames });
        for (int f = 0; f < frames; f++) uniform[0, 0, f] = NumOps.FromDouble(1.0 / frames);
        var (globalMean, globalStd) = Statistics(x, _engine.TensorBroadcastTo(uniform, x._shape));
        var context = _engine.TensorConcatenate(new[] { x, _engine.TensorBroadcastTo(globalMean, x._shape), _engine.TensorBroadcastTo(globalStd, x._shape) }, 1);
        var scores = AttentionScore.Forward(_engine.Tanh(Run(AttentionTdnn, context)));
        var (mean, std) = Statistics(x, _engine.Softmax(scores, axis: 2));
        var pooled = Normalize(PoolingNorm, _engine.TensorConcatenate(new[] { mean, std }, 1));
        return _engine.Reshape(Embedding.Forward(pooled), new[] { _c.EmbeddingSize });
    }

    /// <summary>Loads <c>embedding_model.ckpt</c> (speechbrain's ECAPA_TDNN state dict).</summary>
    public void Load(Func<string, int[], double[]> read)
    {
        void LoadConv(NormedConv1DLayer<T> conv, string name, int inputs, int outputs, int kernel) =>
            conv.LoadTorchWeights(read(name + ".weight", new[] { outputs, inputs, kernel }), null, read(name + ".bias", new[] { outputs }));
        void LoadTdnn(Tdnn block, string name)
        {
            LoadConv(block.Conv, name + ".conv.conv", block.Inputs, block.Outputs, block.Kernel);
            LoadNorm(block.Norm, name + ".norm.norm", block.Outputs);
        }
        void LoadNorm(BatchNormalizationLayer<T> norm, string n, int channels)
        {
            TorchParameters.BatchNorm(norm, read(n + ".weight", new[] { channels }), read(n + ".bias", new[] { channels }),
                read(n + ".running_mean", new[] { channels }), read(n + ".running_var", new[] { channels }));
            norm.SetTrainingMode(false);
        }
        LoadTdnn(First, "blocks.0");
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"blocks.{i + 1}";
            int width = b.Reduce.Outputs;
            LoadTdnn(b.Reduce, p + ".tdnn1");
            for (int j = 0; j < b.Branches.Count; j++) LoadTdnn(b.Branches[j], $"{p}.res2net_block.blocks.{j}");
            LoadTdnn(b.Expand, p + ".tdnn2");
            LoadConv(b.Squeeze, p + ".se_block.conv1.conv", width, _c.SeChannels, 1);
            LoadConv(b.Excite, p + ".se_block.conv2.conv", _c.SeChannels, width, 1);
            if (b.Shortcut is not null) LoadConv(b.Shortcut, p + ".shortcut.conv", b.Reduce.Inputs, width, 1);
        }
        LoadTdnn(Mfa, "mfa");
        LoadTdnn(AttentionTdnn, "asp.tdnn");
        int mfa = Mfa.Outputs;
        LoadConv(AttentionScore, "asp.conv.conv", _c.AttentionChannels, mfa, 1);
        LoadNorm(PoolingNorm, "asp_bn.norm", 2 * mfa);
        LoadConv(Embedding, "fc.conv", 2 * mfa, _c.EmbeddingSize, 1);
    }
}
