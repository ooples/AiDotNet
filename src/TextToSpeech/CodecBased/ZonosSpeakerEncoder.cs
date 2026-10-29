using AiDotNet.Audio.Features;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.StyleEmotion;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Zonos's speaker model (Zyphra/Zonos <c>zonos/speaker_cloning.py</c>, weights <c>Zyphra/Zonos-v0.1-speaker-embedding</c>):
/// the waveform resampled to 16 kHz (torchaudio's Resample), an 80-bin log mel spectrogram (torchaudio's MelSpectrogram:
/// FFT 512, window 400, hop 160, power, HTK mels; log(x + 1e-6), the mean over time removed), a ResNet-293 of SimAM basic
/// blocks (10, 20, 64, 3 blocks of 64, 128, 256, 512 channels), attentive statistics pooling, a 256-dimensional bottleneck,
/// and an LDA to 128 dimensions.
/// </summary>
/// <remarks>Frozen: the batch norms read their running statistics. Its layers are added to the owner's list.</remarks>
internal sealed class ZonosSpeakerEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly CenteredComplexStft<T> _stft;
    private readonly Tensor<T> _mel;            // [bins, mels]
    private readonly Dictionary<int, TorchaudioResample<T>> _resamplers = new();

    internal sealed record Block(NormedConv2DLayer<T> Conv1, BatchNormalizationLayer<T> Norm1, NormedConv2DLayer<T> Conv2,
        BatchNormalizationLayer<T> Norm2, NormedConv2DLayer<T>? Shortcut, BatchNormalizationLayer<T>? ShortcutNorm);

    internal sealed record Norm1D(BiasFreeLinearLayer<T> Weight, BiasFreeLinearLayer<T> Bias, BiasFreeLinearLayer<T> Mean, BiasFreeLinearLayer<T> Variance);

    public ZonosSpeakerEncoder(IEngine engine, List<LayerBase<T>> layers, int[]? blocks = null, int channels = 64, int melBins = 80,
        int embedding = 256, int ldaDim = 128)
    {
        _engine = engine;
        blocks ??= new[] { 10, 20, 64, 3 };
        if (blocks.Length != 4 || melBins % 8 != 0) throw new ArgumentException("Four stages over a mel axis that halves three times.");
        Blocks = blocks;
        Channels = channels;
        MelBins = melBins;
        _stft = new CenteredComplexStft<T>(engine, 512, 160, 400);
        var bank = MelFilterbanks.Htk(16000, 512, melBins, 0, 8000);                                            // [mels, bins]
        _mel = new Tensor<T>(new[] { bank.GetLength(1), melBins });
        for (int m = 0; m < melBins; m++)
            for (int k = 0; k < bank.GetLength(1); k++) _mel[k, m] = NumOps.FromDouble(bank[m, k]);
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        NormedConv2DLayer<T> Conv(int input, int output, int kernel, int stride) =>
            Own(new NormedConv2DLayer<T>(input, output, kernel, kernel, stride, stride, kernel / 2, kernel / 2, 1, ConvolutionNormalization.None, useBias: false));
        BatchNormalizationLayer<T> Norm(int c) => Own(new BatchNormalizationLayer<T>(c, 1e-5, 0.9));
        BiasFreeLinearLayer<T> Values(int count) => Own(new BiasFreeLinearLayer<T>(1, count));
        StemConv = Conv(1, channels, 3, 1);
        StemNorm = Norm(channels);
        int input = channels;
        for (int stage = 0; stage < 4; stage++)
        {
            int planes = channels << stage;
            var list = new List<Block>();
            for (int b = 0; b < blocks[stage]; b++)
            {
                int stride = b == 0 && stage > 0 ? 2 : 1;
                bool project = stride != 1 || input != planes;
                list.Add(new Block(Conv(input, planes, 3, stride), Norm(planes), Conv(planes, planes, 3, 1), Norm(planes),
                    project ? Conv(input, planes, 1, stride) : null, project ? Norm(planes) : null));
                input = planes;
            }
            Stages.Add(list);
        }
        PooledChannels = channels * 8 * (melBins / 8);
        AttentionIn = Own(new BiasFreeLinearLayer<T>(PooledChannels, 128));
        AttentionInBias = Values(128);
        AttentionNorm = new Norm1D(Values(128), Values(128), Values(128), Values(128));
        AttentionOut = Own(new BiasFreeLinearLayer<T>(128, PooledChannels));
        AttentionOutBias = Values(PooledChannels);
        Bottleneck = Own(new BiasFreeLinearLayer<T>(2 * PooledChannels, embedding));
        BottleneckBias = Values(embedding);
        Lda = Own(new BiasFreeLinearLayer<T>(embedding, ldaDim));
        LdaBias = Values(ldaDim);
        EmbeddingDim = embedding;
        LdaDim = ldaDim;
    }

    public int[] Blocks { get; }
    public int Channels { get; }
    public int MelBins { get; }
    public int PooledChannels { get; }
    public int EmbeddingDim { get; }
    public int LdaDim { get; }
    public NormedConv2DLayer<T> StemConv { get; }
    public BatchNormalizationLayer<T> StemNorm { get; }
    public List<List<Block>> Stages { get; } = new();
    public BiasFreeLinearLayer<T> AttentionIn { get; }
    public BiasFreeLinearLayer<T> AttentionInBias { get; }
    public Norm1D AttentionNorm { get; }
    public BiasFreeLinearLayer<T> AttentionOut { get; }
    public BiasFreeLinearLayer<T> AttentionOutBias { get; }
    public BiasFreeLinearLayer<T> Bottleneck { get; }
    public BiasFreeLinearLayer<T> BottleneckBias { get; }
    public BiasFreeLinearLayer<T> Lda { get; }
    public BiasFreeLinearLayer<T> LdaBias { get; }

    private static Tensor<T> Value(BiasFreeLinearLayer<T> holder) => holder.Weights;

    private Tensor<T> Linear(BiasFreeLinearLayer<T> weight, BiasFreeLinearLayer<T> bias, Tensor<T> rows)
    {
        var y = weight.Forward(rows);
        return _engine.TensorAdd(y, _engine.TensorBroadcastTo(_engine.Reshape(Value(bias), new[] { 1, y.Shape[1] }), y._shape));
    }

    /// <summary>The 16 kHz waveform of <paramref name="audio"/> at <paramref name="sampleRate"/>, mono (channels
    /// <c>[channels, samples]</c> averaged).</summary>
    public Tensor<T> Prepare(Tensor<T> audio, int sampleRate)
    {
        if (audio.Rank == 2) audio = _engine.ReduceMean(audio, new[] { 0 }, keepDims: false);
        if (sampleRate == 16000) return _engine.Reshape(audio, new[] { audio.Length });
        if (!_resamplers.TryGetValue(sampleRate, out var resampler)) _resamplers[sampleRate] = resampler = new TorchaudioResample<T>(_engine, sampleRate, 16000);
        return resampler.Forward(audio);
    }

    /// <summary>logFbankCal: the mean-normalized log mel spectrogram <c>[mels, frames]</c> of a 16 kHz waveform.</summary>
    public Tensor<T> Features(Tensor<T> audio)
    {
        var (re, im) = _stft.Forward(audio);                                                                     // [1, bins, frames]
        int bins = re.Shape[1], frames = re.Shape[2];
        var power = _engine.TensorAdd(_engine.TensorMultiply(re, re), _engine.TensorMultiply(im, im));
        var mel = _engine.TensorTranspose(_engine.TensorMatMul(_engine.TensorTranspose(_engine.Reshape(power, new[] { bins, frames })), _mel)); // [mels, frames]
        var log = _engine.TensorLog(_engine.TensorAddScalar(mel, NumOps.FromDouble(1e-6)));
        return _engine.TensorSubtract(log, _engine.TensorBroadcastTo(_engine.ReduceMean(log, new[] { 1 }, keepDims: true), log._shape));
    }

    // SimAM: X · σ(d / (4 (v + λ)) + ½), d the squared deviation from the map's mean, v their sum over (n − 1).
    private Tensor<T> SimAm(Tensor<T> x, double lambda = 1e-4)
    {
        var shape = x._shape;
        int n = shape[2] * shape[3] - 1;
        var mean = _engine.ReduceMean(x, new[] { 2, 3 }, keepDims: true);
        var centered = _engine.TensorSubtract(x, _engine.TensorBroadcastTo(mean, shape));
        var d = _engine.TensorMultiply(centered, centered);
        var v = _engine.TensorMultiplyScalar(_engine.ReduceSum(d, new[] { 2, 3 }, keepDims: true), NumOps.FromDouble(1.0 / n));
        var denominator = _engine.TensorMultiplyScalar(_engine.TensorAddScalar(v, NumOps.FromDouble(lambda)), NumOps.FromDouble(4));
        var energy = _engine.TensorAddScalar(_engine.TensorDivide(d, _engine.TensorBroadcastTo(denominator, shape)), NumOps.FromDouble(0.5));
        return _engine.TensorMultiply(x, _engine.Sigmoid(energy));
    }

    /// <summary>The 256-dimensional embedding and its 128-dimensional LDA projection of a 16 kHz waveform.</summary>
    public (double[] Embedding, double[] Lda) Forward(Tensor<T> audio)
    {
        var features = Features(audio);
        int frames = features.Shape[1];
        var x = _engine.Reshape(features, new[] { 1, 1, MelBins, frames });
        x = _engine.ReLU(StemNorm.Forward(StemConv.Forward(x)));
        foreach (var stage in Stages)
            foreach (var b in stage)
            {
                var y = _engine.ReLU(b.Norm1.Forward(b.Conv1.Forward(x)));
                y = SimAm(b.Norm2.Forward(b.Conv2.Forward(y)));
                var shortcut = b.Shortcut is null ? x : b.ShortcutNorm!.Forward(b.Shortcut.Forward(x));
                x = _engine.ReLU(_engine.TensorAdd(y, shortcut));
            }
        // ASP: x [1, C·F, T]; attention weights over time from conv1d → ReLU → BatchNorm1d → conv1d → softmax.
        int t = x.Shape[3], c = PooledChannels;
        var rows = _engine.TensorTranspose(_engine.Reshape(x, new[] { c, t }));                                   // [T, C·F]
        var hidden = _engine.ReLU(Linear(AttentionIn, AttentionInBias, rows));                                     // [T, 128]
        var n = AttentionNorm;
        var scale = _engine.TensorMultiply(Value(n.Weight), _engine.TensorPower(_engine.TensorAddScalar(Value(n.Variance), NumOps.FromDouble(1e-5)), NumOps.FromDouble(-0.5)));
        var shift = _engine.TensorSubtract(Value(n.Bias), _engine.TensorMultiply(Value(n.Mean), scale));
        hidden = _engine.TensorAdd(_engine.TensorMultiply(hidden, _engine.TensorBroadcastTo(_engine.Reshape(scale, new[] { 1, 128 }), hidden._shape)),
            _engine.TensorBroadcastTo(_engine.Reshape(shift, new[] { 1, 128 }), hidden._shape));
        var weights = _engine.Softmax(Linear(AttentionOut, AttentionOutBias, hidden), axis: 0);                    // [T, C·F]
        var mu = _engine.ReduceSum(_engine.TensorMultiply(rows, weights), new[] { 0 }, keepDims: true);
        var second = _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorMultiply(rows, rows), weights), new[] { 0 }, keepDims: true);
        var sg = _engine.TensorSqrt(_engine.TensorMax(_engine.TensorSubtract(second, _engine.TensorMultiply(mu, mu)), NumOps.FromDouble(1e-5)));
        var pooled = _engine.TensorConcatenate(new[] { mu, sg }, 1);                                               // [1, 2·C·F]
        var embedding = Linear(Bottleneck, BottleneckBias, pooled);
        var lda = Linear(Lda, LdaBias, embedding);
        return (embedding.ToArray().Select(v => NumOps.ToDouble(v)).ToArray(), lda.ToArray().Select(v => NumOps.ToDouble(v)).ToArray());
    }

    /// <summary>Loads <c>ResNet293_SimAM_ASP_base.pt</c> (names without a prefix) and the LDA (<c>lda.weight</c>,
    /// <c>lda.bias</c> as passed by the caller); returns the names read.</summary>
    public HashSet<string> LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var used = new HashSet<string>(StringComparer.Ordinal);
        double[] Read(string name, int[] shape)
        {
            used.Add(name);
            return read(name, shape);
        }
        void Conv(NormedConv2DLayer<T> conv, string name) => conv.LoadTorchWeights(Read(name + ".weight", conv.TorchWeightShape), null);
        void Norm(BatchNormalizationLayer<T> norm, string name, int width) => TorchParameters.BatchNorm(norm, Read(name + ".weight", new[] { width }),
            Read(name + ".bias", new[] { width }), Read(name + ".running_mean", new[] { width }), Read(name + ".running_var", new[] { width }));
        void Vector(BiasFreeLinearLayer<T> holder, string name, int count) => TorchParameters.Linear(holder, 1, count, Read(name, new[] { count }));
        Conv(StemConv, "front.conv1");
        Norm(StemNorm, "front.bn1", Channels);
        for (int s = 0; s < Stages.Count; s++)
            for (int b = 0; b < Stages[s].Count; b++)
            {
                var block = Stages[s][b];
                string p = $"front.layer{s + 1}.{b}";
                int planes = block.Conv1.TorchWeightShape[0];
                Conv(block.Conv1, p + ".conv1");
                Norm(block.Norm1, p + ".bn1", planes);
                Conv(block.Conv2, p + ".conv2");
                Norm(block.Norm2, p + ".bn2", planes);
                if (block.Shortcut is not null)
                {
                    Conv(block.Shortcut, p + ".downsample.0");
                    Norm(block.ShortcutNorm!, p + ".downsample.1", planes);
                }
            }
        int c = PooledChannels;
        TorchParameters.Linear(AttentionIn, c, 128, Read("pooling.attention.0.weight", new[] { 128, c, 1 }));
        Vector(AttentionInBias, "pooling.attention.0.bias", 128);
        Vector(AttentionNorm.Weight, "pooling.attention.2.weight", 128);
        Vector(AttentionNorm.Bias, "pooling.attention.2.bias", 128);
        Vector(AttentionNorm.Mean, "pooling.attention.2.running_mean", 128);
        Vector(AttentionNorm.Variance, "pooling.attention.2.running_var", 128);
        TorchParameters.Linear(AttentionOut, 128, c, Read("pooling.attention.3.weight", new[] { c, 128, 1 }));
        Vector(AttentionOutBias, "pooling.attention.3.bias", c);
        TorchParameters.Linear(Bottleneck, 2 * c, EmbeddingDim, Read("bottleneck.weight", new[] { EmbeddingDim, 2 * c }));
        Vector(BottleneckBias, "bottleneck.bias", EmbeddingDim);
        TorchParameters.Linear(Lda, EmbeddingDim, LdaDim, Read("lda.weight", new[] { LdaDim, EmbeddingDim }));
        Vector(LdaBias, "lda.bias", LdaDim);
        return used;
    }
}
