using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.Audio.Speaker;

/// <summary>The sizes of speechbrain's x-vector extractor (<c>speechbrain/spkrec-xvect-voxceleb</c>).</summary>
internal sealed record SpeechBrainXVectorConfiguration
{
    public int SampleRate { get; init; } = 16000;
    public int Mels { get; init; } = 24;
    public int[] Channels { get; init; } = { 512, 512, 512, 512, 1500 };
    public int[] Kernels { get; init; } = { 5, 3, 3, 1, 1 };
    public int[] Dilations { get; init; } = { 1, 2, 3, 1, 1 };
    public int EmbeddingSize { get; init; } = 512;
}

/// <summary>
/// speechbrain's x-vector speaker embedding (Snyder et al. 2018) as <c>spkrec-xvect-voxceleb</c> computes it, the
/// embedding SpeechT5 conditions on: speechbrain's <c>Fbank</c> (a 25 ms Hamming-windowed STFT every 10 ms, centred with
/// zero padding, the power spectrum through 24 triangular HTK-mel filters, 10·log10 floored at 80 dB below the maximum),
/// the sentence mean removed, then five TDNN layers (dilated convolutions with reflect padding, leaky ReLU, BatchNorm),
/// statistics pooling (the mean plus speechbrain's small positive noise, and the unbiased standard deviation plus 1e-5)
/// and a linear embedding.
/// </summary>
internal sealed class SpeechBrainXVector<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly SpeechBrainXVectorConfiguration _c;
    private readonly double[,] _filters;

    public SpeechBrainXVector(IEngine engine, List<LayerBase<T>> layers, SpeechBrainXVectorConfiguration configuration, Random random)
    {
        _engine = engine;
        _c = configuration;
        int inputs = _c.Mels;
        for (int i = 0; i < _c.Channels.Length; i++)
        {
            int outputs = _c.Channels[i], k = _c.Kernels[i];
            var conv = new NormedConv1DLayer<T>(inputs, outputs, k, 1, _c.Dilations[i], 1, 0, false, ConvolutionNormalization.None);
            layers.Add(conv);
            conv.LoadTorchWeights(IndexTtsOps.Uniform(random, outputs * inputs * k, inputs * k), null, IndexTtsOps.Uniform(random, outputs, inputs * k));
            var norm = new BatchNormalizationLayer<T>(outputs, epsilon: 1e-5, momentum: 0.9);
            layers.Add(norm);
            Blocks.Add((conv, norm));
            inputs = outputs;
        }
        Embedding = IndexTtsOps.Linear(engine, layers, random, 2 * inputs, _c.EmbeddingSize);
        _filters = SpeechBrainFbank.Filters(_c.Mels, _c.SampleRate);
    }

    public List<(NormedConv1DLayer<T> Conv, BatchNormalizationLayer<T> Norm)> Blocks { get; } = new();
    public DenseLayer<T> Embedding { get; }

    /// <summary>speechbrain's <c>Fbank</c> of a 16 kHz waveform, the sentence mean removed: <c>[frames, mels]</c>.</summary>
    public double[,] Features(IReadOnlyList<double> wave) => SpeechBrainFbank.Features(wave, _filters);

    /// <summary>
    /// The x-vector <c>[EmbeddingSize]</c> of a 16 kHz waveform (<c>encode_batch</c> with all frames valid).
    /// <paramref name="noise"/> draws speechbrain's pooling noise (<c>ε((1 − 9)·g + 9)</c> with g a min-max normalized
    /// Gaussian, ε = 1e-5); <paramref name="noiseSample"/> gives the normalized Gaussian draws instead.
    /// </summary>
    public Tensor<T> Embed(IReadOnlyList<double> wave, Random? noise, double[]? noiseSample = null)
    {
        var features = Features(wave);
        int frames = features.GetLength(0);
        var x = new Tensor<T>(new[] { 1, _c.Mels, frames });
        for (int f = 0; f < frames; f++)
            for (int m = 0; m < _c.Mels; m++) x[0, m, f] = NumOps.FromDouble(features[f, m]);
        for (int i = 0; i < Blocks.Count; i++)
        {
            var (conv, norm) = Blocks[i];
            int pad = _c.Dilations[i] * (_c.Kernels[i] - 1) / 2;
            var y = conv.Forward(VocoderOps.ReflectPad(_engine, x, pad, pad));
            y = _engine.LeakyReLU(y, NumOps.FromDouble(0.01));
            int channels = y.Shape[1], length = y.Shape[2];
            x = _engine.Reshape(norm.Forward(_engine.Reshape(y, new[] { 1, channels, length, 1 })), new[] { 1, channels, length });
        }
        int c = x.Shape[1], t = x.Shape[2];
        var rows = _engine.Reshape(x, new[] { c, t });
        var mean = _engine.ReduceMean(rows, new[] { 1 }, false);
        var centered = _engine.TensorSubtract(rows, _engine.TensorBroadcastTo(_engine.Reshape(mean, new[] { c, 1 }), new[] { c, t }));
        var std = _engine.TensorSqrt(_engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(centered, centered), new[] { 1 }, false),
            NumOps.FromDouble(1.0 / (t - 1))));
        var g = noiseSample ?? (noise is null ? null : Enumerable.Range(0, c).Select(_ => IndexTtsOps.Gaussian(noise)).ToArray());
        if (g is not null)
        {
            double lo = g.Min();
            var shifted = g.Select(v => v - lo).ToArray();
            double hi = shifted.Max();
            var offset = new Tensor<T>(new[] { c });
            for (int i = 0; i < c; i++) offset[i] = NumOps.FromDouble(1e-5 * ((1 - 9) * (shifted[i] / hi) + 9));
            mean = _engine.TensorAdd(mean, offset);
        }
        std = _engine.TensorAddScalar(std, NumOps.FromDouble(1e-5));
        var pooled = _engine.Reshape(_engine.TensorConcatenate(new[] { mean, std }, 0), new[] { 1, 2 * c });
        return _engine.Reshape(Embedding.Forward(pooled), new[] { _c.EmbeddingSize });
    }

    /// <summary>Loads <c>embedding_model.ckpt</c> (<c>blocks.{3i}.conv</c>, <c>blocks.{3i+2}.norm</c>,
    /// <c>blocks.{3L+1}.w</c>).</summary>
    public void Load(Func<string, int[], double[]> read)
    {
        int inputs = _c.Mels;
        for (int i = 0; i < Blocks.Count; i++)
        {
            var (conv, norm) = Blocks[i];
            int outputs = _c.Channels[i];
            conv.LoadTorchWeights(read($"blocks.{3 * i}.conv.weight", new[] { outputs, inputs, _c.Kernels[i] }), null, read($"blocks.{3 * i}.conv.bias", new[] { outputs }));
            string n = $"blocks.{3 * i + 2}.norm.";
            TorchParameters.BatchNorm(norm, read(n + "weight", new[] { outputs }), read(n + "bias", new[] { outputs }),
                read(n + "running_mean", new[] { outputs }), read(n + "running_var", new[] { outputs }));
            norm.SetTrainingMode(false);
            inputs = outputs;
        }
        IndexTtsOps.LoadLinear(_engine, read, Embedding, $"blocks.{3 * Blocks.Count + 1}.w", 2 * inputs, _c.EmbeddingSize);
    }
}
