using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// Parallel WaveGAN's generator (Yamamoto et al. 2020, §3.1, §4.1.2): a non-causal WaveNet that turns Gaussian noise
/// into a waveform, conditioned on the mel spectrogram upsampled to the sample rate.
/// </summary>
/// <remarks>
/// <para>30 dilated residual blocks in three cycles (dilations 1, 2, …, 512), 64 residual and skip channels, kernel 3
/// and weight normalization throughout (§4.1.2). The rest follows kan-bayashi/ParallelWaveGAN
/// (<c>ParallelWaveGANGenerator</c>, <c>parallel_wavegan.v1</c>): a gate of 128 channels split into tanh and sigmoid
/// halves with the local condition added by a 1×1 convolution; residual outputs scaled by √½ and skip connections summed
/// and scaled by √(1/30); ReLU, 1×1, ReLU, 1×1 to the waveform. The condition is replicate-padded by two frames, passed
/// through a 5-wide convolution (<c>ConvInUpsampleNetwork</c>, aux context window 2) and upsampled stage by stage by
/// nearest-neighbour stretching and a <c>(2s + 1)</c>-wide smoothing convolution initialized to <c>1/(2s + 1)</c>.
/// Convolutions are Kaiming-normal initialized with zero biases, as the reference's <c>Conv1d</c> is.</para>
/// </remarks>
internal sealed class ParallelWaveGanGenerator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly int _auxContext;
    private readonly int _gate;
    private readonly NormedConv1DLayer<T> _convIn;
    private readonly List<(int Scale, NormedConv1DLayer<T> Smooth)> _upsample = new();
    private readonly NormedConv1DLayer<T> _first;
    private readonly List<(NormedConv1DLayer<T> Dilated, NormedConv1DLayer<T> Aux, NormedConv1DLayer<T> Skip, NormedConv1DLayer<T> Out)> _blocks = new();
    private readonly NormedConv1DLayer<T> _last1;
    private readonly NormedConv1DLayer<T> _last2;

    public ParallelWaveGanGenerator(IEngine engine, Random random, int melChannels, int[] upsampleScales, int auxContextWindow,
        int layers, int stacks, int residualChannels, int gateChannels, int skipChannels, int kernelSize)
    {
        if (layers % stacks != 0) throw new ArgumentException($"The layers ({layers}) must split evenly into {stacks} stacks.");
        _engine = engine;
        _auxContext = auxContextWindow;
        _gate = gateChannels;
        _convIn = Kaiming(new NormedConv1DLayer<T>(melChannels, melChannels, 2 * auxContextWindow + 1, 1, 1, 1, 0, false, ConvolutionNormalization.Weight, useBias: false), random);
        foreach (int s in upsampleScales)
        {
            var smooth = Add(new NormedConv1DLayer<T>(1, 1, 2 * s + 1, 1, 1, 1, s, false, ConvolutionNormalization.Weight, useBias: false));
            smooth.Reinitialize(() => 1.0 / (2 * s + 1));
            _upsample.Add((s, smooth));
        }
        _first = Kaiming(new NormedConv1DLayer<T>(1, residualChannels, 1, 1, 1, 1, 0, false, ConvolutionNormalization.Weight), random);
        int perStack = layers / stacks;
        for (int l = 0; l < layers; l++)
        {
            int d = 1 << (l % perStack);
            _blocks.Add((
                Kaiming(new NormedConv1DLayer<T>(residualChannels, gateChannels, kernelSize, 1, d, 1, (kernelSize - 1) / 2 * d, false, ConvolutionNormalization.Weight), random),
                Kaiming(new NormedConv1DLayer<T>(melChannels, gateChannels, 1, 1, 1, 1, 0, false, ConvolutionNormalization.Weight, useBias: false), random),
                Kaiming(new NormedConv1DLayer<T>(gateChannels / 2, skipChannels, 1, 1, 1, 1, 0, false, ConvolutionNormalization.Weight), random),
                Kaiming(new NormedConv1DLayer<T>(gateChannels / 2, residualChannels, 1, 1, 1, 1, 0, false, ConvolutionNormalization.Weight), random)));
        }
        _last1 = Kaiming(new NormedConv1DLayer<T>(skipChannels, skipChannels, 1, 1, 1, 1, 0, false, ConvolutionNormalization.Weight), random);
        _last2 = Kaiming(new NormedConv1DLayer<T>(skipChannels, 1, 1, 1, 1, 1, 0, false, ConvolutionNormalization.Weight), random);
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    private NormedConv1DLayer<T> Add(NormedConv1DLayer<T> layer)
    {
        _layers.Add(layer);
        return layer;
    }

    // torch.nn.init.kaiming_normal_(w, nonlinearity="relu"): N(0, 2 / fan_in); bias 0.
    private NormedConv1DLayer<T> Kaiming(NormedConv1DLayer<T> layer, Random random)
    {
        Add(layer);
        double std = Math.Sqrt(2.0 / layer.FanIn);
        layer.Reinitialize(() =>
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            return std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        });
        return layer;
    }

    /// <summary>The local condition <c>[1, mel, frames · Π s]</c> for a mel spectrogram <c>[1, mel, frames]</c>.</summary>
    private Tensor<T> Upsample(Tensor<T> mel)
    {
        int c = mel.Shape[1];
        var x = _convIn.Forward(ReplicatePad(mel, _auxContext));
        foreach (var (s, smooth) in _upsample)
        {
            int t = x.Shape[2];
            var index = new Tensor<int>(new[] { t * s });
            for (int i = 0; i < index.Length; i++) index[i] = i / s;
            var stretched = _engine.TensorIndexSelect(_engine.TensorTranspose(_engine.Reshape(x, new[] { c, t })), index, 0);   // [t·s, c]
            // Each mel band is smoothed by the same single-channel kernel: bands become the batch.
            var bands = _engine.Reshape(_engine.TensorTranspose(stretched), new[] { c, 1, t * s });
            x = _engine.Reshape(smooth.Forward(bands), new[] { 1, c, t * s });
        }
        return x;
    }

    private Tensor<T> ReplicatePad(Tensor<T> x, int pad)
    {
        if (pad == 0) return x;
        int c = x.Shape[1], t = x.Shape[2];
        var index = new Tensor<int>(new[] { t + 2 * pad });
        for (int i = 0; i < index.Length; i++) index[i] = Math.Min(Math.Max(i - pad, 0), t - 1);
        var rows = _engine.TensorIndexSelect(_engine.TensorTranspose(_engine.Reshape(x, new[] { c, t })), index, 0);
        return _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, c, t + 2 * pad });
    }

    /// <summary>A waveform <c>[1, 1, samples]</c> from Gaussian noise <c>[1, 1, samples]</c> and a mel spectrogram
    /// <c>[1, mel, samples / hop]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> noise, Tensor<T> mel)
    {
        var c = Upsample(mel);
        var x = _first.Forward(noise);
        Tensor<T>? skips = null;
        int half = _gate / 2;
        foreach (var (dilated, aux, skip, output) in _blocks)
        {
            var residual = x;
            var h = _engine.TensorAdd(dilated.Forward(x), aux.Forward(c));
            int t = h.Shape[2];
            var a = _engine.TensorSlice(h, new[] { 0, 0, 0 }, new[] { 1, half, t });
            var b = _engine.TensorSlice(h, new[] { 0, half, 0 }, new[] { 1, half, t });
            var z = _engine.TensorMultiply(_engine.Tanh(a), _engine.Sigmoid(b));
            var s = skip.Forward(z);
            skips = skips is null ? s : _engine.TensorAdd(skips, s);
            x = _engine.TensorMultiplyScalar(_engine.TensorAdd(output.Forward(z), residual), NumOps.FromDouble(Math.Sqrt(0.5)));
        }
        var y = _engine.TensorMultiplyScalar(skips!, NumOps.FromDouble(Math.Sqrt(1.0 / _blocks.Count)));
        y = _last1.Forward(_engine.ReLU(y));
        return _last2.Forward(_engine.ReLU(y));
    }
}

/// <summary>
/// Parallel WaveGAN's discriminator (§4.1.2): ten non-causal 1-D convolutions of 64 channels and kernel 3, stride 1, with
/// dilations 1, 1, 2, …, 8 (linear, the first and last layers undilated) and leaky ReLU (0.2) after all but the last,
/// which gives one score per sample; weight-normalized and Kaiming-normal initialized (reference
/// <c>ParallelWaveGANDiscriminator</c>).
/// </summary>
internal sealed class ParallelWaveGanDiscriminator<T>
{
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly List<NormedConv1DLayer<T>> _convs = new();

    public ParallelWaveGanDiscriminator(IEngine engine, Random random, int layers, int channels, int kernelSize)
    {
        _engine = engine;
        for (int i = 0; i < layers - 1; i++)
        {
            int d = i == 0 ? 1 : i, input = i == 0 ? 1 : channels;
            _convs.Add(Kaiming(new NormedConv1DLayer<T>(input, channels, kernelSize, 1, d, 1, (kernelSize - 1) / 2 * d, false, ConvolutionNormalization.Weight), random));
        }
        _convs.Add(Kaiming(new NormedConv1DLayer<T>(channels, 1, kernelSize, 1, 1, 1, (kernelSize - 1) / 2, false, ConvolutionNormalization.Weight), random));
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    private NormedConv1DLayer<T> Kaiming(NormedConv1DLayer<T> layer, Random random)
    {
        _layers.Add(layer);
        double std = Math.Sqrt(2.0 / layer.FanIn);
        layer.Reinitialize(() =>
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            return std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        });
        return layer;
    }

    /// <summary>Per-sample scores <c>[1, 1, samples]</c> for a waveform <c>[samples]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> audio)
    {
        var x = _engine.Reshape(audio, new[] { 1, 1, audio.Length });
        for (int i = 0; i < _convs.Count - 1; i++) x = VocoderOps.LeakyRelu(_engine, _convs[i].Forward(x), 0.2);
        return _convs[^1].Forward(x);
    }
}
