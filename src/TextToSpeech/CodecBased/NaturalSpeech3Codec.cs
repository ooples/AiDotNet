using AiDotNet.Audio.Codecs;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>FACodec's shape (Amphion <c>models/codec/ns3_codec</c>; the released checkpoints' configuration).</summary>
internal sealed record FACodecConfiguration
{
    public int SampleRate { get; init; } = 16000;
    public int EncoderDim { get; init; } = 32;
    public int[] EncoderRates { get; init; } = { 2, 4, 5, 5 };
    public int Dim { get; init; } = 256;
    public int DecoderInitial { get; init; } = 1024;
    public int[] DecoderRates { get; init; } = { 5, 5, 4, 2 };
    public int ProsodyQuantizers { get; init; } = 1;
    public int ContentQuantizers { get; init; } = 2;
    public int DetailQuantizers { get; init; } = 3;
    public int CodebookDim { get; init; } = 8;
    public int CodebookSize { get; init; } = 1024;
    public int TimbreLayers { get; init; } = 4;
    public int TimbreHeads { get; init; } = 4;
    public int TimbreFilter { get; init; } = 1024;
    public int TimbreKernel { get; init; } = 5;
    public double TimbreDropout { get; init; } = 0.1;
    public int Phones { get; init; } = 5003;
    public int Speakers { get; init; } = 245200;
    public bool DetailF0Reversal { get; init; } = true;
    public bool DetailPhoneReversal { get; init; } = true;
    public bool TimbreReversal { get; init; } = true;
    public bool ContentF0Reversal { get; init; }
    public bool ProsodyPhoneReversal { get; init; }
    public bool MelProsody { get; init; }
    public int ProsodyMels { get; init; } = 20;
    public int RedecoderInitial { get; init; } = 1280;
    public bool Redecoder { get; init; }

    public int Hop => EncoderRates.Aggregate(1, (a, b) => a * b);
    public int Codebooks => ProsodyQuantizers + ContentQuantizers + DetailQuantizers;
}

/// <summary>FACodec's anti-aliased SnakeBeta (<c>Activation1d(SnakeBeta(alpha_logscale=True))</c>), whose stored filters a
/// released checkpoint carries.</summary>
internal sealed class FACodecActivation<T>
{
    private readonly AntiAliasedSnake<T> _filtered;

    public FACodecActivation(IEngine engine, List<LayerBase<T>> owner, int channels)
    {
        Snake = new XCodec2SnakeBetaLayer<T>(channels);
        _filtered = new AntiAliasedSnake<T>(engine, owner, Snake);
    }

    public XCodec2SnakeBetaLayer<T> Snake { get; }

    public Tensor<T> Forward(Tensor<T> x) => _filtered.Forward(x);

    public void Load(Func<string, double[]> read, Func<string, double[]?> optional, string name)
    {
        Snake.Load(read(name + ".act.alpha"), read(name + ".act.beta"));
        if (optional(name + ".upsample.filter") is { } up && optional(name + ".downsample.lowpass.filter") is { } down) _filtered.LoadFilters(up, down);
    }
}

/// <summary>FACodec's (DAC's) residual unit: SnakeBeta → a dilated k = 7 convolution → SnakeBeta → a 1×1 convolution, added
/// to the input; all convolutions weight-normalized.</summary>
internal sealed class FACodecResidualUnit<T>
{
    private readonly IEngine _engine;

    public FACodecResidualUnit(IEngine engine, List<LayerBase<T>> owner, int dim, int dilation)
    {
        _engine = engine;
        Act1 = new FACodecActivation<T>(engine, owner, dim);
        Conv1 = XCodec2.WeightNormConv(owner, dim, dim, 7, 1, dilation, 3 * dilation);
        Act2 = new FACodecActivation<T>(engine, owner, dim);
        Conv2 = XCodec2.WeightNormConv(owner, dim, dim, 1, 1, 1, 0);
    }

    public FACodecActivation<T> Act1 { get; }
    public NormedConv1DLayer<T> Conv1 { get; }
    public FACodecActivation<T> Act2 { get; }
    public NormedConv1DLayer<T> Conv2 { get; }

    public Tensor<T> Forward(Tensor<T> x) => _engine.TensorAdd(x, Conv2.Forward(Act2.Forward(Conv1.Forward(Act1.Forward(x)))));

    public void Load(Func<string, double[]> read, Func<string, double[]?> optional, string prefix)
    {
        Act1.Load(read, optional, prefix + ".block.0");
        FACodecWeights.Conv(read, Conv1, prefix + ".block.1");
        Act2.Load(read, optional, prefix + ".block.2");
        FACodecWeights.Conv(read, Conv2, prefix + ".block.3");
    }
}

internal static class FACodecWeights
{
    public static void Conv<T>(Func<string, double[]> read, NormedConv1DLayer<T> conv, string name) =>
        conv.LoadTorchWeights(read(name + ".weight_v"), read(name + ".weight_g"), read(name + ".bias"));

    /// <summary>The gradient reversal layer (Ganin and Lempitsky 2015): the identity forward, the gradient times −α backward,
    /// as <c>(1 + α)·stop(x) − α·x</c>.</summary>
    public static Tensor<T> Reverse<T>(IEngine engine, Tensor<T> x, double alpha)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var stopped = new Tensor<T>(x._shape, x.ToVector());
        return engine.TensorSubtract(engine.TensorMultiplyScalar(stopped, ops.FromDouble(1 + alpha)), engine.TensorMultiplyScalar(x, ops.FromDouble(alpha)));
    }
}

/// <summary>
/// FACodec's speech encoder (<c>FACodecEncoder</c>, after DAC with anti-aliased SnakeBeta): a k = 7 convolution to ngf; per
/// stride s three residual units (dilations 1, 3, 9) at the current width, SnakeBeta and a strided convolution (kernel 2s)
/// doubling it; SnakeBeta and a k = 3 convolution to the latent width. A hop of 200 at 16 kHz.
/// </summary>
internal sealed class FACodecEncoder<T>
{
    private readonly NormedConv1DLayer<T> _first, _last;
    private readonly List<(FACodecResidualUnit<T>[] Units, FACodecActivation<T> Act, NormedConv1DLayer<T> Down)> _blocks = new();
    private readonly FACodecActivation<T> _act;

    public FACodecEncoder(IEngine engine, List<LayerBase<T>> owner, FACodecConfiguration c)
    {
        int d = c.EncoderDim;
        _first = XCodec2.WeightNormConv(owner, 1, d, 7, 1, 1, 3);
        foreach (int s in c.EncoderRates)
        {
            var units = new[] { 1, 3, 9 }.Select(dilation => new FACodecResidualUnit<T>(engine, owner, d, dilation)).ToArray();
            _blocks.Add((units, new FACodecActivation<T>(engine, owner, d), XCodec2.WeightNormConv(owner, d, 2 * d, 2 * s, s, 1, s / 2 + s % 2)));
            d *= 2;
        }
        _act = new FACodecActivation<T>(engine, owner, d);
        _last = XCodec2.WeightNormConv(owner, d, c.Dim, 3, 1, 1, 1);
    }

    /// <summary>The latent h <c>[1, dim, frames]</c> of a waveform <c>[1, 1, samples]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x)
    {
        x = _first.Forward(x);
        foreach (var (units, act, down) in _blocks)
        {
            foreach (var unit in units) x = unit.Forward(x);
            x = down.Forward(act.Forward(x));
        }
        return _last.Forward(_act.Forward(x));
    }

    public void Load(Func<string, double[]> read, Func<string, double[]?> optional, string prefix)
    {
        FACodecWeights.Conv(read, _first, prefix + "block.0");
        for (int i = 0; i < _blocks.Count; i++)
        {
            var (units, act, down) = _blocks[i];
            string p = $"{prefix}block.{i + 1}.block";
            for (int u = 0; u < units.Length; u++) units[u].Load(read, optional, $"{p}.{u}");
            act.Load(read, optional, $"{p}.{units.Length}");
            FACodecWeights.Conv(read, down, $"{p}.{units.Length + 1}");
        }
        _act.Load(read, optional, $"{prefix}block.{_blocks.Count + 1}");
        FACodecWeights.Conv(read, _last, $"{prefix}block.{_blocks.Count + 2}");
    }
}

/// <summary>
/// FACodec's waveform decoder (<c>model</c>): a k = 7 convolution to the initial width; per stride s SnakeBeta, a transposed
/// convolution (kernel 2s, padding ⌊s/2⌋ + s mod 2, output padding s mod 2) halving the width and three residual units
/// (dilations 1, 3, 9); SnakeBeta, a k = 7 convolution to one channel and tanh.
/// </summary>
internal sealed class FACodecDecoderStack<T>
{
    private readonly IEngine _engine;
    private readonly NormedConv1DLayer<T> _first, _last;
    private readonly List<(FACodecActivation<T> Act, NormedConv1DLayer<T> Up, FACodecResidualUnit<T>[] Units)> _blocks = new();
    private readonly FACodecActivation<T> _act;

    public FACodecDecoderStack(IEngine engine, List<LayerBase<T>> owner, int input, int initial, int[] rates)
    {
        _engine = engine;
        _first = XCodec2.WeightNormConv(owner, input, initial, 7, 1, 1, 3);
        int channels = initial;
        foreach (int s in rates)
        {
            int output = channels / 2;
            var act = new FACodecActivation<T>(engine, owner, channels);
            var up = new NormedConv1DLayer<T>(channels, output, 2 * s, s, 1, 1, s / 2 + s % 2, true, ConvolutionNormalization.Weight, outputPadding: s % 2);
            owner.Add(up);
            var units = new[] { 1, 3, 9 }.Select(d => new FACodecResidualUnit<T>(engine, owner, output, d)).ToArray();
            _blocks.Add((act, up, units));
            channels = output;
        }
        _act = new FACodecActivation<T>(engine, owner, channels);
        _last = XCodec2.WeightNormConv(owner, channels, 1, 7, 1, 1, 3);
    }

    /// <summary>The waveform <c>[1, 1, samples]</c> of <c>[1, input, frames]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x)
    {
        x = _first.Forward(x);
        foreach (var (act, up, units) in _blocks)
        {
            x = up.Forward(act.Forward(x));
            foreach (var unit in units) x = unit.Forward(x);
        }
        return _engine.Tanh(_last.Forward(_act.Forward(x)));
    }

    public void Load(Func<string, double[]> read, Func<string, double[]?> optional, string prefix)
    {
        FACodecWeights.Conv(read, _first, prefix + "0");
        for (int i = 0; i < _blocks.Count; i++)
        {
            var (act, up, units) = _blocks[i];
            string p = $"{prefix}{i + 1}.block";
            act.Load(read, optional, $"{p}.0");
            FACodecWeights.Conv(read, up, $"{p}.1");
            for (int u = 0; u < units.Length; u++) units[u].Load(read, optional, $"{p}.{u + 2}");
        }
        _act.Load(read, optional, $"{prefix}{_blocks.Count + 1}");
        FACodecWeights.Conv(read, _last, $"{prefix}{_blocks.Count + 2}");
    }
}

/// <summary>
/// FACodec's supervision and adversarial head (<c>CNNLSTM</c>, despite the name convolutional): three residual units
/// (dilations 1, 2, 3), SnakeBeta, then linear heads per frame, or on the frames' mean (the speaker head).
/// </summary>
internal sealed class FACodecPredictor<T>
{
    private readonly IEngine _engine;
    private readonly FACodecResidualUnit<T>[] _units;
    private readonly FACodecActivation<T> _act;
    private readonly int _input, _output;
    private readonly bool _global;

    public FACodecPredictor(IEngine engine, List<LayerBase<T>> owner, Random random, int input, int output, int heads, bool global = false)
    {
        _engine = engine;
        _input = input;
        _output = output;
        _global = global;
        _units = new[] { 1, 2, 3 }.Select(d => new FACodecResidualUnit<T>(engine, owner, input, d)).ToArray();
        _act = new FACodecActivation<T>(engine, owner, input);
        for (int h = 0; h < heads; h++) Heads.Add(IndexTtsOps.Linear(engine, owner, random, input, output));
    }

    public List<DenseLayer<T>> Heads { get; } = new();

    /// <summary>Each head's output: <c>[frames, out]</c>, or <c>[1, out]</c> for a global head.</summary>
    public List<Tensor<T>> Forward(Tensor<T> x)
    {
        foreach (var unit in _units) x = unit.Forward(x);
        var rows = Ns2Ops.Rows(_engine, _act.Forward(x));
        if (_global)
            rows = _engine.TensorMultiplyScalar(_engine.ReduceSum(rows, new[] { 0 }, keepDims: true), MathHelper.GetNumericOperations<T>().FromDouble(1.0 / rows.Shape[0]));
        return Heads.Select(h => h.Forward(rows)).ToList();
    }

    public void Load(Func<string, double[]> read, Func<string, double[]?> optional, string prefix)
    {
        for (int u = 0; u < _units.Length; u++) _units[u].Load(read, optional, $"{prefix}model.{u}");
        _act.Load(read, optional, $"{prefix}model.3");
        for (int h = 0; h < Heads.Count; h++)
            TorchParameters.Linear(_engine, Heads[h], _input, _output, read($"{prefix}heads.{h}.weight"), read($"{prefix}heads.{h}.bias"));
    }
}
