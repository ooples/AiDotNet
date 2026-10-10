using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>μ-law companding and quantization (WaveNet §2.2, Eq. 1; ITU-T G.711):
/// <c>f(x) = sign(x) ln(1 + μ|x|) / ln(1 + μ)</c> quantized to μ + 1 levels.</summary>
internal static class MuLaw
{
    /// <summary>The class (0..levels − 1) of a sample in [−1, 1] (reference <c>mulaw_quantize</c>:
    /// <c>⌊(f(x) + 1) / 2 · μ + 0.5⌋</c>).</summary>
    public static int Encode(double x, int levels)
    {
        double mu = levels - 1;
        x = Math.Max(-1, Math.Min(1, x));
        double f = Math.Sign(x) * Math.Log(1 + mu * Math.Abs(x)) / Math.Log(1 + mu);
        return (int)Math.Floor((f + 1) / 2 * mu + 0.5);
    }

    /// <summary>The sample of a class (reference <c>inv_mulaw_quantize</c>).</summary>
    public static double Decode(int q, int levels)
    {
        double mu = levels - 1;
        double f = 2.0 * q / mu - 1;
        return Math.Sign(f) * (Math.Pow(1 + mu, Math.Abs(f)) - 1) / mu;
    }
}

/// <summary>
/// A learned upsampler of conditioning features (WaveNet §2.5: "a transposed convolutional network"): one transposed
/// convolution per scale s with kernel 2s − (s mod 2), stride s and padding ⌊s/2⌋, which multiplies the length by s
/// exactly for even and odd s.
/// </summary>
internal sealed class TransposedUpsampler<T>
{
    private readonly List<NormedConv1DLayer<T>> _layers = new();

    public TransposedUpsampler(int channels, int[] scales, List<LayerBase<T>> owner)
    {
        foreach (int s in scales)
        {
            var up = new NormedConv1DLayer<T>(channels, channels, 2 * s - s % 2, s, 1, 1, s / 2, true, ConvolutionNormalization.None);
            owner.Add(up);
            _layers.Add(up);
        }
    }

    /// <summary>The features <c>[1, channels, frames]</c> upsampled to <c>[1, channels, frames · Π s]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x)
    {
        foreach (var up in _layers) x = up.Forward(x);
        return x;
    }
}

/// <summary>
/// WaveNet (van den Oord et al. 2016, §2, Fig. 4; reference r9y9/wavenet_vocoder for what the paper leaves open): a
/// causal convolution over the one-hot μ-law input, a stack of residual blocks — a dilated causal convolution with
/// kernel 2 and dilations 1, 2, …, 512 repeated, the gated unit <c>tanh(W_f x + V_f y) ⊙ σ(W_g x + V_g y)</c> with
/// the upsampled mel y projected by 1×1 convolutions (§2.5), 1×1 residual and skip projections — and the output
/// ReLU → 1×1 → ReLU → 1×1 → softmax over the μ-law classes. The mel spectrogram is upsampled by a transposed
/// convolutional network (§2.5).
/// </summary>
/// <remarks>Reference choices: weight-normalized convolutions with Kaiming-normal weights and zero biases, bias-free
/// condition projections, the residual sum scaled by √0.5 and the skip sum by √(1 / layers).</remarks>
internal sealed class WaveNetNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly int _classes;
    private readonly int _gate;
    private readonly int _kernel;
    private readonly TransposedUpsampler<T> _upsample;
    private readonly NormedConv1DLayer<T> _input;
    private readonly List<Block> _blocks = new();
    private readonly NormedConv1DLayer<T> _post1;
    private readonly NormedConv1DLayer<T> _post2;

    private sealed record Block(int Dilation, NormedConv1DLayer<T> Dilated, NormedConv1DLayer<T> Condition,
        NormedConv1DLayer<T> Residual, NormedConv1DLayer<T> Skip);

    public WaveNetNetwork(IEngine engine, Random initialization, int classes, int melChannels, int residual, int gate, int skip,
        int layers, int cycle, int kernel, int[] upsampleScales)
    {
        _engine = engine;
        _classes = classes;
        _gate = gate;
        _kernel = kernel;
        NormedConv1DLayer<T> Conv(int input, int output, int k, int dilation, bool bias)
        {
            var conv = new NormedConv1DLayer<T>(input, output, k, 1, dilation, 1, 0, false, ConvolutionNormalization.Weight, bias);
            // nn.init.kaiming_normal_(nonlinearity = "relu"): N(0, 2 / fan_in); zero bias; weight norm from the result.
            double std = Math.Sqrt(2.0 / conv.FanIn);
            conv.Reinitialize(() =>
            {
                double u1 = 1.0 - initialization.NextDouble(), u2 = initialization.NextDouble();
                return std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            });
            _layers.Add(conv);
            return conv;
        }
        _upsample = new TransposedUpsampler<T>(melChannels, upsampleScales, _layers);
        _input = Conv(classes, residual, kernel, 1, true);
        for (int i = 0; i < layers; i++)
        {
            int d = 1 << (i % cycle);
            _blocks.Add(new Block(d, Conv(residual, 2 * gate, kernel, d, true), Conv(melChannels, 2 * gate, 1, 1, false),
                Conv(gate, residual, 1, 1, true), Conv(gate, skip, 1, 1, true)));
        }
        _post1 = Conv(skip, skip, 1, 1, true);
        _post2 = Conv(skip, classes, 1, 1, true);
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    /// <summary>The samples of history every layer needs (the receptive field): 1 + (k − 1)(1 + Σ dilations).</summary>
    public int ReceptiveField => 1 + (_kernel - 1) * (1 + _blocks.Sum(b => b.Dilation));

    /// <summary>The mel spectrogram <c>[1, mel, frames]</c> upsampled to one vector per sample <c>[1, mel, frames ·
    /// Π scales]</c>.</summary>
    public Tensor<T> Upsample(Tensor<T> mel) => _upsample.Forward(mel);

    private Tensor<T> Slice(Tensor<T> x, int from, int count)
        => _engine.TensorSlice(x, new[] { 0, from, 0 }, new[] { 1, count, x.Shape[2] });

    private Tensor<T> Causal(NormedConv1DLayer<T> conv, Tensor<T> x, int dilation)
        => conv.Forward(VocoderOps.ZeroPad(_engine, x, (_kernel - 1) * dilation, 0));

    private (Tensor<T> Residual, Tensor<T> Skip) Gate(Block b, Tensor<T> dilated, Tensor<T> condition)
    {
        var a = _engine.TensorAdd(dilated, b.Condition.Forward(condition));
        var z = _engine.TensorMultiply(_engine.Tanh(Slice(a, 0, _gate)), _engine.Sigmoid(Slice(a, _gate, _gate)));
        return (b.Residual.Forward(z), b.Skip.Forward(z));
    }

    private Tensor<T> Output(Tensor<T> skips)
    {
        var scaled = _engine.TensorMultiplyScalar(skips, NumOps.FromDouble(Math.Sqrt(1.0 / _blocks.Count)));
        return _post2.Forward(_engine.ReLU(_post1.Forward(_engine.ReLU(scaled))));
    }

    /// <summary>The logits <c>[1, classes, samples]</c> of every next sample given the one-hot previous samples
    /// <c>[1, classes, samples]</c> and the upsampled condition <c>[1, mel, samples]</c> (teacher forcing).</summary>
    public Tensor<T> Forward(Tensor<T> previous, Tensor<T> condition)
    {
        var x = Causal(_input, previous, 1);
        Tensor<T>? skips = null;
        var half = NumOps.FromDouble(Math.Sqrt(0.5));
        foreach (var b in _blocks)
        {
            var (res, skip) = Gate(b, Causal(b.Dilated, x, b.Dilation), condition);
            x = _engine.TensorMultiplyScalar(_engine.TensorAdd(x, res), half);
            skips = skips is null ? skip : _engine.TensorAdd(skips, skip);
        }
        return Output(skips!);
    }

    /// <summary>Incremental generation state: the inputs every causal convolution has seen.</summary>
    public sealed class State
    {
        internal readonly List<Tensor<T>> Inputs = new();
        internal readonly List<List<Tensor<T>>> Hidden = new();
    }

    /// <summary>A fresh generation state.</summary>
    public State NewState()
    {
        var state = new State();
        foreach (var _ in _blocks) state.Hidden.Add(new List<Tensor<T>>());
        return state;
    }

    private Tensor<T> Tap(List<Tensor<T>> history, int back, int channels)
        => history.Count - 1 - back >= 0 ? history[history.Count - 1 - back] : new Tensor<T>(new[] { 1, channels, 1 });

    private IReadOnlyList<Tensor<T>> Taps(List<Tensor<T>> history, int dilation, int channels)
    {
        var taps = new Tensor<T>[_kernel];
        for (int j = 0; j < _kernel; j++) taps[j] = Tap(history, (_kernel - 1 - j) * dilation, channels);
        return taps;
    }

    /// <summary>The logits <c>[1, classes, 1]</c> of the next sample given the one-hot previous sample
    /// <c>[1, classes, 1]</c> and the condition at the next sample <c>[1, mel, 1]</c>, advancing the state: the same
    /// function as <see cref="Forward"/> evaluated one step at a time.</summary>
    public Tensor<T> Step(State state, Tensor<T> previous, Tensor<T> condition)
    {
        state.Inputs.Add(previous);
        var x = _input.ForwardTaps(Taps(state.Inputs, 1, _classes));
        Tensor<T>? skips = null;
        var half = NumOps.FromDouble(Math.Sqrt(0.5));
        for (int i = 0; i < _blocks.Count; i++)
        {
            var b = _blocks[i];
            var history = state.Hidden[i];
            history.Add(x);
            if (history.Count > (_kernel - 1) * b.Dilation + 1) history.RemoveAt(0);
            var (res, skip) = Gate(b, b.Dilated.ForwardTaps(Taps(history, b.Dilation, x.Shape[1])), condition);
            x = _engine.TensorMultiplyScalar(_engine.TensorAdd(x, res), half);
            skips = skips is null ? skip : _engine.TensorAdd(skips, skip);
        }
        if (state.Inputs.Count > _kernel) state.Inputs.RemoveAt(0);
        return Output(skips!);
    }
}
