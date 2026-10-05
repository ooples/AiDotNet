using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// DiffWave's spectrogram upsampler layer (Kong et al. 2021, §5.1; reference <c>SpectrogramUpsampler</c>): a
/// single-channel transposed 2-D convolution over [mel, frames] with a (3, 32) kernel, stride (1, 16) and padding
/// (1, 8), which multiplies the frame rate by 16 and keeps the mel axis.
/// </summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.UpSampling)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 1, 4, 3", TestConstructorArgs = "16")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal sealed partial class DiffWaveUpsampleLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _stride;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _kernel;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _bias;

    public override bool SupportsTraining => true;

    public DiffWaveUpsampleLayer([LayerState] int stride)
        : base(new[] { 1 }, new[] { 1 })
    {
        _stride = stride;
        var random = RandomSeed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        // PyTorch's ConvTranspose2d default: U(−1/√fan_in, 1/√fan_in) with fan_in = out_channels · 3 · 2·stride.
        double bound = 1.0 / Math.Sqrt(3 * 2 * stride);
        _kernel = new Tensor<T>(new[] { 1, 1, 3, 2 * stride });
        for (int i = 0; i < _kernel.Length; i++) _kernel[i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);
        _bias = new Tensor<T>(new[] { 1 });
        _bias[0] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);
        RegisterTrainableParameter(_kernel, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    /// <summary>The upsampled map <c>[1, 1, mel, frames · stride]</c> of <c>[1, 1, mel, frames]</c>.</summary>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var y = Engine.ConvTranspose2D(input, _kernel, new[] { 1, _stride }, new[] { 1, _stride / 2 }, new[] { 0, 0 });
        return Engine.TensorAdd(y, Engine.TensorTile(Engine.Reshape(_bias, new[] { 1, 1, 1, 1 }), new[] { 1, 1, y.Shape[2], y.Shape[3] }));
    }

    public override void ResetState()
    {
    }

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Stride"] = _stride.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}

/// <summary>
/// The single-level Haar discrete wavelet transform along time (FreGrad §3.1, reference <c>pytorch_wavelets</c>
/// <c>DWT1DForward</c> / <c>DWT1DInverse</c> with the default <c>db1</c> wavelet): the low band
/// <c>(x[2i] + x[2i+1]) / √2</c> and the high band <c>(x[2i] − x[2i+1]) / √2</c>, each half the length, and its exact
/// inverse.
/// </summary>
internal static class HaarWavelet
{
    private static readonly double InverseSqrt2 = 1 / Math.Sqrt(2);

    /// <summary>The low and high bands <c>[1, C, L/2]</c> of <c>[1, C, L]</c> (L even).</summary>
    public static (Tensor<T> Low, Tensor<T> High) Forward<T>(IEngine engine, Tensor<T> x)
    {
        int c = x.Shape[1], half = x.Shape[2] / 2;
        if (x.Shape[2] % 2 != 0)
            throw new ArgumentException($"The Haar transform needs an even length, got {x.Shape[2]}.", nameof(x));
        var pairs = engine.Reshape(x, new[] { 1, c, half, 2 });
        var even = engine.Reshape(engine.TensorSlice(pairs, new[] { 0, 0, 0, 0 }, new[] { 1, c, half, 1 }), new[] { 1, c, half });
        var odd = engine.Reshape(engine.TensorSlice(pairs, new[] { 0, 0, 0, 1 }, new[] { 1, c, half, 1 }), new[] { 1, c, half });
        var scale = MathHelper.GetNumericOperations<T>().FromDouble(InverseSqrt2);
        return (engine.TensorMultiplyScalar(engine.TensorAdd(even, odd), scale), engine.TensorMultiplyScalar(engine.TensorSubtract(even, odd), scale));
    }

    /// <summary>The signal <c>[1, C, 2N]</c> of the low and high bands <c>[1, C, N]</c>.</summary>
    public static Tensor<T> Inverse<T>(IEngine engine, Tensor<T> low, Tensor<T> high)
    {
        int c = low.Shape[1], n = low.Shape[2];
        var scale = MathHelper.GetNumericOperations<T>().FromDouble(InverseSqrt2);
        var even = engine.Reshape(engine.TensorMultiplyScalar(engine.TensorAdd(low, high), scale), new[] { 1, c, n, 1 });
        var odd = engine.Reshape(engine.TensorMultiplyScalar(engine.TensorSubtract(low, high), scale), new[] { 1, c, n, 1 });
        return engine.Reshape(engine.TensorConcatenate(new[] { even, odd }, 3), new[] { 1, c, 2 * n });
    }
}

/// <summary>
/// DiffWave's ε_θ (Kong et al. 2021, §3, Fig. 2–3; reference lmnt-com/diffwave <c>DiffWave</c>): a 1×1 input projection
/// with ReLU; a sinusoidal diffusion-step encoding (128 values) through two shared 512-wide fully connected layers with
/// SiLU and, per residual layer, a C-wide projection added to that layer's input; N residual layers of a dilated
/// convolution (dilations 1, 2, …, 2^(cycle−1), repeating) to 2C channels plus a 1×1 projection of the upsampled mel,
/// a gate <c>σ(a) ⊙ tanh(b)</c> and a 1×1 projection split into the residual (added and scaled by 1/√2) and the skip;
/// the skips summed and scaled by 1/√N, a 1×1 projection, ReLU and a zero-initialized 1×1 output projection.
/// </summary>
/// <remarks>
/// <para>Convolutions are Kaiming-normal initialized (biases keep PyTorch's default), as the reference's
/// <c>Conv1d</c>; fully connected layers keep PyTorch's default.</para>
/// <para>FreGrad's variant (Nguyen et al. 2024, §3.1–3.2) takes and predicts several channels (the wavelet sub-bands)
/// and replaces each dilated convolution with the frequency-aware dilated convolution Freq-DConv: the Haar transform of
/// the hidden signal, the two bands concatenated along channels, one dilated convolution (PyTorch's default
/// initialization, as the reference's <c>DWTDilatedConv1D</c>) to twice the output channels, split into the two output
/// bands and inverse-transformed back to the input length.</para>
/// </remarks>
internal sealed class DiffWaveNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly int _channels;
    private readonly int _steps;
    private readonly NormedConv1DLayer<T> _input;
    private readonly NormedConv1DLayer<T> _embed1;
    private readonly NormedConv1DLayer<T> _embed2;
    private readonly List<DiffWaveUpsampleLayer<T>> _upsamplers = new();
    private readonly List<(NormedConv1DLayer<T> Dilated, NormedConv1DLayer<T> Step, NormedConv1DLayer<T> Condition, NormedConv1DLayer<T> Output)> _blocks = new();
    private readonly NormedConv1DLayer<T> _skip;
    private readonly NormedConv1DLayer<T> _output;
    private readonly bool _frequencyAware;

    public DiffWaveNetwork(IEngine engine, Random initialization, int melChannels, int channels, int layers, int cycle, int steps, int[] upsampleStrides,
        int audioChannels = 1, bool frequencyAware = false)
    {
        _engine = engine;
        _channels = channels;
        _steps = steps;
        _frequencyAware = frequencyAware;
        NormedConv1DLayer<T> Conv(int input, int output, int kernel, int dilation, int padding, bool kaiming)
        {
            var conv = new NormedConv1DLayer<T>(input, output, kernel, 1, dilation, 1, padding, false, ConvolutionNormalization.None);
            if (kaiming)
            {
                // torch.nn.init.kaiming_normal_ (fan_in, a = 0): N(0, 2 / fan_in).
                double std = Math.Sqrt(2.0 / conv.FanIn);
                conv.Reinitialize(() =>
                {
                    double u1 = 1.0 - initialization.NextDouble(), u2 = initialization.NextDouble();
                    return std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
                }, keepBias: true);
            }
            _layers.Add(conv);
            return conv;
        }
        _input = Conv(audioChannels, channels, 1, 1, 0, true);
        _embed1 = Conv(128, 512, 1, 1, 0, false);
        _embed2 = Conv(512, 512, 1, 1, 0, false);
        foreach (int stride in upsampleStrides)
        {
            var up = new DiffWaveUpsampleLayer<T>(stride);
            _layers.Add(up);
            _upsamplers.Add(up);
        }
        for (int i = 0; i < layers; i++)
        {
            int d = 1 << (i % cycle);
            var dilated = frequencyAware ? Conv(2 * channels, 4 * channels, 3, d, d, false) : Conv(channels, 2 * channels, 3, d, d, true);
            _blocks.Add((dilated, Conv(512, channels, 1, 1, 0, false),
                Conv(melChannels, 2 * channels, 1, 1, 0, true), Conv(channels, 2 * channels, 1, 1, 0, true)));
        }
        _skip = Conv(channels, channels, 1, 1, 0, true);
        _output = Conv(channels, audioChannels, 1, 1, 0, false);
        _output.Reinitialize(() => 0.0, keepBias: true);
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    // The step encoding [1, 128, 1]: sin and cos of t · 10^(4i/63), linearly interpolated for a fractional t.
    private Tensor<T> StepEncoding(double t)
    {
        double[] Row(int step)
        {
            var r = new double[128];
            for (int i = 0; i < 64; i++)
            {
                double v = step * Math.Pow(10.0, i * 4.0 / 63.0);
                r[i] = Math.Sin(v);
                r[64 + i] = Math.Cos(v);
            }
            return r;
        }
        int low = (int)Math.Floor(t), high = (int)Math.Ceiling(t);
        var a = Row(low);
        var b = Row(high);
        var encoding = new Tensor<T>(new[] { 1, 128, 1 });
        for (int i = 0; i < 128; i++) encoding[0, i, 0] = NumOps.FromDouble(a[i] + (b[i] - a[i]) * (t - low));
        return encoding;
    }

    // Freq-DConv: Haar bands of the hidden signal, concatenated, one dilated convolution, split, inverse Haar.
    private Tensor<T> FrequencyAware(NormedConv1DLayer<T> conv, Tensor<T> y)
    {
        var (low, high) = HaarWavelet.Forward(_engine, y);
        var z = conv.Forward(_engine.TensorConcatenate(new[] { low, high }, 1));
        int c = 2 * _channels, n = z.Shape[2];
        return HaarWavelet.Inverse(_engine, _engine.TensorSlice(z, new[] { 0, 0, 0 }, new[] { 1, c, n }),
            _engine.TensorSlice(z, new[] { 0, c, 0 }, new[] { 1, c, n }));
    }

    private Tensor<T> Silu(Tensor<T> x) => _engine.TensorMultiply(x, _engine.Sigmoid(x));

    /// <summary>ε_θ <c>[1, audio channels, samples]</c> for the noisy signal <c>[1, audio channels, samples]</c> at step
    /// <paramref name="step"/> given the mel spectrogram <c>[1, mel, frames]</c> (samples = frames · the upsampling).</summary>
    public Tensor<T> Forward(Tensor<T> noisy, double step, Tensor<T> mel)
    {
        int samples = noisy.Shape[2];
        var x = _engine.ReLU(_input.Forward(noisy));
        var embedding = Silu(_embed2.Forward(Silu(_embed1.Forward(StepEncoding(step)))));                     // [1, 512, 1]
        int m = mel.Shape[1], frames = mel.Shape[2];
        var c = _engine.Reshape(mel, new[] { 1, 1, m, frames });
        foreach (var up in _upsamplers) c = VocoderOps.LeakyRelu(_engine, up.Forward(c), 0.4);
        var condition = _engine.Reshape(c, new[] { 1, m, c.Shape[3] });
        Tensor<T>? skips = null;
        foreach (var (dilated, stepProjection, conditionProjection, output) in _blocks)
        {
            var y = _engine.TensorAdd(x, _engine.TensorTile(stepProjection.Forward(embedding), new[] { 1, 1, samples }));
            y = _engine.TensorAdd(_frequencyAware ? FrequencyAware(dilated, y) : dilated.Forward(y), conditionProjection.Forward(condition));
            var gate = _engine.TensorSlice(y, new[] { 0, 0, 0 }, new[] { 1, _channels, samples });
            var filter = _engine.TensorSlice(y, new[] { 0, _channels, 0 }, new[] { 1, _channels, samples });
            y = output.Forward(_engine.TensorMultiply(_engine.Sigmoid(gate), _engine.Tanh(filter)));
            var residual = _engine.TensorSlice(y, new[] { 0, 0, 0 }, new[] { 1, _channels, samples });
            var skip = _engine.TensorSlice(y, new[] { 0, _channels, 0 }, new[] { 1, _channels, samples });
            x = _engine.TensorMultiplyScalar(_engine.TensorAdd(x, residual), NumOps.FromDouble(1 / Math.Sqrt(2)));
            skips = skips is null ? skip : _engine.TensorAdd(skips, skip);
        }
        var h = _engine.TensorMultiplyScalar(skips!, NumOps.FromDouble(1 / Math.Sqrt(_blocks.Count)));
        return _output.Forward(_engine.ReLU(_skip.Forward(h)));
    }
}
