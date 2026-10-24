using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// BigVGAN's anti-aliased Snake activation (Lee et al. 2023, §3.3, App. A; reference <c>Activation1d</c>): upsample ×2
/// with a Kaiser-windowed sinc low-pass filter, apply Snake, low-pass and downsample ×2.
/// </summary>
/// <remarks>The filter follows StyleGAN3 as the reference's <c>kaiser_sinc_filter1d</c>: 12 taps, cutoff 0.25 and
/// half-width 0.3 of the doubled rate, Kaiser β from the attenuation <c>A = 2.285 (n/2 − 1) π · 4·0.3 + 7.95</c>,
/// normalized to unit sum. Upsampling replicate-pads 5 samples, transposes-convolves with twice the filter at stride 2 and
/// crops 15 samples at each end; downsampling replicate-pads 5 and 6 samples and convolves at stride 2.</remarks>
internal sealed class AntiAliasedSnake<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const int Taps = 12;
    private readonly IEngine _engine;
    private readonly SnakeLayer<T> _snake;
    private readonly Tensor<T> _upFilter;     // [1, 1, 1, 12], scaled by the ratio
    private readonly Tensor<T> _downFilter;   // [1, 1, 1, 12]

    public AntiAliasedSnake(IEngine engine, List<LayerBase<T>> owned, int channels)
    {
        _engine = engine;
        owned.Add(_snake = new SnakeLayer<T>(channels));
        var filter = KaiserSinc(0.25, 0.3, Taps);
        _upFilter = new Tensor<T>(new[] { 1, 1, 1, Taps });
        _downFilter = new Tensor<T>(new[] { 1, 1, 1, Taps });
        for (int i = 0; i < Taps; i++)
        {
            _upFilter[0, 0, 0, i] = NumOps.FromDouble(2 * filter[i]);
            _downFilter[0, 0, 0, i] = NumOps.FromDouble(filter[i]);
        }
    }

    internal static double[] KaiserSinc(double cutoff, double halfWidth, int kernel)
    {
        bool even = kernel % 2 == 0;
        int half = kernel / 2;
        double deltaF = 4 * halfWidth;
        double a = 2.285 * (half - 1) * Math.PI * deltaF + 7.95;
        double beta = a > 50 ? 0.1102 * (a - 8.7) : a >= 21 ? 0.5842 * Math.Pow(a - 21, 0.4) + 0.07886 * (a - 21) : 0.0;
        var result = new double[kernel];
        double sum = 0;
        for (int n = 0; n < kernel; n++)
        {
            double ratio = 2.0 * n / (kernel - 1) - 1;                                  // torch.kaiser_window(periodic=False)
            double window = BesselI0(beta * Math.Sqrt(Math.Max(0, 1 - ratio * ratio))) / BesselI0(beta);
            double time = even ? n - half + 0.5 : n - half;
            double x = 2 * cutoff * time;
            double sinc = x == 0 ? 1.0 : Math.Sin(Math.PI * x) / (Math.PI * x);
            result[n] = 2 * cutoff * window * sinc;
            sum += result[n];
        }
        for (int n = 0; n < kernel; n++) result[n] /= sum;
        return result;
    }

    private static double BesselI0(double x)
    {
        double total = 1, term = 1, halfX = x / 2;
        for (int k = 1; k < 200; k++)
        {
            term *= halfX / k * (halfX / k);
            total += term;
            if (term < 1e-17 * total) break;
        }
        return total;
    }

    // Replicate padding of [C, 1, 1, T] along time.
    private Tensor<T> ReplicatePad(Tensor<T> x, int left, int right)
    {
        int c = x.Shape[0], t = x.Shape[3];
        var index = new Tensor<int>(new[] { t + left + right });
        for (int i = 0; i < index.Length; i++) index[i] = Math.Min(Math.Max(i - left, 0), t - 1);
        var rows = _engine.TensorIndexSelect(_engine.TensorTranspose(_engine.Reshape(x, new[] { c, t })), index, 0);
        return _engine.Reshape(_engine.TensorTranspose(rows), new[] { c, 1, 1, t + left + right });
    }

    /// <summary>The activation of <c>[1, C, T]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x)
    {
        int c = x.Shape[1], t = x.Shape[2];
        // Channels become the batch so one single-channel filter acts on each (a depthwise filter).
        var bands = _engine.Reshape(x, new[] { c, 1, 1, t });
        var up = _engine.ConvTranspose2D(ReplicatePad(bands, 5, 5), _upFilter, new[] { 1, 2 }, new[] { 0, 0 }, new[] { 0, 0 });
        int upLength = up.Shape[3];
        up = _engine.TensorSlice(up, new[] { 0, 0, 0, 15 }, new[] { c, 1, 1, upLength - 30 });                   // 2T samples
        var activated = _engine.Reshape(_snake.Forward(_engine.Reshape(up, new[] { 1, c, 2 * t })), new[] { c, 1, 1, 2 * t });
        var down = _engine.Conv2D(ReplicatePad(activated, 5, 6), _downFilter, new[] { 1, 2 }, new[] { 0, 0 }, new[] { 1, 1 });
        return _engine.Reshape(down, new[] { 1, c, t });
    }
}

/// <summary>
/// BigVGAN's generator (Lee et al. 2023, §3.2–3.4, App. A, Fig. 3; reference <c>BigVGAN</c>, <c>AMPBlock1</c>): HiFi-GAN's
/// generator with every leaky ReLU replaced by the anti-aliased Snake. A 7-wide convolution to h channels; per upsampling
/// stage a transposed convolution halving the channels and the average of AMP blocks (per kernel, per dilation:
/// activation, dilated convolution, activation, convolution, residual); a final anti-aliased Snake, a 7-wide convolution
/// to one channel and tanh. Weight-normalized throughout; the upsampling, AMP and output convolutions draw N(0, 0.01).
/// </summary>
internal sealed class BigVganGenerator<T>
{
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly NormedConv1DLayer<T> _pre;
    private readonly List<NormedConv1DLayer<T>> _ups = new();
    private readonly List<List<List<(AntiAliasedSnake<T> A1, NormedConv1DLayer<T> C1, AntiAliasedSnake<T> A2, NormedConv1DLayer<T> C2)>>> _blocks = new();
    private readonly AntiAliasedSnake<T> _postActivation;
    private readonly NormedConv1DLayer<T> _post;

    public BigVganGenerator(IEngine engine, Random initialization, int melChannels, int initialChannels, int[] upsampleRates,
        int[] upsampleKernels, int[] resblockKernels, int[][] resblockDilations)
    {
        _engine = engine;
        Func<double> normal = () =>
        {
            double u1 = 1.0 - initialization.NextDouble(), u2 = initialization.NextDouble();
            return 0.01 * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        };
        _pre = Add(new NormedConv1DLayer<T>(melChannels, initialChannels, 7, 1, 1, 1, 3, false, ConvolutionNormalization.Weight));
        int channels = initialChannels;
        for (int i = 0; i < upsampleRates.Length; i++)
        {
            int next = initialChannels >> (i + 1);
            var up = Add(new NormedConv1DLayer<T>(channels, next, upsampleKernels[i], upsampleRates[i], 1, 1,
                (upsampleKernels[i] - upsampleRates[i]) / 2, true, ConvolutionNormalization.Weight));
            up.Reinitialize(normal, keepBias: true);
            _ups.Add(up);
            channels = next;
            var stage = new List<List<(AntiAliasedSnake<T>, NormedConv1DLayer<T>, AntiAliasedSnake<T>, NormedConv1DLayer<T>)>>();
            for (int j = 0; j < resblockKernels.Length; j++)
            {
                int k = resblockKernels[j];
                var block = new List<(AntiAliasedSnake<T>, NormedConv1DLayer<T>, AntiAliasedSnake<T>, NormedConv1DLayer<T>)>();
                foreach (int d in resblockDilations[j])
                {
                    var a1 = new AntiAliasedSnake<T>(engine, _layers, channels);
                    var c1 = Add(new NormedConv1DLayer<T>(channels, channels, k, 1, d, 1, (k * d - d) / 2, false, ConvolutionNormalization.Weight));
                    var a2 = new AntiAliasedSnake<T>(engine, _layers, channels);
                    var c2 = Add(new NormedConv1DLayer<T>(channels, channels, k, 1, 1, 1, (k - 1) / 2, false, ConvolutionNormalization.Weight));
                    c1.Reinitialize(normal, keepBias: true);
                    c2.Reinitialize(normal, keepBias: true);
                    block.Add((a1, c1, a2, c2));
                }
                stage.Add(block);
            }
            _blocks.Add(stage);
        }
        _postActivation = new AntiAliasedSnake<T>(engine, _layers, channels);
        _post = Add(new NormedConv1DLayer<T>(channels, 1, 7, 1, 1, 1, 3, false, ConvolutionNormalization.Weight));
        _post.Reinitialize(normal, keepBias: true);
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    private NormedConv1DLayer<T> Add(NormedConv1DLayer<T> layer)
    {
        _layers.Add(layer);
        return layer;
    }

    /// <summary>A waveform <c>[1, 1, frames · Π u]</c> from a mel spectrogram <c>[1, mel, frames]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> mel)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var x = _pre.Forward(mel);
        for (int i = 0; i < _ups.Count; i++)
        {
            x = _ups[i].Forward(x);
            Tensor<T>? sum = null;
            foreach (var block in _blocks[i])
            {
                var h = x;
                foreach (var (a1, c1, a2, c2) in block)
                    h = _engine.TensorAdd(c2.Forward(a2.Forward(c1.Forward(a1.Forward(h)))), h);
                sum = sum is null ? h : _engine.TensorAdd(sum, h);
            }
            x = _engine.TensorMultiplyScalar(sum!, ops.FromDouble(1.0 / _blocks[i].Count));
        }
        return _engine.Tanh(_post.Forward(_postActivation.Forward(x)));
    }
}
