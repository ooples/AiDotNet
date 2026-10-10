using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Waveform operations the GAN vocoders share.</summary>
internal static class VocoderOps
{
    /// <summary>Reflection padding of <c>[1, C, T]</c> along time (PyTorch <c>ReflectionPad1d</c>).</summary>
    public static Tensor<T> ReflectPad<T>(IEngine engine, Tensor<T> x, int left, int right)
    {
        if (left == 0 && right == 0) return x;
        int c = x.Shape[1], t = x.Shape[2];
        var index = new Tensor<int>(new[] { t + left + right });
        for (int i = 0; i < index.Length; i++) index[i] = DifferentiableMel<T>.Reflect(i - left, t);
        var rows = engine.TensorTranspose(engine.Reshape(x, new[] { c, t }));                   // [T, C]
        var padded = engine.TensorIndexSelect(rows, index, 0);
        return engine.Reshape(engine.TensorTranspose(padded), new[] { 1, c, t + left + right });
    }

    /// <summary>Zero padding of <c>[1, C, T]</c> along time.</summary>
    public static Tensor<T> ZeroPad<T>(IEngine engine, Tensor<T> x, int left, int right)
    {
        if (left == 0 && right == 0) return x;
        int c = x.Shape[1];
        var parts = new List<Tensor<T>>();
        if (left > 0) parts.Add(new Tensor<T>(new[] { 1, c, left }));
        parts.Add(x);
        if (right > 0) parts.Add(new Tensor<T>(new[] { 1, c, right }));
        return engine.TensorConcatenate(parts.ToArray(), 2);
    }

    /// <summary>
    /// <c>AvgPool1d(kernel, stride, padding, count_include_pad=False)</c> of <c>[1, C, T]</c>: the mean of the real
    /// samples under each window of the zero-padded signal.
    /// </summary>
    public static Tensor<T> AveragePool<T>(IEngine engine, Tensor<T> x, int kernel, int stride, int padding)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int c = x.Shape[1], t = x.Shape[2];
        int outLength = (t + 2 * padding - kernel) / stride + 1;
        var ones = new Tensor<T>(new[] { c, 1, 1, kernel });
        for (int i = 0; i < ones.Length; i++) ones[i] = ops.One;
        var padded = engine.Reshape(ZeroPad(engine, x, padding, padding), new[] { 1, c, 1, t + 2 * padding });
        Tensor<T> sums;
        if (c == 1)
        {
            sums = engine.Conv2D(padded, ones, new[] { 1, stride }, new[] { 0, 0 }, new[] { 1, 1 });
        }
        else
        {
            var parts = new Tensor<T>[c];
            for (int ch = 0; ch < c; ch++)
                parts[ch] = engine.Conv2D(engine.TensorSlice(padded, new[] { 0, ch, 0, 0 }, new[] { 1, 1, 1, t + 2 * padding }),
                    engine.TensorSlice(ones, new[] { ch, 0, 0, 0 }, new[] { 1, 1, 1, kernel }), new[] { 1, stride }, new[] { 0, 0 }, new[] { 1, 1 });
            sums = engine.TensorConcatenate(parts, 1);
        }
        var inverse = new Tensor<T>(new[] { 1, 1, 1, outLength });
        for (int o = 0; o < outLength; o++)
        {
            int from = o * stride - padding, to = from + kernel;
            int count = Math.Min(to, t) - Math.Max(from, 0);
            inverse[0, 0, 0, o] = ops.FromDouble(1.0 / Math.Max(count, 1));
        }
        var mean = engine.TensorMultiply(sums, engine.TensorTile(inverse, new[] { 1, c, 1, 1 }));
        return engine.Reshape(mean, new[] { 1, c, outLength });
    }

    /// <summary>Leaky ReLU with the given slope.</summary>
    public static Tensor<T> LeakyRelu<T>(IEngine engine, Tensor<T> x, double slope)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        // max(x, 0) + slope · min(x, 0) = relu(x) − slope · relu(−x)
        return engine.TensorSubtract(engine.ReLU(x), engine.TensorMultiplyScalar(engine.ReLU(engine.TensorNegate(x)), ops.FromDouble(slope)));
    }
}

/// <summary>
/// MelGAN's generator (Kumar et al. 2019, §2.1, App. A Table 6a, Fig. 4): a 7-wide convolution to 16·ngf channels, then
/// per upsampling ratio r a leaky ReLU, a transposed convolution of kernel 2r and stride r halving the channels, and a
/// residual stack of three dilated blocks (dilations 1, 3, 9); finally a leaky ReLU, a 7-wide convolution to one channel
/// and tanh. Every convolution is weight-normalized; the 7-wide convolutions and the dilated ones reflect-pad their input.
/// </summary>
/// <remarks>Each residual block is <c>x + [lReLU → 3×1 conv (dilation d) → lReLU → 3×1 conv (dilation 1)](x)</c> through
/// a weight-normalized 1×1 shortcut (reference <c>ResnetBlock</c>; the paper's Fig. 4 gives the second convolution a
/// 3-wide kernel, the reference a 1-wide one). Leaky ReLU slope 0.2.</remarks>
internal sealed class MelGanGenerator<T>
{
    private const double Slope = 0.2;
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly NormedConv1DLayer<T> _pre;
    private readonly List<(NormedConv1DLayer<T> Up, List<(int Dilation, NormedConv1DLayer<T> Conv1, NormedConv1DLayer<T> Conv2, NormedConv1DLayer<T> Shortcut)> Stack)> _stages = new();
    private readonly NormedConv1DLayer<T> _post;

    /// <param name="engine">The engine.</param>
    /// <param name="melChannels">Input mel bands.</param>
    /// <param name="initialChannels">The first convolution's width (MelGAN: ngf · 2^ratios = 512); every upsampling
    /// halves it.</param>
    /// <param name="ratios">Upsampling ratios.</param>
    /// <param name="residualLayers">Dilated blocks per residual stack (dilations 1, 3, 9, ...).</param>
    /// <param name="outChannels">Output channels (1, or the sub-bands of Multi-band MelGAN).</param>
    public MelGanGenerator(IEngine engine, int melChannels, int initialChannels, int[] ratios, int residualLayers, int outChannels = 1)
    {
        _engine = engine;
        int channels = initialChannels;
        _pre = Add(new NormedConv1DLayer<T>(melChannels, channels, 7, 1, 1, 1, 0, false, ConvolutionNormalization.Weight));
        foreach (int r in ratios)
        {
            int next = channels / 2;
            var up = Add(new NormedConv1DLayer<T>(channels, next, 2 * r, r, 1, 1, r / 2 + r % 2, true, ConvolutionNormalization.Weight,
                outputPadding: r % 2));
            var stack = new List<(int, NormedConv1DLayer<T>, NormedConv1DLayer<T>, NormedConv1DLayer<T>)>();
            for (int j = 0; j < residualLayers; j++)
            {
                int d = (int)Math.Pow(3, j);
                stack.Add((d,
                    Add(new NormedConv1DLayer<T>(next, next, 3, 1, d, 1, 0, false, ConvolutionNormalization.Weight)),
                    Add(new NormedConv1DLayer<T>(next, next, 3, 1, 1, 1, 0, false, ConvolutionNormalization.Weight)),
                    Add(new NormedConv1DLayer<T>(next, next, 1, 1, 1, 1, 0, false, ConvolutionNormalization.Weight))));
            }
            _stages.Add((up, stack));
            channels = next;
        }
        _post = Add(new NormedConv1DLayer<T>(channels, outChannels, 7, 1, 1, 1, 0, false, ConvolutionNormalization.Weight));
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    private NormedConv1DLayer<T> Add(NormedConv1DLayer<T> layer)
    {
        _layers.Add(layer);
        return layer;
    }

    /// <summary>A waveform <c>[1, outChannels, frames · Π r]</c> from a mel spectrogram <c>[1, mel, frames]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> mel)
    {
        var x = _pre.Forward(VocoderOps.ReflectPad(_engine, mel, 3, 3));
        foreach (var (up, stack) in _stages)
        {
            x = up.Forward(VocoderOps.LeakyRelu(_engine, x, Slope));
            foreach (var (d, conv1, conv2, shortcut) in stack)
            {
                var y = conv1.Forward(VocoderOps.ReflectPad(_engine, VocoderOps.LeakyRelu(_engine, x, Slope), d, d));
                y = conv2.Forward(VocoderOps.ReflectPad(_engine, VocoderOps.LeakyRelu(_engine, y, Slope), 1, 1));
                x = _engine.TensorAdd(shortcut.Forward(x), y);
            }
        }
        x = _post.Forward(VocoderOps.ReflectPad(_engine, VocoderOps.LeakyRelu(_engine, x, Slope), 3, 3));
        return _engine.Tanh(x);
    }
}

/// <summary>
/// MelGAN's multi-scale discriminator (§2.2, Table 6b): three identical window-based discriminators on the raw audio and
/// on audio average-pooled by 2 and 4 (kernel 4, stride 2, padding 1, not counting the padding). Each is a reflect-padded
/// 15-wide convolution to 16 channels, four grouped convolutions of kernel 41 and stride 4 (64, 256, 1024, 1024
/// channels; groups 4, 16, 64, 256), a 5-wide convolution and a 3-wide convolution to one score per window, with leaky
/// ReLU (0.2) between them and weight normalization throughout.
/// </summary>
internal sealed class MelGanDiscriminators<T>
{
    private const double Slope = 0.2;
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly List<List<NormedConv1DLayer<T>>> _discriminators = new();

    /// <param name="engine">The engine.</param>
    /// <param name="scales">Discriminators (3).</param>
    /// <param name="downsampleChannels">The strided convolutions' widths (MelGAN 64, 256, 1024, 1024; Multi-band
    /// MelGAN 64, 256, 512); each has a quarter of its input channels as groups.</param>
    /// <param name="widthDivisor">A divisor on every width (1: the paper's), keeping the group structure.</param>
    public MelGanDiscriminators(IEngine engine, int scales, int[] downsampleChannels, int widthDivisor = 1)
    {
        _engine = engine;
        int Width(int w) => Math.Max(1, w / widthDivisor);
        for (int k = 0; k < scales; k++)
        {
            var convs = new List<NormedConv1DLayer<T>>();
            int nf = Width(16);
            convs.Add(Add(new NormedConv1DLayer<T>(1, nf, 15, 1, 1, 1, 0, false, ConvolutionNormalization.Weight)));
            foreach (int o in downsampleChannels)
            {
                int next = Width(o), groups = Math.Max(1, nf / 4);
                while (nf % groups != 0 || next % groups != 0) groups--;
                convs.Add(Add(new NormedConv1DLayer<T>(nf, next, 41, 4, 1, groups, 20, false, ConvolutionNormalization.Weight)));
                nf = next;
            }
            convs.Add(Add(new NormedConv1DLayer<T>(nf, nf, 5, 1, 1, 1, 2, false, ConvolutionNormalization.Weight)));
            convs.Add(Add(new NormedConv1DLayer<T>(nf, 1, 3, 1, 1, 1, 1, false, ConvolutionNormalization.Weight)));
            _discriminators.Add(convs);
        }
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    private NormedConv1DLayer<T> Add(NormedConv1DLayer<T> layer)
    {
        _layers.Add(layer);
        return layer;
    }

    /// <summary>Per discriminator, the window scores and the intermediate feature maps, for a waveform <c>[samples]</c>.</summary>
    public List<(Tensor<T> Score, List<Tensor<T>> Features)> Forward(Tensor<T> audio)
    {
        var results = new List<(Tensor<T>, List<Tensor<T>>)>();
        var x = _engine.Reshape(audio, new[] { 1, 1, audio.Length });
        for (int k = 0; k < _discriminators.Count; k++)
        {
            if (k > 0) x = VocoderOps.AveragePool(_engine, x, 4, 2, 1);
            var convs = _discriminators[k];
            var features = new List<Tensor<T>>();
            var h = VocoderOps.ReflectPad(_engine, x, 7, 7);
            for (int i = 0; i < convs.Count - 1; i++)
            {
                h = VocoderOps.LeakyRelu(_engine, convs[i].Forward(h), Slope);
                features.Add(h);
            }
            results.Add((convs[^1].Forward(h), features));
        }
        return results;
    }
}
