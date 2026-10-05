using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// UnivNet's kernel predictor (Jang et al. 2021, §3.1; LVCNet, Zeng et al. 2021; reference maum-ai/univnet
/// <c>KernelPredictor</c>): from the log-mel spectrogram, the kernels and biases of every location-variable convolution
/// of one residual stack, one set per mel frame.
/// </summary>
/// <remarks>A 5-wide convolution to 64 channels with leaky ReLU, three residual blocks of two 3-wide convolutions with
/// leaky ReLU, then 3-wide convolutions to <c>layers · in · out · kernel</c> kernel and <c>layers · out</c> bias channels;
/// weight-normalized throughout.</remarks>
internal sealed class UnivNetKernelPredictor<T>
{
    private readonly IEngine _engine;
    private readonly double _slope;
    private readonly NormedConv1DLayer<T> _input;
    private readonly List<(NormedConv1DLayer<T> A, NormedConv1DLayer<T> B)> _residual = new();
    private readonly NormedConv1DLayer<T> _kernel;
    private readonly NormedConv1DLayer<T> _bias;

    public UnivNetKernelPredictor(IEngine engine, List<LayerBase<T>> owned, int condChannels, int inChannels, int outChannels,
        int layers, int kernelSize, int hidden, int convSize, double slope)
    {
        _engine = engine;
        _slope = slope;
        int pad = (convSize - 1) / 2;
        owned.Add(_input = new NormedConv1DLayer<T>(condChannels, hidden, 5, 1, 1, 1, 2, false, ConvolutionNormalization.Weight));
        for (int i = 0; i < 3; i++)
        {
            var a = new NormedConv1DLayer<T>(hidden, hidden, convSize, 1, 1, 1, pad, false, ConvolutionNormalization.Weight);
            var b = new NormedConv1DLayer<T>(hidden, hidden, convSize, 1, 1, 1, pad, false, ConvolutionNormalization.Weight);
            owned.Add(a);
            owned.Add(b);
            _residual.Add((a, b));
        }
        owned.Add(_kernel = new NormedConv1DLayer<T>(hidden, inChannels * outChannels * kernelSize * layers, convSize, 1, 1, 1, pad, false, ConvolutionNormalization.Weight));
        owned.Add(_bias = new NormedConv1DLayer<T>(hidden, outChannels * layers, convSize, 1, 1, 1, pad, false, ConvolutionNormalization.Weight));
    }

    /// <summary>The kernels <c>[1, layers · in · out · kernel, frames]</c> and biases <c>[1, layers · out, frames]</c>.</summary>
    public (Tensor<T> Kernels, Tensor<T> Biases) Forward(Tensor<T> mel)
    {
        var c = VocoderOps.LeakyRelu(_engine, _input.Forward(mel), _slope);
        foreach (var (a, b) in _residual)
            c = _engine.TensorAdd(c, VocoderOps.LeakyRelu(_engine, b.Forward(VocoderOps.LeakyRelu(_engine, a.Forward(c), _slope)), _slope));
        return (_kernel.Forward(c), _bias.Forward(c));
    }
}

/// <summary>
/// One UnivNet residual stack (reference <c>LVCBlock</c>): a leaky ReLU and transposed convolution upsampling by the
/// stride, then per dilation a leaky-ReLU-wrapped dilated convolution, a location-variable convolution with this
/// position's predicted kernel (doubling the channels) and a gated activation unit added to the input:
/// <c>x ← x + σ(o_a) ⊙ tanh(o_b)</c>.
/// </summary>
internal sealed class UnivNetLvcBlock<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly double _slope;
    private readonly int _channels;
    private readonly int _kernelSize;
    private readonly int _hop;
    private readonly int[] _dilations;
    private readonly NormedConv1DLayer<T> _up;
    private readonly List<NormedConv1DLayer<T>> _convs = new();
    private readonly UnivNetKernelPredictor<T> _predictor;
    private readonly Tensor<int> _kernelOrder;

    public UnivNetLvcBlock(IEngine engine, List<LayerBase<T>> owned, int channels, int condChannels, int stride, int[] dilations,
        double slope, int condHopLength, int kernelSize, int predictorHidden, int predictorConvSize)
    {
        _engine = engine;
        _slope = slope;
        _channels = channels;
        _kernelSize = kernelSize;
        _hop = condHopLength;
        _dilations = dilations;
        _predictor = new UnivNetKernelPredictor<T>(engine, owned, condChannels, channels, 2 * channels, dilations.Length, kernelSize,
            predictorHidden, predictorConvSize, slope);
        owned.Add(_up = new NormedConv1DLayer<T>(channels, channels, 2 * stride, stride, 1, 1, stride / 2 + stride % 2, true,
            ConvolutionNormalization.Weight, outputPadding: stride % 2));
        foreach (int d in dilations)
        {
            var conv = new NormedConv1DLayer<T>(channels, channels, kernelSize, 1, d, 1, d * (kernelSize - 1) / 2, false, ConvolutionNormalization.Weight);
            owned.Add(conv);
            _convs.Add(conv);
        }
        // The predictor lays a layer's kernel out as [in, out, k] (view(batch, layers, in, out, k, frames)); the batched
        // product needs rows in (k, in) order with out as columns: row (j · in + i) · out + o reads ((i · out + o) · k + j).
        int c = channels, o2 = 2 * channels, k = kernelSize;
        _kernelOrder = new Tensor<int>(new[] { c * o2 * k });
        for (int j = 0; j < k; j++)
            for (int i = 0; i < c; i++)
                for (int o = 0; o < o2; o++)
                    _kernelOrder[(j * c + i) * o2 + o] = (i * o2 + o) * k + j;
    }

    /// <summary>The stack's output <c>[1, channels, T · stride]</c> for its input <c>[1, channels, T]</c> and the mel
    /// spectrogram <c>[1, mel, frames]</c>, where <c>T · stride = frames · hop</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T> mel)
    {
        x = _up.Forward(VocoderOps.LeakyRelu(_engine, x, _slope));
        var (kernels, biases) = _predictor.Forward(mel);
        int frames = mel.Shape[2], c = _channels, o2 = 2 * c, k = _kernelSize, perLayer = c * o2 * k;
        for (int layer = 0; layer < _dilations.Length; layer++)
        {
            var h = VocoderOps.LeakyRelu(_engine, _convs[layer].Forward(VocoderOps.LeakyRelu(_engine, x, _slope)), _slope);
            var kernel = _engine.Reshape(_engine.TensorSlice(kernels, new[] { 0, layer * perLayer, 0 }, new[] { 1, perLayer, frames }), new[] { perLayer, frames });
            var bias = _engine.Reshape(_engine.TensorSlice(biases, new[] { 0, layer * o2, 0 }, new[] { 1, o2, frames }), new[] { o2, frames });
            var output = LocationVariableConvolution(h, kernel, bias, frames);
            var gateA = _engine.TensorSlice(output, new[] { 0, 0, 0 }, new[] { 1, c, output.Shape[2] });
            var gateB = _engine.TensorSlice(output, new[] { 0, c, 0 }, new[] { 1, c, output.Shape[2] });
            x = _engine.TensorAdd(x, _engine.TensorMultiply(_engine.Sigmoid(gateA), _engine.Tanh(gateB)));
        }
        return x;
    }

    // Reference location_variable_convolution (dilation 1): output sample t of frame l = t / hop is the kernel-k
    // convolution of the zero-padded input around t with frame l's kernel [in, out, k], plus frame l's bias.
    private Tensor<T> LocationVariableConvolution(Tensor<T> x, Tensor<T> kernel, Tensor<T> bias, int frames)
    {
        int c = _channels, o2 = 2 * c, k = _kernelSize, length = x.Shape[2], pad = (k - 1) / 2;
        if (length != frames * _hop)
            throw new InvalidOperationException($"The input has {length} samples; {frames} frames of {_hop} need {frames * _hop}.");
        var padded = _engine.TensorTranspose(_engine.Reshape(VocoderOps.ZeroPad(_engine, x, pad, pad), new[] { c, length + 2 * pad }));   // [T + 2p, c]
        var index = new Tensor<int>(new[] { frames * _hop * k });
        for (int l = 0; l < frames; l++)
            for (int h = 0; h < _hop; h++)
                for (int j = 0; j < k; j++)
                    index[(l * _hop + h) * k + j] = l * _hop + h + j;
        var windows = _engine.Reshape(_engine.TensorIndexSelect(padded, index, 0), new[] { frames, _hop, k * c });             // [L, hop, k·c]
        var weights = _engine.Reshape(_engine.TensorTranspose(_engine.TensorIndexSelect(kernel, _kernelOrder, 0)), new[] { frames, k * c, o2 });
        var y = _engine.BatchMatMul(windows, weights);                                                                        // [L, hop, 2c]
        var b = _engine.TensorTile(_engine.Reshape(_engine.TensorTranspose(bias), new[] { frames, 1, o2 }), new[] { 1, _hop, 1 });
        var summed = _engine.Reshape(_engine.TensorAdd(y, b), new[] { length, o2 });
        return _engine.Reshape(_engine.TensorTranspose(summed), new[] { 1, o2, length });
    }
}

/// <summary>
/// UnivNet's generator (Jang et al. 2021, §3.1, §4.3; reference <c>Generator</c>): a reflect-padded 7-wide convolution
/// from 64 noise channels to c_G, three LVC residual stacks upsampling ×8, ×8, ×4 (dilations 1, 3, 9, 27 each), then a
/// leaky ReLU, a reflect-padded 7-wide convolution to one channel and tanh. Leaky ReLU slope 0.2, weight normalization
/// throughout.
/// </summary>
internal sealed class UnivNetGenerator<T>
{
    private readonly IEngine _engine;
    private readonly double _slope;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly NormedConv1DLayer<T> _pre;
    private readonly List<UnivNetLvcBlock<T>> _blocks = new();
    private readonly NormedConv1DLayer<T> _post;

    public UnivNetGenerator(IEngine engine, int melChannels, int noiseDim, int channels, int[] strides, int[] dilations, double slope,
        int predictorHidden, int predictorConvSize)
    {
        _engine = engine;
        _slope = slope;
        _layers.Add(_pre = new NormedConv1DLayer<T>(noiseDim, channels, 7, 1, 1, 1, 0, false, ConvolutionNormalization.Weight));
        int hop = 1;
        foreach (int s in strides)
        {
            hop *= s;
            _blocks.Add(new UnivNetLvcBlock<T>(engine, _layers, channels, melChannels, s, dilations, slope, hop, 3, predictorHidden, predictorConvSize));
        }
        _layers.Add(_post = new NormedConv1DLayer<T>(channels, 1, 7, 1, 1, 1, 0, false, ConvolutionNormalization.Weight));
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    /// <summary>A waveform <c>[1, 1, frames · Π s]</c> from a mel spectrogram and noise <c>[1, noiseDim, frames]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> mel, Tensor<T> noise)
    {
        var z = _pre.Forward(VocoderOps.ReflectPad(_engine, noise, 3, 3));
        foreach (var block in _blocks) z = block.Forward(z, mel);
        z = _post.Forward(VocoderOps.ReflectPad(_engine, VocoderOps.LeakyRelu(_engine, z, _slope), 3, 3));
        return _engine.Tanh(z);
    }
}

/// <summary>
/// UnivNet's multi-resolution spectrogram discriminator (Jang et al. 2021, §3.2; reference <c>DiscriminatorR</c>): per
/// STFT resolution, the linear magnitude <c>[bins, frames]</c> of the waveform (reflect-padded by (n_fft − hop)/2, not
/// centred, an unwindowed frame of the window length) through five weight-normalized 2-D convolutions of 32 channels
/// ((3, 9) kernels, the middle three striding 2 in time; then (3, 3)) with leaky ReLU, and a (3, 3) convolution to one
/// score map.
/// </summary>
internal sealed class MultiResolutionSpectrogramDiscriminators<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly double _slope;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly List<(int Fft, int Hop, Tensor<T> Cos, Tensor<T> Sin, List<Conv2DLayer<T>> Convs, Conv2DLayer<T> Post)> _discriminators = new();

    public MultiResolutionSpectrogramDiscriminators(IEngine engine, int[] fftSizes, int[] hopSizes, int[] windowSizes, int channels, double slope)
    {
        _engine = engine;
        _slope = slope;
        for (int r = 0; r < fftSizes.Length; r++)
        {
            int fft = fftSizes[r], win = windowSizes[r], bins = fft / 2 + 1, offset = (fft - win) / 2;
            // torch.stft without a window uses ones of the window length, zero-padded to the FFT size.
            var cos = new Tensor<T>(new[] { fft, bins });
            var sin = new Tensor<T>(new[] { fft, bins });
            for (int n = 0; n < win; n++)
                for (int k = 0; k < bins; k++)
                {
                    double a = 2 * Math.PI * (offset + n) * k / fft;
                    cos[offset + n, k] = NumOps.FromDouble(Math.Cos(a));
                    sin[offset + n, k] = NumOps.FromDouble(-Math.Sin(a));
                }
            var convs = new List<Conv2DLayer<T>>
            {
                Add(new Conv2DLayer<T>(1, channels, 3, 9, 1, 1, 1, 4, true, ConvolutionNormalization.Weight)),
                Add(new Conv2DLayer<T>(channels, channels, 3, 9, 1, 2, 1, 4, true, ConvolutionNormalization.Weight)),
                Add(new Conv2DLayer<T>(channels, channels, 3, 9, 1, 2, 1, 4, true, ConvolutionNormalization.Weight)),
                Add(new Conv2DLayer<T>(channels, channels, 3, 9, 1, 2, 1, 4, true, ConvolutionNormalization.Weight)),
                Add(new Conv2DLayer<T>(channels, channels, 3, 3, 1, 1, 1, 1, true, ConvolutionNormalization.Weight)),
            };
            _discriminators.Add((fft, hopSizes[r], cos, sin, convs, Add(new Conv2DLayer<T>(channels, 1, 3, 3, 1, 1, 1, 1, true, ConvolutionNormalization.Weight))));
        }
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    private Conv2DLayer<T> Add(Conv2DLayer<T> layer)
    {
        _layers.Add(layer);
        return layer;
    }

    /// <summary>Per resolution, the score map and the feature maps for a waveform <c>[samples]</c>.</summary>
    public List<(Tensor<T> Score, List<Tensor<T>> Features)> Forward(Tensor<T> audio)
    {
        var results = new List<(Tensor<T>, List<Tensor<T>>)>();
        int length = audio.Length;
        foreach (var (fft, hop, cos, sin, convs, post) in _discriminators)
        {
            int pad = (fft - hop) / 2, total = length + 2 * pad, frames = 1 + (total - fft) / hop;
            var index = new Tensor<int>(new[] { frames * fft });
            for (int f = 0; f < frames; f++)
                for (int n = 0; n < fft; n++)
                    index[f * fft + n] = DifferentiableMel<T>.Reflect(f * hop + n - pad, length);
            var framed = _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(audio, new[] { length, 1 }), index, 0), new[] { frames, fft });
            var re = _engine.TensorMatMul(framed, cos);
            var im = _engine.TensorMatMul(framed, sin);
            // torch.norm over (re, im); the 1e-9 keeps its gradient finite where a bin is exactly zero.
            var magnitude = _engine.TensorPow(_engine.TensorAddScalar(_engine.TensorAdd(_engine.TensorMultiply(re, re), _engine.TensorMultiply(im, im)),
                NumOps.FromDouble(1e-9)), NumOps.FromDouble(0.5));                                                               // [frames, bins]
            var h = _engine.Reshape(_engine.TensorTranspose(magnitude), new[] { 1, 1, magnitude.Shape[1], frames });
            var features = new List<Tensor<T>>();
            foreach (var conv in convs)
            {
                h = VocoderOps.LeakyRelu(_engine, conv.Forward(h), _slope);
                features.Add(h);
            }
            h = post.Forward(h);
            features.Add(h);
            results.Add((h, features));
        }
        return results;
    }
}
