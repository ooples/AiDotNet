using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.Audio.Codecs;

/// <summary>Shared construction for DAC's weight-normalized convolutions (reference <c>WNConv1d</c>).</summary>
internal static class DacOps
{
    /// <summary>A weight-normalized convolution with the reference's effective initialization: PyTorch's default
    /// kernel (U(±1/√fan_in), gain ‖V‖) and a zero bias. <c>init_weights</c> also draws the weight from a truncated normal,
    /// but under <c>torch.nn.utils.weight_norm</c> the weight is recomputed from g and V on every forward, so only the
    /// zero bias takes effect.</summary>
    public static NormedConv1DLayer<T> Conv<T>(int input, int output, int kernel, int stride, int dilation, int padding, bool transposed,
        List<LayerBase<T>> owner)
    {
        var conv = new NormedConv1DLayer<T>(input, output, kernel, stride, dilation, 1, padding, transposed, ConvolutionNormalization.Weight);
        if (!transposed) conv.ZeroBias();
        owner.Add(conv);
        return conv;
    }
}

/// <summary>
/// DAC's residual unit (reference <c>ResidualUnit</c>): Snake → a dilated k = 7 weight-normalized convolution → Snake →
/// a 1×1 convolution, added to the input.
/// </summary>
internal sealed class DacResidualUnit<T>
{
    private readonly IEngine _engine;
    private readonly SnakeLayer<T> _snake1, _snake2;
    private readonly NormedConv1DLayer<T> _conv1, _conv2;

    public DacResidualUnit(IEngine engine, int dim, int dilation, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _snake1 = new SnakeLayer<T>(dim);
        owner.Add(_snake1);
        _conv1 = DacOps.Conv<T>(dim, dim, 7, 1, dilation, 3 * dilation, false, owner);
        _snake2 = new SnakeLayer<T>(dim);
        owner.Add(_snake2);
        _conv2 = DacOps.Conv<T>(dim, dim, 1, 1, 1, 0, false, owner);
    }

    /// <summary>The unit's modules under the reference's names (<c>snake1</c>, <c>conv1</c>, <c>snake2</c>, <c>conv2</c>).</summary>
    internal IEnumerable<(string Name, LayerBase<T> Layer)> Named(string prefix)
    {
        yield return ($"{prefix}.snake1", _snake1);
        yield return ($"{prefix}.conv1", _conv1);
        yield return ($"{prefix}.snake2", _snake2);
        yield return ($"{prefix}.conv2", _conv2);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = _conv2.Forward(_snake2.Forward(_conv1.Forward(_snake1.Forward(x))));
        int pad = (x.Shape[2] - y.Shape[2]) / 2;
        if (pad > 0) x = _engine.TensorSlice(x, new[] { 0, 0, pad }, new[] { 1, x.Shape[1], y.Shape[2] });
        return _engine.TensorAdd(x, y);
    }
}

/// <summary>
/// DAC's encoder (Kumar et al. 2023, §4.3; reference <c>Encoder</c>): a k = 7 convolution to d channels, then per stride
/// (2, 4, 8, 8) three residual units (dilations 1, 3, 9) at the current width, Snake and a strided convolution (kernel 2s,
/// padding ⌈s/2⌉) doubling the width, then Snake and a k = 3 convolution to the latent dimension.
/// </summary>
internal sealed class DacEncoder<T>
{
    private readonly NormedConv1DLayer<T> _first, _last;
    private readonly List<(DacResidualUnit<T>[] Units, SnakeLayer<T> Snake, NormedConv1DLayer<T> Down)> _blocks = new();
    private readonly SnakeLayer<T> _snake;

    public DacEncoder(IEngine engine, int channels, int[] strides, int latent, List<LayerBase<T>> owner)
    {
        int d = channels;
        _first = DacOps.Conv<T>(1, d, 7, 1, 1, 3, false, owner);
        foreach (int s in strides)
        {
            var units = new[] { 1, 3, 9 }.Select(dilation => new DacResidualUnit<T>(engine, d, dilation, owner)).ToArray();
            var snake = new SnakeLayer<T>(d);
            owner.Add(snake);
            var down = DacOps.Conv<T>(d, 2 * d, 2 * s, s, 1, (s + 1) / 2, false, owner);
            _blocks.Add((units, snake, down));
            d *= 2;
        }
        _snake = new SnakeLayer<T>(d);
        owner.Add(_snake);
        _last = DacOps.Conv<T>(d, latent, 3, 1, 1, 1, false, owner);
    }

    /// <summary>The encoder's modules under the Hugging Face DAC names (<c>encoder.conv1</c>, <c>encoder.block.{i}</c>,
    /// <c>encoder.snake1</c>, <c>encoder.conv2</c>).</summary>
    internal IEnumerable<(string Name, LayerBase<T> Layer)> Named(string prefix)
    {
        yield return ($"{prefix}.conv1", _first);
        for (int i = 0; i < _blocks.Count; i++)
        {
            var (units, snake, down) = _blocks[i];
            for (int u = 0; u < units.Length; u++)
                foreach (var named in units[u].Named($"{prefix}.block.{i}.res_unit{u + 1}")) yield return named;
            yield return ($"{prefix}.block.{i}.snake1", snake);
            yield return ($"{prefix}.block.{i}.conv1", down);
        }
        yield return ($"{prefix}.snake1", _snake);
        yield return ($"{prefix}.conv2", _last);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        x = _first.Forward(x);
        foreach (var (units, snake, down) in _blocks)
        {
            foreach (var unit in units) x = unit.Forward(x);
            x = down.Forward(snake.Forward(x));
        }
        return _last.Forward(_snake.Forward(x));
    }
}

/// <summary>
/// DAC's decoder (reference <c>Decoder</c>): a k = 7 convolution to the decoder width, then per rate (8, 8, 4, 2) Snake, a
/// transposed convolution (kernel 2s, padding ⌈s/2⌉) halving the width and three residual units (dilations 1, 3, 9), then
/// Snake, a k = 7 convolution to one channel and tanh.
/// </summary>
internal sealed class DacDecoder<T>
{
    private readonly IEngine _engine;
    private readonly NormedConv1DLayer<T> _first, _last;
    private readonly List<(SnakeLayer<T> Snake, NormedConv1DLayer<T> Up, DacResidualUnit<T>[] Units)> _blocks = new();
    private readonly SnakeLayer<T> _snake;

    public DacDecoder(IEngine engine, int latent, int channels, int[] rates, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _first = DacOps.Conv<T>(latent, channels, 7, 1, 1, 3, false, owner);
        int output = channels;
        for (int i = 0; i < rates.Length; i++)
        {
            int input = channels >> i, s = rates[i];
            output = channels >> (i + 1);
            var snake = new SnakeLayer<T>(input);
            owner.Add(snake);
            var up = DacOps.Conv<T>(input, output, 2 * s, s, 1, (s + 1) / 2, true, owner);
            var units = new[] { 1, 3, 9 }.Select(dilation => new DacResidualUnit<T>(engine, output, dilation, owner)).ToArray();
            _blocks.Add((snake, up, units));
        }
        _snake = new SnakeLayer<T>(output);
        owner.Add(_snake);
        _last = DacOps.Conv<T>(output, 1, 7, 1, 1, 3, false, owner);
    }

    /// <summary>The decoder's modules under the Hugging Face DAC names (<c>decoder.conv1</c>, <c>decoder.block.{i}</c>,
    /// <c>decoder.snake1</c>, <c>decoder.conv2</c>).</summary>
    internal IEnumerable<(string Name, LayerBase<T> Layer)> Named(string prefix)
    {
        yield return ($"{prefix}.conv1", _first);
        for (int i = 0; i < _blocks.Count; i++)
        {
            var (snake, up, units) = _blocks[i];
            yield return ($"{prefix}.block.{i}.snake1", snake);
            yield return ($"{prefix}.block.{i}.conv_t1", up);
            for (int u = 0; u < units.Length; u++)
                foreach (var named in units[u].Named($"{prefix}.block.{i}.res_unit{u + 1}")) yield return named;
        }
        yield return ($"{prefix}.snake1", _snake);
        yield return ($"{prefix}.conv2", _last);
    }

    public Tensor<T> Forward(Tensor<T> z)
    {
        var x = _first.Forward(z);
        foreach (var (snake, up, units) in _blocks)
        {
            x = up.Forward(snake.Forward(x));
            foreach (var unit in units) x = unit.Forward(x);
        }
        return _engine.Tanh(_last.Forward(_snake.Forward(x)));
    }
}

/// <summary>
/// DAC's multi-band, multi-scale complex STFT discriminator for one window (§3.4; reference <c>MRD</c>): the STFT with the
/// reference's stride-matched framing (reflect padding of (w − h)/2 plus the samples that round the length up to whole
/// hops, a centred Hann STFT, the two frames at each end dropped), its real and imaginary parts as two channels laid out
/// [time, frequency], split into the bands [0, .1, .25, .5, .75, 1] of the bins; each band through its own stack — a 3×9
/// convolution to 32 channels, three 3×9 convolutions with frequency stride 2 and a 3×3 convolution, LeakyReLU (0.1) after
/// each — and the bands, concatenated along frequency, through a 3×3 convolution to the logits.
/// </summary>
internal sealed class DacBandDiscriminator<T>
{
    private const double Slope = 0.1;
    private readonly IEngine _engine;
    private readonly int _window, _hop;
    private readonly AiDotNet.TextToSpeech.Vocoders.CenteredComplexStft<T> _stft;
    private readonly (int From, int To)[] _bands;
    private readonly List<List<Conv2DLayer<T>>> _stacks = new();
    private readonly Conv2DLayer<T> _post;

    public DacBandDiscriminator(IEngine engine, int window, double[] bandEdges, int channels, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _window = window;
        _hop = window / 4;
        _stft = new AiDotNet.TextToSpeech.Vocoders.CenteredComplexStft<T>(engine, window, _hop, window);
        int bins = window / 2 + 1;
        _bands = Enumerable.Range(0, bandEdges.Length - 1).Select(i => ((int)(bandEdges[i] * bins), (int)(bandEdges[i + 1] * bins))).ToArray();
        Conv2DLayer<T> Add(Conv2DLayer<T> layer)
        {
            owner.Add(layer);
            return layer;
        }
        foreach (var _ in _bands)
        {
            _stacks.Add(new List<Conv2DLayer<T>>
            {
                Add(new Conv2DLayer<T>(2, channels, 3, 9, 1, 1, 1, 4, true, ConvolutionNormalization.Weight)),
                Add(new Conv2DLayer<T>(channels, channels, 3, 9, 1, 2, 1, 4, true, ConvolutionNormalization.Weight)),
                Add(new Conv2DLayer<T>(channels, channels, 3, 9, 1, 2, 1, 4, true, ConvolutionNormalization.Weight)),
                Add(new Conv2DLayer<T>(channels, channels, 3, 9, 1, 2, 1, 4, true, ConvolutionNormalization.Weight)),
                Add(new Conv2DLayer<T>(channels, channels, 3, 3, 1, 1, 1, 1, true, ConvolutionNormalization.Weight)),
            });
        }
        _post = Add(new Conv2DLayer<T>(channels, 1, 3, 3, 1, 1, 1, 1, true, ConvolutionNormalization.Weight));
    }

    /// <summary>The logits and every layer's output (the logits last) for a waveform <c>[samples]</c>.</summary>
    public (Tensor<T> Logits, List<Tensor<T>> Features) Forward(Tensor<T> audio)
    {
        int length = audio.Length;
        int rightPad = (length + _hop - 1) / _hop * _hop - length, pad = (_window - _hop) / 2;
        var padded = Seanet.Pad(_engine, _engine.Reshape(audio, new[] { 1, 1, length }), pad, pad + rightPad, reflect: true);
        var (re, im) = _stft.Forward(_engine.Reshape(padded, new[] { padded.Length }));         // [1, bins, frames], centred
        int frames = re.Shape[2] - 4;
        Tensor<T> Frames(Tensor<T> x) => _engine.TensorTranspose(_engine.Reshape(_engine.TensorSlice(x, new[] { 0, 0, 2 }, new[] { 1, x.Shape[1], frames }),
            new[] { x.Shape[1], frames }));                                                          // [frames, bins]
        var spectrum = _engine.Reshape(_engine.TensorConcatenate(new[] { Frames(re), Frames(im) }, 0), new[] { 1, 2, frames, re.Shape[1] });
        var features = new List<Tensor<T>>();
        var outputs = new List<Tensor<T>>();
        for (int b = 0; b < _bands.Length; b++)
        {
            var (from, to) = _bands[b];
            var band = _engine.TensorSlice(spectrum, new[] { 0, 0, 0, from }, new[] { 1, 2, frames, to - from });
            foreach (var conv in _stacks[b])
            {
                band = VocoderOps.LeakyRelu(_engine, conv.Forward(band), Slope);
                features.Add(band);
            }
            outputs.Add(band);
        }
        var logits = _post.Forward(_engine.TensorConcatenate(outputs.ToArray(), 3));
        features.Add(logits);
        return (logits, features);
    }
}
