using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Codecs;

/// <summary>
/// A short-time Fourier transform on the autodiff tape with torch.stft's <c>center=False</c> framing (frames start at
/// sample 0, 1 + (L − n_fft) / hop of them), a periodic Hann window, and optional window normalization (torchaudio
/// <c>normalized=True</c>: divided by √Σw²).
/// </summary>
internal sealed class FramedComplexStft<T>
{
    private readonly IEngine _engine;
    private readonly int _fft, _hop;
    private readonly Tensor<T> _cos, _sin;

    public FramedComplexStft(IEngine engine, int fftSize, int hopSize, int windowSize, bool normalized)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        _engine = engine;
        _fft = fftSize;
        _hop = hopSize;
        int bins = fftSize / 2 + 1, offset = (fftSize - windowSize) / 2;
        var window = new double[fftSize];
        double energy = 0;
        for (int n = 0; n < windowSize; n++)
        {
            window[offset + n] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / windowSize);
            energy += window[offset + n] * window[offset + n];
        }
        double scale = normalized ? 1.0 / Math.Sqrt(energy) : 1.0;
        _cos = new Tensor<T>(new[] { fftSize, bins });
        _sin = new Tensor<T>(new[] { fftSize, bins });
        for (int n = 0; n < fftSize; n++)
            for (int k = 0; k < bins; k++)
            {
                double a = 2 * Math.PI * n * k / fftSize;
                _cos[n, k] = ops.FromDouble(scale * window[n] * Math.Cos(a));
                _sin[n, k] = ops.FromDouble(-scale * window[n] * Math.Sin(a));
            }
    }

    public int Bins => _fft / 2 + 1;

    /// <summary>The real and imaginary parts <c>[frames, bins]</c> of <paramref name="audio"/> <c>[samples]</c>.</summary>
    public (Tensor<T> Re, Tensor<T> Im) Forward(Tensor<T> audio)
    {
        int length = audio.Length;
        if (length < _fft) throw new ArgumentException($"The STFT needs at least {_fft} samples, got {length}.", nameof(audio));
        int frames = 1 + (length - _fft) / _hop;
        var index = new Tensor<int>(new[] { frames * _fft });
        for (int f = 0; f < frames; f++)
            for (int n = 0; n < _fft; n++) index[f * _fft + n] = f * _hop + n;
        var framed = _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(audio, new[] { length, 1 }), index, 0), new[] { frames, _fft });
        return (_engine.TensorMatMul(framed, _cos), _engine.TensorMatMul(framed, _sin));
    }
}

/// <summary>
/// Signal helpers shared by the codecs: SEANet-style padding and torchaudio's HTK mel filterbank.
/// </summary>
internal static class CodecSignal
{
    /// <summary>torchaudio <c>melscale_fbanks</c> with <c>mel_scale="htk"</c> and <c>norm=None</c>: <c>[bins, mels]</c>.</summary>
    public static double[,] HtkMelFilterbank(int bins, double fMin, double fMax, int mels, int sampleRate)
    {
        static double ToMel(double f) => 2595.0 * Math.Log10(1.0 + f / 700.0);
        static double ToHz(double m) => 700.0 * (Math.Pow(10.0, m / 2595.0) - 1.0);
        double mMin = ToMel(fMin), mMax = ToMel(fMax);
        var points = new double[mels + 2];
        for (int i = 0; i < mels + 2; i++) points[i] = ToHz(mMin + (mMax - mMin) * i / (mels + 1));
        var fb = new double[bins, mels];
        for (int k = 0; k < bins; k++)
        {
            double f = bins == 1 ? 0 : (sampleRate / 2) * (double)k / (bins - 1);
            for (int m = 0; m < mels; m++)
            {
                double down = (f - points[m]) / (points[m + 1] - points[m]);
                double up = (points[m + 2] - f) / (points[m + 2] - points[m + 1]);
                fb[k, m] = Math.Max(0, Math.Min(down, up));
            }
        }
        return fb;
    }
}

/// <summary>
/// One mel scale of EnCodec's frequency-domain loss (§3.4, Eq. 1): a 64-bin mel spectrogram from a normalized STFT with
/// window 2^i and hop 2^i / 4. The paper leaves the rest to the reference: the authors' training code (audiocraft
/// <c>MelSpectrogramWrapper</c>) reflect-pads (n_fft − hop) / 2 on each side, zero-pads the end to whole frames, frames
/// without centring, and takes the power spectrum through torchaudio's HTK filterbank.
/// </summary>
internal sealed class CodecMelScale<T>
{
    private readonly IEngine _engine;
    private readonly int _fft, _hop;
    private readonly FramedComplexStft<T> _stft;
    private readonly Tensor<T> _filterbank;

    public CodecMelScale(IEngine engine, int sampleRate, int fftSize, int hopSize, int mels, double fMin, double? fMax, bool normalized)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        _engine = engine;
        _fft = fftSize;
        _hop = hopSize;
        _stft = new FramedComplexStft<T>(engine, fftSize, hopSize, fftSize, normalized);
        var fb = CodecSignal.HtkMelFilterbank(_stft.Bins, fMin, fMax ?? sampleRate / 2.0, mels, sampleRate);
        _filterbank = new Tensor<T>(new[] { _stft.Bins, mels });
        for (int k = 0; k < _stft.Bins; k++)
            for (int m = 0; m < mels; m++) _filterbank[k, m] = ops.FromDouble(fb[k, m]);
    }

    /// <summary>The mel power spectrogram <c>[frames, mels]</c> of <paramref name="audio"/> <c>[samples]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> audio)
    {
        int pad = (_fft - _hop) / 2;
        var x = Seanet.Pad(_engine, _engine.Reshape(audio, new[] { 1, 1, audio.Length }), pad, pad, reflect: true);
        int extra = Seanet.ExtraPadding(x.Shape[2], _fft, _hop, 0);
        if (extra > 0) x = Seanet.Pad(_engine, x, 0, extra, reflect: false);
        var (re, im) = _stft.Forward(_engine.Reshape(x, new[] { x.Length }));
        var power = _engine.TensorAdd(_engine.TensorMultiply(re, re), _engine.TensorMultiply(im, im));
        return _engine.TensorMatMul(power, _filterbank);
    }
}

/// <summary>
/// One sub-discriminator of EnCodec's multi-scale STFT discriminator (§3.4, Fig. 2; reference <c>msstftd.py</c>
/// <c>DiscriminatorSTFT</c>): the normalized complex STFT (window w, hop w / 4, no centring) with its real and imaginary
/// parts as two channels laid out [time, frequency]; a k = 3×9 convolution to 32 channels; three 3×9 convolutions with
/// stride (1, 2) and time dilations 1, 2, 4; a 3×3 convolution; LeakyReLU after each; and a 3×3 convolution to the logit.
/// </summary>
/// <remarks>Every convolution is weight-normalized: the paper applies weight normalization to the discriminator network
/// (§3.4); the reference leaves the first convolution unnormalized. The kernel is 3×9 as in the paper's Fig. 2 and the
/// reference (the text's "3 × 8" disagrees with both). The LeakyReLU slope (0.2) is the reference's.</remarks>
internal sealed class StftSubDiscriminator<T>
{
    private readonly IEngine _engine;
    private readonly FramedComplexStft<T> _stft;
    private readonly List<Conv2DLayer<T>> _convs = new();
    private readonly Conv2DLayer<T> _post;
    private readonly double _slope;

    public StftSubDiscriminator(IEngine engine, int window, int hop, int filters, int[] dilations, int kernelTime, int kernelFrequency,
        double slope, ConvolutionNormalization normalization, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _slope = slope;
        _stft = new FramedComplexStft<T>(engine, window, hop, window, normalized: true);
        int padTime = (kernelTime - 1) / 2, padFrequency = (kernelFrequency - 1) / 2;
        Conv2DLayer<T> Add(Conv2DLayer<T> layer)
        {
            owner.Add(layer);
            return layer;
        }
        _convs.Add(Add(new Conv2DLayer<T>(2, filters, kernelTime, kernelFrequency, 1, 1, padTime, padFrequency, true, normalization)));
        foreach (int d in dilations)
            _convs.Add(Add(new Conv2DLayer<T>(filters, filters, kernelTime, kernelFrequency, 1, 2, padTime * d, padFrequency, true, normalization,
                dilationHeight: d)));
        _convs.Add(Add(new Conv2DLayer<T>(filters, filters, kernelTime, kernelTime, 1, 1, padTime, padTime, true, normalization)));
        _post = Add(new Conv2DLayer<T>(filters, 1, kernelTime, kernelTime, 1, 1, padTime, padTime, true, normalization));
    }

    /// <summary>The logits and the feature maps (after each activation) for a waveform <c>[samples]</c>.</summary>
    public (Tensor<T> Logits, List<Tensor<T>> Features) Forward(Tensor<T> audio)
    {
        var (re, im) = _stft.Forward(audio);                                                           // [frames, bins]
        int frames = re.Shape[0], bins = re.Shape[1];
        var z = _engine.Reshape(_engine.TensorConcatenate(new[] { re, im }, 0), new[] { 1, 2, frames, bins });
        var features = new List<Tensor<T>>();
        foreach (var conv in _convs)
        {
            z = AiDotNet.TextToSpeech.Vocoders.VocoderOps.LeakyRelu(_engine, conv.Forward(z), _slope);
            features.Add(z);
        }
        return (_post.Forward(z), features);
    }
}

/// <summary>
/// The multi-scale STFT discriminator (EnCodec §3.4): identical sub-discriminators over STFT windows
/// [2048, 1024, 512, 256, 128] (hop w / 4).
/// </summary>
internal sealed class MultiScaleStftDiscriminator<T>
{
    private readonly List<StftSubDiscriminator<T>> _discriminators = new();

    public MultiScaleStftDiscriminator(IEngine engine, int[] windows, int filters, int[] dilations, int kernelTime, int kernelFrequency,
        double slope, List<LayerBase<T>> owner)
    {
        foreach (int w in windows)
            _discriminators.Add(new StftSubDiscriminator<T>(engine, w, w / 4, filters, dilations, kernelTime, kernelFrequency, slope,
                ConvolutionNormalization.Weight, owner));
    }

    public List<(Tensor<T> Logits, List<Tensor<T>> Features)> Forward(Tensor<T> audio)
        => _discriminators.Select(d => d.Forward(audio)).ToList();
}

/// <summary>
/// SoundStream's STFT-based discriminator (Zeghidour et al. 2021, §III-D, Fig. 4): the complex STFT (window W = 1024,
/// hop 256, F = W / 2 bins) as real and imaginary channels laid out [time, frequency], a 7×7 convolution to C channels, six
/// residual units ResidualUnit(N, m, (s_t, s_f)) — a 3×3 convolution then a (s_t + 2)×(s_f + 2) convolution to mN
/// channels with stride (s_t, s_f) — with (N, m, s) = (C, 2, (1, 2)), (2C, 2, (2, 2)), (4C, 1, (1, 2)), (4C, 2, (2, 2)),
/// (8C, 1, (1, 2)), (8C, 2, (2, 2)), and a 1 × F/2⁶ convolution that aggregates the frequency bins into one logit per
/// (down-sampled) frame.
/// </summary>
/// <remarks>The paper leaves the residual shortcut and the activation unstated: the shortcut is a 1×1 convolution with
/// the unit's stride (the only shape-preserving choice for a strided unit) and the activation LeakyReLU (0.2), as the
/// wave discriminator's. The STFT is not centred; F drops the Nyquist bin so the frequency axis halves exactly six
/// times.</remarks>
internal sealed class SoundStreamStftDiscriminator<T>
{
    private const double Slope = 0.2;
    private readonly IEngine _engine;
    private readonly FramedComplexStft<T> _stft;
    private readonly int _bins;
    private readonly Conv2DLayer<T> _first;
    private readonly List<(Conv2DLayer<T> Conv, Conv2DLayer<T> Strided, Conv2DLayer<T> Shortcut)> _units = new();
    private readonly Conv2DLayer<T> _post;

    public SoundStreamStftDiscriminator(IEngine engine, int window, int hop, int channels, List<LayerBase<T>> owner)
    {
        if (window % 128 != 0) throw new ArgumentException("The STFT window must be a multiple of 128 (F = W/2 halves six times).", nameof(window));
        _engine = engine;
        _stft = new FramedComplexStft<T>(engine, window, hop, window, normalized: false);
        _bins = window / 2;
        Conv2DLayer<T> Add(Conv2DLayer<T> layer)
        {
            owner.Add(layer);
            return layer;
        }
        _first = Add(new Conv2DLayer<T>(2, channels, 7, 7, 1, 1, 3, 3));
        var spec = new (int N, int M, int St, int Sf)[]
        {
            (channels, 2, 1, 2), (2 * channels, 2, 2, 2), (4 * channels, 1, 1, 2),
            (4 * channels, 2, 2, 2), (8 * channels, 1, 1, 2), (8 * channels, 2, 2, 2),
        };
        foreach (var (n, m, st, sf) in spec)
        {
            var conv = Add(new Conv2DLayer<T>(n, n, 3, 3, 1, 1, 1, 1));
            var strided = Add(new Conv2DLayer<T>(n, m * n, st + 2, sf + 2, st, sf, 1, 1));
            var shortcut = Add(new Conv2DLayer<T>(n, m * n, 1, 1, st, sf, 0, 0));
            _units.Add((conv, strided, shortcut));
        }
        _post = Add(new Conv2DLayer<T>(16 * channels, 1, 1, _bins >> 6, 1, 1, 0, 0));
    }

    /// <summary>The logits <c>[1, 1, frames', 1]</c> and the residual units' outputs for a waveform <c>[samples]</c>.</summary>
    public (Tensor<T> Logits, List<Tensor<T>> Features) Forward(Tensor<T> audio)
    {
        var (re, im) = _stft.Forward(audio);                                                           // [frames, bins + 1]
        int frames = re.Shape[0];
        re = _engine.TensorSlice(re, new[] { 0, 0 }, new[] { frames, _bins });
        im = _engine.TensorSlice(im, new[] { 0, 0 }, new[] { frames, _bins });
        var z = _engine.Reshape(_engine.TensorConcatenate(new[] { re, im }, 0), new[] { 1, 2, frames, _bins });
        z = AiDotNet.TextToSpeech.Vocoders.VocoderOps.LeakyRelu(_engine, _first.Forward(z), Slope);
        var features = new List<Tensor<T>> { z };
        foreach (var (conv, strided, shortcut) in _units)
        {
            var y = AiDotNet.TextToSpeech.Vocoders.VocoderOps.LeakyRelu(_engine, conv.Forward(z), Slope);
            y = strided.Forward(y);
            var skip = shortcut.Forward(z);
            int t = Math.Min(y.Shape[2], skip.Shape[2]), f = Math.Min(y.Shape[3], skip.Shape[3]);
            y = _engine.TensorSlice(y, new[] { 0, 0, 0, 0 }, new[] { 1, y.Shape[1], t, f });
            skip = _engine.TensorSlice(skip, new[] { 0, 0, 0, 0 }, new[] { 1, skip.Shape[1], t, f });
            z = AiDotNet.TextToSpeech.Vocoders.VocoderOps.LeakyRelu(_engine, _engine.TensorAdd(y, skip), Slope);
            features.Add(z);
        }
        return (_post.Forward(z), features);
    }
}
