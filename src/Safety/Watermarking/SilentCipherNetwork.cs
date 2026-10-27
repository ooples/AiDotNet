using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.StyleEmotion;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// The shape of SilentCipher (Singh et al. 2024, "SilentCipher: Deep Audio Watermarking", Interspeech; reference
/// sony/silentcipher at d46d7d0). The defaults are the released 44.1 kHz model (<c>sony/silentcipher</c>
/// <c>44_1_khz/73999_iteration</c>, its <c>hparams.yaml</c>), the one Sesame's CSM watermarks with.
/// </summary>
internal sealed class SilentCipherConfiguration
{
    /// <summary>The model's sampling rate (44100).</summary>
    public int SampleRate { get; init; } = 44100;

    /// <summary>The FFT and window length (4096).</summary>
    public int Fft { get; init; } = 4096;

    /// <summary>The hop (2048).</summary>
    public int Hop { get; init; } = 2048;

    /// <summary>Symbols per message, the end marker included (21: twenty 2-bit symbols of a 5-byte key, then 0).</summary>
    public int MessageLength { get; init; } = 21;

    /// <summary>The one-hot symbol width (5: the end marker and four 2-bit values).</summary>
    public int MessageDim { get; init; } = 5;

    /// <summary>The low frequency bins that carry the message (1024).</summary>
    public int MessageBandSize { get; init; } = 1024;

    /// <summary>The carrier encoder's gated layers (3).</summary>
    public int EncoderLayers { get; init; } = 3;

    /// <summary>The carrier decoder's gated layers (4).</summary>
    public int CarrierDecoderLayers { get; init; } = 4;

    /// <summary>The message decoder's gated layers (10) and width (128).</summary>
    public int MessageDecoderLayers { get; init; } = 10;

    /// <inheritdoc cref="MessageDecoderLayers"/>
    public int MessageDecoderChannels { get; init; } = 128;

    /// <summary>The power every input is scaled to before the transform and back from after it (VCTK's average,
    /// 0.002837200844477648).</summary>
    public double AverageEnergy { get; init; } = 0.002837200844477648;

    /// <summary>The message's default level below the carrier in dB (47; CSM asks for 36).</summary>
    public double DefaultMessageSdr { get; init; } = 47;

    /// <summary>The bytes a message carries (5).</summary>
    public int KeyBytes => (MessageLength - 1) * 2 / 8;

    public int Bins => Fft / 2 + 1;
}

/// <summary>
/// SilentCipher's gated layer (reference <c>model.Layer</c>): BatchNorm(conv(x) ⊙ σ(gate(x))) with two same-shaped 2-D
/// convolutions over <c>[channels, frequency, time]</c>.
/// </summary>
internal sealed class SilentCipherGatedLayer<T>
{
    private readonly IEngine _engine;

    public SilentCipherGatedLayer(IEngine engine, int input, int output, int kernel, List<LayerBase<T>> owner)
    {
        _engine = engine;
        int padding = kernel / 2;
        owner.Add(Conv = new NormedConv2DLayer<T>(input, output, kernel, kernel, 1, 1, padding, padding, 1, ConvolutionNormalization.None));
        owner.Add(Gate = new NormedConv2DLayer<T>(input, output, kernel, kernel, 1, 1, padding, padding, 1, ConvolutionNormalization.None));
        owner.Add(Norm = new BatchNormalizationLayer<T>(output, 1e-5, 0.9));
    }

    public NormedConv2DLayer<T> Conv { get; }
    public NormedConv2DLayer<T> Gate { get; }
    public BatchNormalizationLayer<T> Norm { get; }

    public Tensor<T> Forward(Tensor<T> x) => Norm.Forward(_engine.TensorMultiply(Conv.Forward(x), _engine.Sigmoid(Gate.Forward(x))));

    public void Load(Func<string, double[]?> read, string prefix, Func<string, int, double[]> need)
    {
        var shape = Conv.TorchWeightShape;
        int count = shape.Aggregate(1, (a, b) => a * b), output = shape[0];
        Conv.LoadTorchWeights(need($"{prefix}.conv.weight", count), need($"{prefix}.conv.bias", output));
        Gate.LoadTorchWeights(need($"{prefix}.gate.weight", count), need($"{prefix}.gate.bias", output));
        TorchParameters.BatchNorm(Norm, need($"{prefix}.bn.weight", output), need($"{prefix}.bn.bias", output),
            need($"{prefix}.bn.running_mean", output), need($"{prefix}.bn.running_var", output));
        read($"{prefix}.bn.num_batches_tracked");
    }
}

/// <summary>
/// SilentCipher (Singh et al. 2024; reference sony/silentcipher <c>server.Model.encode_wav</c> and <c>decode_wav</c>),
/// as Sesame's CSM applies it. Embedding: the audio (at <see cref="SilentCipherConfiguration.SampleRate"/>) is scaled to
/// VCTK's average power, right-padded with zeros by a window less its length modulo the window, and transformed (centred
/// STFT, periodic Hann window); the carrier encoder reads the magnitude, the message — 2-bit symbols of the key's bits
/// with an end marker, one-hot, tiled over the frames — is mapped to the message band by a linear layer, and the carrier
/// decoder reads both with 32 copies of the magnitude. Its output's absolute value, zero above the message band, is
/// normalized per frame to the message level below the utterance's RMS magnitude, subtracted from the magnitude
/// (clipped at zero), and the result is inverted with the original phase, trimmed and scaled back. Decoding finds the
/// message decoder's per-frame symbols, takes the most common symbol at each of the message's positions, and rotates the
/// sequence to start after the end marker; with phase-shift decoding it first picks the offset (in steps of 10 samples
/// below the hop) whose frames agree best.
/// </summary>
internal sealed class SilentCipherNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers;
    private readonly SilentCipherGatedLayer<T>[] _encoder, _carrierDecoder, _messageDecoder;
    private readonly DenseLayer<T> _messageMap, _messageReadout;
    private readonly CenteredComplexStft<T> _stft;
    private readonly InverseStft<T> _istft;
    private const int EncoderWidth = 32, CarrierDecoderWidth = 96;

    public SilentCipherNetwork(IEngine engine, SilentCipherConfiguration configuration, List<LayerBase<T>> owner)
    {
        _engine = engine;
        Configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        var c = configuration;
        if (c.EncoderLayers < 2 || c.CarrierDecoderLayers < 2 || c.MessageDecoderLayers < 2)
            throw new ArgumentException("SilentCipher's networks have at least two layers each.", nameof(configuration));
        if ((c.MessageLength - 1) * 2 % 8 != 0)
            throw new ArgumentException("A message's symbols must make whole bytes.", nameof(configuration));
        _layers = owner;
        _encoder = new SilentCipherGatedLayer<T>[c.EncoderLayers];
        for (int i = 0; i < c.EncoderLayers; i++)
            _encoder[i] = new SilentCipherGatedLayer<T>(engine, i == 0 ? 1 : EncoderWidth, EncoderWidth, 3, owner);
        owner.Add(_messageMap = new DenseLayer<T>(c.MessageBandSize, (IActivationFunction<T>)new IdentityActivation<T>()));
        _carrierDecoder = new SilentCipherGatedLayer<T>[c.CarrierDecoderLayers];
        for (int i = 0; i < c.CarrierDecoderLayers; i++)
        {
            bool last = i == c.CarrierDecoderLayers - 1;
            _carrierDecoder[i] = new SilentCipherGatedLayer<T>(engine, i == 0 ? 3 * EncoderWidth : CarrierDecoderWidth,
                last ? 1 : CarrierDecoderWidth, last ? 1 : 3, owner);
        }
        _messageDecoder = new SilentCipherGatedLayer<T>[c.MessageDecoderLayers];
        for (int i = 0; i < c.MessageDecoderLayers; i++)
            _messageDecoder[i] = new SilentCipherGatedLayer<T>(engine, i == 0 ? 1 : c.MessageDecoderChannels,
                i == c.MessageDecoderLayers - 1 ? c.MessageDim : c.MessageDecoderChannels, 3, owner);
        owner.Add(_messageReadout = new DenseLayer<T>(1, (IActivationFunction<T>)new IdentityActivation<T>()));
        _stft = new CenteredComplexStft<T>(engine, c.Fft, c.Hop, c.Fft);
        _istft = new InverseStft<T>(engine, c.Fft, c.Hop, c.Fft);
    }

    public SilentCipherConfiguration Configuration { get; }

    private void Eval()
    {
        foreach (var layer in _layers) layer.SetTrainingMode(false);
    }

    private static Tensor<T> FromArray(double[] values, int start = 0)
    {
        var t = new Tensor<T>(new[] { values.Length - start });
        for (int i = start; i < values.Length; i++) t[i - start] = NumOps.FromDouble(values[i]);
        return t;
    }

    // stft.transform: pad right by win − len mod win, then the centred STFT; |X| (0 where X is exactly 0) and atan2.
    private (double[,] Magnitude, double[,] Phase) Transform(double[] y)
    {
        int win = Configuration.Fft, pad = win - y.Length % win;
        var padded = new double[y.Length + pad];
        Array.Copy(y, padded, y.Length);
        var (re, im) = _stft.Forward(FromArray(padded));
        int bins = re.Shape[1], frames = re.Shape[2];
        var magnitude = new double[bins, frames];
        var phase = new double[bins, frames];
        for (int k = 0; k < bins; k++)
            for (int f = 0; f < frames; f++)
            {
                double r = NumOps.ToDouble(re[0, k, f]), i = NumOps.ToDouble(im[0, k, f]);
                magnitude[k, f] = Math.Sqrt(r * r + i * i);
                phase[k, f] = Math.Atan2(i, r);
            }
        return (magnitude, phase);
    }

    private static Tensor<T> Image(double[,] values)
    {
        int h = values.GetLength(0), w = values.GetLength(1);
        var t = new Tensor<T>(new[] { 1, 1, h, w });
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++) t[0, 0, i, j] = NumOps.FromDouble(values[i, j]);
        return t;
    }

    /// <summary>The key's 2-bit symbols (each byte's bits, most significant first, in pairs).</summary>
    internal static int[] Symbols(IReadOnlyList<int> key)
    {
        var symbols = new int[key.Count * 4];
        for (int b = 0; b < key.Count; b++)
        {
            if (key[b] < 0 || key[b] > 255) throw new ArgumentOutOfRangeException(nameof(key), key[b], "Key values are bytes.");
            for (int s = 0; s < 4; s++) symbols[4 * b + s] = (key[b] >> (6 - 2 * s)) & 3;
        }
        return symbols;
    }

    /// <summary>
    /// The watermarked waveform of <paramref name="audio"/> (at <see cref="SilentCipherConfiguration.SampleRate"/>)
    /// carrying <paramref name="key"/> (<see cref="SilentCipherConfiguration.KeyBytes"/> bytes) at
    /// <paramref name="messageSdr"/> dB below it. Silent audio is returned unchanged.
    /// </summary>
    public double[] Encode(double[] audio, IReadOnlyList<int> key, double? messageSdr = null)
    {
        if (audio is null || audio.Length == 0) throw new ArgumentException("There is no audio.", nameof(audio));
        if (key is null || key.Count != Configuration.KeyBytes)
            throw new ArgumentException($"A key has {Configuration.KeyBytes} bytes.", nameof(key));
        var c = Configuration;
        double sdr = messageSdr ?? c.DefaultMessageSdr;
        double power = audio.Sum(v => v * v) / audio.Length;
        if (power == 0) return (double[])audio.Clone();
        double gain = Math.Sqrt(c.AverageEnergy / power);
        var y = audio.Select(v => v * gain).ToArray();
        using var _ = new NoGradScope<T>();
        Eval();
        var (magnitude, phase) = Transform(y);
        int bins = magnitude.GetLength(0), frames = magnitude.GetLength(1);
        var carrier = Image(magnitude);

        var h = carrier;
        foreach (var layer in _encoder) h = layer.Forward(h);                                                   // [1, 32, bins, frames]

        // letters_encoding: symbols + 1 then the end marker 0, one-hot, tiled over the frames (the last tile partial).
        var symbols = Symbols(key).Select(s => s + 1).Concat(new[] { 0 }).ToArray();
        var message = new Tensor<T>(new[] { frames, c.MessageDim });
        for (int f = 0; f < frames; f++) message[f, symbols[f % c.MessageLength]] = NumOps.One;
        var band = _messageMap.Forward(message);                                                                 // [frames, band]
        var messageImage = new Tensor<T>(new[] { 1, 1, bins, frames });
        for (int k = 0; k < c.MessageBandSize; k++)
            for (int f = 0; f < frames; f++) messageImage[0, 0, k, f] = band[f, k];

        var merged = _engine.TensorConcatenate(new[]
        {
            h,
            _engine.TensorTile(carrier, new[] { 1, EncoderWidth, 1, 1 }),
            _engine.TensorTile(messageImage, new[] { 1, EncoderWidth, 1, 1 }),
        }, 1);
        var info = merged;
        foreach (var layer in _carrierDecoder) info = layer.Forward(info);                                      // [1, 1, bins, frames]

        // CarrierDecoder: |h|, zero above the message band, normalized per frame to 10^(−sdr/20); then the utterance's
        // RMS magnitude; the negated message is added to the carrier and clipped at zero.
        double level = Math.Pow(10, sdr / 20), carrierSquares = 0;
        for (int k = 0; k < bins; k++)
            for (int f = 0; f < frames; f++) carrierSquares += magnitude[k, f] * magnitude[k, f];
        double utterance = Math.Sqrt(carrierSquares / (bins * frames));
        var reconstructed = new Tensor<T>(new[] { 1, bins, frames });
        var phaseTensor = new Tensor<T>(new[] { 1, bins, frames });
        for (int f = 0; f < frames; f++)
        {
            double squares = 0;
            for (int k = 0; k < c.MessageBandSize && k < bins; k++)
            {
                double v = NumOps.ToDouble(info[0, 0, k, f]);
                squares += v * v;
            }
            double rms = Math.Sqrt(squares / bins);
            for (int k = 0; k < bins; k++)
            {
                double m = k < c.MessageBandSize ? Math.Abs(NumOps.ToDouble(info[0, 0, k, f])) / rms / level * utterance : 0.0;
                reconstructed[0, k, f] = NumOps.FromDouble(Math.Max(0.0, magnitude[k, f] - m));
                phaseTensor[0, k, f] = NumOps.FromDouble(phase[k, f]);
            }
        }
        var signal = _istft.Forward(reconstructed, phaseTensor);
        var result = new double[audio.Length];
        double back = Math.Sqrt(power / c.AverageEnergy);
        for (int i = 0; i < result.Length; i++) result[i] = NumOps.ToDouble(signal[i]) * back;
        return result;
    }

    // MsgDecoder: the gated layers over the message band, then a linear readout over frequency: [dim, frames].
    private double[,] MessageScores(double[,] magnitude)
    {
        int frames = magnitude.GetLength(1), band = Configuration.MessageBandSize;
        var input = new Tensor<T>(new[] { 1, 1, band, frames });
        for (int k = 0; k < band; k++)
            for (int f = 0; f < frames; f++) input[0, 0, k, f] = NumOps.FromDouble(magnitude[k, f]);
        var h = input;
        foreach (var layer in _messageDecoder) h = layer.Forward(h);                                            // [1, dim, band, frames]
        int dim = Configuration.MessageDim;
        var rows = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(h, new[] { dim, band, frames }), new[] { 0, 2, 1 }), new[] { dim * frames, band });
        var scores = _messageReadout.Forward(rows);
        var result = new double[dim, frames];
        for (int d = 0; d < dim; d++)
            for (int f = 0; f < frames; f++) result[d, f] = NumOps.ToDouble(scores[d * frames + f, 0]);
        return result;
    }

    /// <summary>The message decoder's scores <c>[symbol dim, frames]</c> on <paramref name="audio"/> scaled to the
    /// reference power, with no phase shift.</summary>
    internal double[,] Scores(double[] audio)
    {
        double power = audio.Sum(v => v * v) / audio.Length, gain = Math.Sqrt(Configuration.AverageEnergy / power);
        using var _ = new NoGradScope<T>();
        Eval();
        return MessageScores(Transform(audio.Select(v => v * gain).ToArray()).Magnitude);
    }

    // The argmax symbol of each whole message's frames: [repeats, message length].
    private int[,] Predictions(double[,] scores)
    {
        int dim = scores.GetLength(0), frames = scores.GetLength(1), length = Configuration.MessageLength;
        int repeats = frames / length;
        var predictions = new int[repeats, length];
        for (int r = 0; r < repeats; r++)
            for (int j = 0; j < length; j++)
            {
                int f = r * length + j, best = 0;
                for (int d = 1; d < dim; d++)
                    if (scores[d, f] > scores[best, f]) best = d;
                predictions[r, j] = best;
            }
        return predictions;
    }

    /// <summary>
    /// The message in <paramref name="audio"/> (at <see cref="SilentCipherConfiguration.SampleRate"/>): whether one was
    /// found, its bytes, and the share of frames that agree with it. Phase-shift decoding first searches the offset into
    /// the audio that the frames agree on best, as CSM's check does.
    /// </summary>
    public (bool Found, int[] Message, double Confidence) Decode(double[] audio, bool phaseShiftDecoding)
    {
        if (audio is null || audio.Length == 0) throw new ArgumentException("There is no audio.", nameof(audio));
        var c = Configuration;
        double power = audio.Sum(v => v * v) / audio.Length;
        if (power == 0) return (false, Array.Empty<int>(), 0);
        double gain = Math.Sqrt(c.AverageEnergy / power);
        var y = audio.Select(v => v * gain).ToArray();
        using var _ = new NoGradScope<T>();
        Eval();
        int shift = 0;
        if (phaseShiftDecoding)
        {
            double bestAccuracy = 0;
            for (int ps = 0; ps < c.Hop; ps += 10)
            {
                if (ps >= y.Length) break;
                var predictions = Predictions(MessageScores(Transform(y.Skip(ps).ToArray()).Magnitude));
                int repeats = predictions.GetLength(0);
                if (repeats == 0) continue;
                double accuracy = 0;
                for (int j = 0; j < c.MessageLength; j++)
                {
                    var counts = new int[c.MessageDim];
                    for (int r = 0; r < repeats; r++) counts[predictions[r, j]]++;
                    accuracy += counts.Max() / (double)repeats;
                }
                accuracy /= c.MessageLength;
                if (accuracy > bestAccuracy)
                {
                    bestAccuracy = accuracy;
                    shift = ps;
                }
            }
        }
        var decoded = Predictions(MessageScores(Transform(y.Skip(shift).ToArray()).Magnitude));
        int count = decoded.GetLength(0);
        if (count == 0) return (false, Array.Empty<int>(), 0);
        // scipy.stats.mode per position (ties to the smallest symbol), then the share of frames agreeing with it.
        var mode = new int[c.MessageLength];
        int agree = 0;
        for (int j = 0; j < c.MessageLength; j++)
        {
            var counts = new int[c.MessageDim];
            for (int r = 0; r < count; r++) counts[decoded[r, j]]++;
            int best = 0;
            for (int d = 1; d < c.MessageDim; d++)
                if (counts[d] > counts[best]) best = d;
            mode[j] = best;
            agree += counts[best];
        }
        double confidence = agree / (double)(count * c.MessageLength);
        int end = Array.IndexOf(mode, 0);
        if (end < 0) return (false, Array.Empty<int>(), 0);
        var ordered = mode.Skip(end + 1).Concat(mode.Take(end)).Select(v => v - 1).ToArray();
        // Each symbol as two bits (a −1 symbol, possible on a wrong decode, prints as "-1" in the reference and spoils
        // the byte; it is reported as not found).
        if (ordered.Any(v => v < 0)) return (false, Array.Empty<int>(), confidence);
        var bytes = new int[ordered.Length * 2 / 8];
        for (int b = 0; b < bytes.Length; b++)
            for (int s = 0; s < 4; s++) bytes[b] = (bytes[b] << 2) | ordered[4 * b + s];
        return (true, bytes, confidence);
    }

    /// <summary>
    /// Loads the release's three state dicts (<c>enc_c.ckpt</c>, <c>dec_c.ckpt</c>, <c>dec_m_0.ckpt</c>; DataParallel's
    /// <c>module.</c> prefix removed). Returns the names read from each.
    /// </summary>
    public (HashSet<string> Encoder, HashSet<string> CarrierDecoder, HashSet<string> MessageDecoder) Load(
        Func<string, double[]?> encoder, Func<string, double[]?> carrierDecoder, Func<string, double[]?> messageDecoder)
    {
        (Func<string, double[]?> Read, Func<string, int, double[]> Need, HashSet<string> Used) Reader(Func<string, double[]?> source)
        {
            var used = new HashSet<string>(StringComparer.Ordinal);
            double[]? Read(string name)
            {
                var values = source(name);
                if (values is not null) used.Add(name);
                return values;
            }
            double[] Need(string name, int count)
            {
                var values = Read(name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
                if (values.Length != count) throw new InvalidDataException($"'{name}' has {values.Length} values; this model expects {count}.");
                return values;
            }
            return (Read, Need, used);
        }
        var c = Configuration;
        var e = Reader(encoder);
        for (int i = 0; i < _encoder.Length; i++) _encoder[i].Load(e.Read, $"main.{i}", e.Need);
        TorchParameters.Linear(_engine, _messageMap, c.MessageDim, c.MessageBandSize, e.Need("linear.weight", c.MessageBandSize * c.MessageDim),
            e.Need("linear.bias", c.MessageBandSize));
        var d = Reader(carrierDecoder);
        for (int i = 0; i < _carrierDecoder.Length; i++) _carrierDecoder[i].Load(d.Read, $"main.{i}", d.Need);
        var m = Reader(messageDecoder);
        // MsgDecoder's Sequential interleaves a Dropout(0) before each layer: the layers sit at odd indices.
        for (int i = 0; i < _messageDecoder.Length; i++) _messageDecoder[i].Load(m.Read, $"main.{2 * i + 1}", m.Need);
        TorchParameters.Linear(_engine, _messageReadout, c.MessageBandSize, 1, m.Need("linear.weight", c.MessageBandSize), m.Need("linear.bias", 1));
        return (e.Used, d.Used, m.Used);
    }

    /// <summary>Loads the release's checkpoint directory (<c>44_1_khz/73999_iteration</c>).</summary>
    public void LoadCheckpointDirectory(string directory)
    {
        Func<string, double[]?> Open(string file)
        {
            var path = Path.Combine(directory, file);
            if (!File.Exists(path)) throw new FileNotFoundException($"SilentCipher's {file} is missing.", path);
            var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
            foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path))
            {
                string name = entry.Key.StartsWith("module.", StringComparison.Ordinal) ? entry.Key.Substring("module.".Length) : entry.Key;
                state[name] = entry.Value.ToVector().Select(v => (double)v).ToArray();
            }
            return name => state.TryGetValue(name, out var v) ? v : null;
        }
        Load(Open("enc_c.ckpt"), Open("dec_c.ckpt"), Open("dec_m_0.ckpt"));
    }
}

/// <summary>Watermarking as Sesame's CSM applies it (<c>watermarking.py</c> <c>watermark</c> and <c>verify</c>).</summary>
internal static class SilentCipherWatermark
{
    /// <summary>The release's public key, which identifies audio made by CSM 1B ([212, 211, 146, 56, 201]).</summary>
    public static readonly int[] Csm1BKey = { 212, 211, 146, 56, 201 };

    /// <summary>The level of CSM's watermark below the audio (36 dB).</summary>
    public const double CsmMessageSdr = 36;

    /// <summary>
    /// Watermarks <paramref name="audio"/> at <paramref name="sampleRate"/>: resampled to the model's rate (torchaudio's
    /// windowed sinc), embedded, and resampled to min(model rate, <paramref name="sampleRate"/>).
    /// </summary>
    public static double[] Apply<T>(IEngine engine, SilentCipherNetwork<T> network, double[] audio, int sampleRate, IReadOnlyList<int> key,
        double messageSdr = CsmMessageSdr)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int rate = network.Configuration.SampleRate;
        double[] Resample(double[] x, int from, int to)
        {
            if (from == to) return x;
            var t = new Tensor<T>(new[] { x.Length });
            for (int i = 0; i < x.Length; i++) t[i] = ops.FromDouble(x[i]);
            using var _ = new NoGradScope<T>();
            var y = new TorchaudioResample<T>(engine, from, to).Forward(t);
            return Enumerable.Range(0, y.Length).Select(i => ops.ToDouble(y[i])).ToArray();
        }
        var encoded = network.Encode(Resample(audio, sampleRate, rate), key, messageSdr);
        return Resample(encoded, rate, Math.Min(rate, sampleRate));
    }

    /// <summary>Whether <paramref name="audio"/> carries <paramref name="key"/> (CSM's <c>verify</c>: resampled to the
    /// model's rate, phase-shift decoding).</summary>
    public static bool Verify<T>(IEngine engine, SilentCipherNetwork<T> network, double[] audio, int sampleRate, IReadOnlyList<int> key)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int rate = network.Configuration.SampleRate;
        var x = audio;
        if (sampleRate != rate)
        {
            var t = new Tensor<T>(new[] { audio.Length });
            for (int i = 0; i < audio.Length; i++) t[i] = ops.FromDouble(audio[i]);
            using var _ = new NoGradScope<T>();
            var y = new TorchaudioResample<T>(engine, sampleRate, rate).Forward(t);
            x = Enumerable.Range(0, y.Length).Select(i => ops.ToDouble(y[i])).ToArray();
        }
        var (found, message, _) = network.Decode(x, phaseShiftDecoding: true);
        return found && message.SequenceEqual(key);
    }
}
