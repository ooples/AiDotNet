using AiDotNet.Audio.Features;
using AiDotNet.Models.Options;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// LAION-CLAP's audio input pipeline (Wu et al. 2023; reference <c>training/data.py get_audio_features</c> and
/// <c>htsat.py</c>): fits a waveform to the 10-second clip and turns it into the log-mel input HTS-AT reads.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>Log-mel.</b> A centred, reflect-padded STFT with a periodic Hann window, the power spectrum, a mel
/// filterbank and <c>10 · log10(max(x, 1e-10))</c> (no top-dB clipping). The filterbank follows the model variant:
/// the fusion models were trained on torchaudio's (<see cref="MelFilterbanks.Htk"/>), the non-fusion models on
/// torchlibrosa's, which is librosa's (<see cref="MelFilterbanks.Slaney"/>).</para>
/// <para><b>Fitting the clip.</b> Audio shorter than the clip is repeated a whole number of times and zero-padded
/// ("repeatpad"). Longer audio is randomly cropped to the clip without fusion ("rand_trunc"). With fusion it
/// becomes four mel views: the whole clip's mel shrunk to the clip's frame count with antialiased bilinear
/// resampling (torchvision <c>Resize</c>), and three clip-length crops whose starts are drawn from the front,
/// middle and back thirds of the possible offsets. Such a sample is marked "longer"; any other sample's four views
/// are the same mel.</para>
/// </remarks>
internal sealed class ClapAudioFrontEnd<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly CLAPModelOptions _options;
    private readonly Tensor<T> _window;
    private readonly Tensor<T> _filterbank;   // [bins, mels]

    public ClapAudioFrontEnd(CLAPModelOptions options)
    {
        Guard.NotNull(options);
        _options = options;
        int n = options.StftWindowSize;
        _window = new Tensor<T>(new[] { n });
        // torch.hann_window / librosa get_window('hann', fftbins=True): the PERIODIC window, 2π n / N.
        for (int i = 0; i < n; i++) _window[i] = NumOps.FromDouble(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / n));
        double fMax = Math.Min(options.MelMaxFrequency, options.SampleRate / 2.0);
        var weights = options.EnableFeatureFusion
            ? MelFilterbanks.Htk(options.SampleRate, n, options.NumMelBands, options.MelMinFrequency, fMax)
            : MelFilterbanks.Slaney(options.SampleRate, n, options.NumMelBands, options.MelMinFrequency, fMax);
        int bins = n / 2 + 1;
        _filterbank = new Tensor<T>(new[] { bins, options.NumMelBands });
        for (int m = 0; m < options.NumMelBands; m++)
            for (int k = 0; k < bins; k++) _filterbank[k, m] = NumOps.FromDouble(weights[m, k]);
    }

    /// <summary>Frames a clip of <see cref="CLAPModelOptions.ClipSamples"/> samples yields.</summary>
    public int ClipFrames => _options.ClipSamples / _options.HopLength + 1;

    /// <summary>The log-mel spectrogram of each row: <c>[batch, samples]</c> → <c>[batch, frames, mels]</c>.</summary>
    public Tensor<T> LogMel(IEngine engine, Tensor<T> waveforms)
    {
        if (waveforms.Rank != 2) throw new ArgumentException("Expected [batch, samples].", nameof(waveforms));
        int batch = waveforms.Shape[0];
        engine.STFT(waveforms, _options.StftWindowSize, _options.HopLength, _window, center: true, out var magnitude, out _);
        int bins = _options.StftWindowSize / 2 + 1;
        int frames = magnitude.Length / (batch * bins);
        var power = engine.TensorMultiply(magnitude, magnitude);
        var framesFirst = engine.Reshape(engine.TensorPermute(engine.Reshape(power, new[] { batch, bins, frames }), new[] { 0, 2, 1 }),
            new[] { batch * frames, bins });
        var mel = engine.TensorMatMul(framesFirst, _filterbank);
        // AmplitudeToDB(stype="power", top_db=None) / torchlibrosa power_to_db(ref=1, amin=1e-10, top_db=None).
        var floored = engine.TensorClampMin(mel, NumOps.FromDouble(1e-10));
        var db = engine.TensorMultiplyScalar(engine.TensorLog(floored), NumOps.FromDouble(10.0 / Math.Log(10.0)));
        return engine.Reshape(db, new[] { batch, frames, _options.NumMelBands });
    }

    /// <summary>
    /// Fits each row to the clip and returns the model input <c>[batch, views, frames, mels]</c> (one view without
    /// fusion, four with it) and the <c>[batch]</c> longer mask (all zero without fusion).
    /// </summary>
    public (Tensor<T> Features, Tensor<T> Longer) Prepare(IEngine engine, Tensor<T> rawAudio, Random random)
    {
        var waveforms = rawAudio.Rank == 1 ? engine.Reshape(rawAudio, new[] { 1, rawAudio.Shape[0] }) : rawAudio;
        if (waveforms.Rank != 2) throw new ArgumentException("Audio must be [samples] or [batch, samples].", nameof(rawAudio));
        int batch = waveforms.Shape[0], samples = waveforms.Shape[1], clip = _options.ClipSamples, mels = _options.NumMelBands;
        var longer = new Tensor<T>(new[] { batch });

        if (!_options.EnableFeatureFusion)
        {
            var fitted = samples > clip ? RandomCrop(waveforms, clip, random) : RepeatPad(waveforms, clip);
            var mel = LogMel(engine, fitted);
            return (engine.Reshape(mel, new[] { batch, 1, mel.Shape[1], mels }), longer);
        }

        int chunk = ClipFrames;
        if (samples <= clip)
        {
            var mel = LogMel(engine, RepeatPad(waveforms, clip));
            return (FourViews(engine, mel), longer);
        }

        var full = LogMel(engine, waveforms);                              // [B, total, mels]
        int total = full.Shape[1];
        if (total == chunk) return (FourViews(engine, full), longer);      // longer than the clip by under one hop

        var views = new Tensor<T>(new[] { batch, 4, chunk, mels });
        var ranges = SplitOffsets(total - chunk + 1);
        for (int b = 0; b < batch; b++)
        {
            longer[b] = NumOps.One;
            var row = engine.Reshape(engine.TensorSlice(full, new[] { b, 0, 0 }, new[] { 1, total, mels }), new[] { total, mels });
            var shrunk = AntialiasedBilinearShrink(row, chunk);
            Copy(shrunk, 0, views, b, 0, chunk, mels);
            for (int part = 0; part < 3; part++)
            {
                var range = ranges[part];
                int start = range[random.Next(range.Length)];
                Copy(row, start, views, b, part + 1, chunk, mels);
            }
        }
        return (views, longer);
    }

    private static void Copy(Tensor<T> source, int startFrame, Tensor<T> target, int b, int view, int frames, int mels)
    {
        for (int t = 0; t < frames; t++)
            for (int m = 0; m < mels; m++) target[b, view, t, m] = source[startFrame + t, m];
    }

    private static Tensor<T> FourViews(IEngine engine, Tensor<T> mel)
    {
        int batch = mel.Shape[0], frames = mel.Shape[1], mels = mel.Shape[2];
        var one = engine.Reshape(mel, new[] { batch, 1, frames, mels });
        return engine.TensorConcatenate(new[] { one, one, one, one }, 1);
    }

    // np.array_split(range(count), 3), with an empty middle or back third replaced by [0].
    internal static int[][] SplitOffsets(int count)
    {
        var parts = new int[3][];
        int baseSize = count / 3, extra = count % 3, start = 0;
        for (int p = 0; p < 3; p++)
        {
            int size = baseSize + (p < extra ? 1 : 0);
            parts[p] = Enumerable.Range(start, size).ToArray();
            start += size;
        }
        for (int p = 1; p < 3; p++) if (parts[p].Length == 0) parts[p] = new[] { 0 };
        return parts;
    }

    // audio.repeat(int(clip / len)) then zero-pad to the clip.
    private static Tensor<T> RepeatPad(Tensor<T> waveforms, int clip)
    {
        int batch = waveforms.Shape[0], samples = waveforms.Shape[1];
        if (samples == clip) return waveforms;
        if (samples <= 0) throw new ArgumentException("Audio must contain at least one sample.", nameof(waveforms));
        int repeats = clip / samples;
        var result = new Tensor<T>(new[] { batch, clip });
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < repeats * samples; i++) result[b, i] = waveforms[b, i % samples];
        return result;
    }

    private static Tensor<T> RandomCrop(Tensor<T> waveforms, int clip, Random random)
    {
        int batch = waveforms.Shape[0], overflow = waveforms.Shape[1] - clip;
        var result = new Tensor<T>(new[] { batch, clip });
        for (int b = 0; b < batch; b++)
        {
            int offset = random.Next(overflow + 1);
            for (int i = 0; i < clip; i++) result[b, i] = waveforms[b, offset + i];
        }
        return result;
    }

    /// <summary>
    /// Resamples the rows of <paramref name="rows"/> (<c>[inputs, mels]</c>) to <paramref name="outputs"/> rows with
    /// PyTorch's antialiased bilinear filter along that axis (<c>_upsample_bilinear2d_aa</c>, which torchvision's
    /// <c>Resize</c> uses on tensors): a triangle filter whose support widens by the downsampling factor, centred at
    /// <c>scale · (i + 0.5)</c>, normalized per output.
    /// </summary>
    /// <remarks>
    /// Each output reads only its filter window, about <c>2 · inputs / outputs</c> rows, so the cost is
    /// <c>O(inputs · mels)</c>. A dense <c>[outputs, inputs]</c> resampling matrix grew with the clip length: a
    /// ten-minute clip at the default 1001-frame chunk built a matrix of about 60 million entries.
    /// </remarks>
    internal static Tensor<T> AntialiasedBilinearShrink(Tensor<T> rows, int outputs)
    {
        int inputs = rows.Shape[0], mels = rows.Shape[1];
        double scale = inputs / (double)outputs;
        double support = scale >= 1.0 ? scale : 1.0;
        double invScale = scale >= 1.0 ? 1.0 / scale : 1.0;
        var result = new Tensor<T>(new[] { outputs, mels });
        // One flat copy of the rows: the inner loop reads each row's mels contiguously, which a per-element tensor
        // indexer would make several times slower, and the copy is the size of the input it reads.
        var source = rows.ToArray();
        var accumulator = new double[mels];
        // Sized to the widest filter window (support reaches about scale + 1 rows on each side), not to the input.
        var weights = new double[(int)Math.Ceiling(2 * support) + 2];
        for (int i = 0; i < outputs; i++)
        {
            double center = scale * (i + 0.5);
            int xmin = Math.Max((int)(center - support + 0.5), 0);
            int xmax = Math.Min((int)(center + support + 0.5), inputs);
            double total = 0;
            for (int x = xmin; x < xmax; x++)
            {
                double arg = Math.Abs((x - center + 0.5) * invScale);
                double weight = arg < 1.0 ? 1.0 - arg : 0.0;
                weights[x - xmin] = weight;
                total += weight;
            }

            Array.Clear(accumulator, 0, mels);
            for (int x = xmin; x < xmax; x++)
            {
                double weight = total > 0 ? weights[x - xmin] / total : 0.0;
                // Triangle weights are finite and never negative, so this skips exactly the zero-weight taps.
                if (weight <= 0.0) continue;
                int offset = x * mels;
                for (int m = 0; m < mels; m++) accumulator[m] += weight * NumOps.ToDouble(source[offset + m]);
            }
            for (int m = 0; m < mels; m++) result[i, m] = NumOps.FromDouble(accumulator[m]);
        }
        return result;
    }
}
