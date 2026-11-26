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
        // torch.hann_window / librosa get_window('hann', fftbins=True): the PERIODIC window, 2π n / N.
        _window = n > 0 ? AiDotNetEngine.Current.TensorHannWindow<T>(n, periodic: true) : new Tensor<T>(new[] { n });
        double fMax = Math.Min(options.MelMaxFrequency, options.SampleRate / 2.0);
        var weights = options.EnableFeatureFusion
            ? MelFilterbanks.Htk(options.SampleRate, n, options.NumMelBands, options.MelMinFrequency, fMax)
            : MelFilterbanks.Slaney(options.SampleRate, n, options.NumMelBands, options.MelMinFrequency, fMax);
        int bins = n / 2 + 1;
        // The filterbank weights ([mels, bins]) as [bins, mels], converted to T once, in bulk
        int bands = options.NumMelBands;
        if (bands * bins == 0)
        {
            _filterbank = new Tensor<T>(new[] { bins, bands });
        }
        else
        {
            var values = new double[bands * bins];
            for (int m = 0; m < bands; m++)
                for (int k = 0; k < bins; k++) values[m * bins + k] = weights[m, k];
            _filterbank = AiDotNetEngine.Current.TensorTranspose(new Tensor<double>(values, new[] { bands, bins }).Cast<T>());
        }
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

        if (batch == 0) return (new Tensor<T>(new[] { batch, 4, chunk, mels }), longer);
        longer = Tensor<T>.CreateDefault(new[] { batch }, NumOps.One);
        var ranges = SplitOffsets(total - chunk + 1);

        // The global view: every row's whole mel shrunk to the clip's frame count, [B, total, mels] -> [B, chunk, mels].
        var framesFirst = engine.Reshape(engine.TensorPermute(full, new[] { 1, 0, 2 }), new[] { total, batch * mels });
        var global = engine.TensorPermute(
            engine.Reshape(AntialiasedBilinearShrink(framesFirst, chunk), new[] { chunk, batch, mels }), new[] { 1, 0, 2 });

        // The three local views: clip-length crops starting in the front, middle and back thirds of the offsets, one
        // start per row and third, drawn in that order. The crops are a gather through per-step frame indices, so a
        // compiled training plan redraws the starts on every replay instead of keeping the traced step's slices.
        var frames = AiDotNet.Training.CompiledStepRandom<T>.DrawData<int>(new[] { batch * 3 * chunk }, indices =>
        {
            var span = indices.AsWritableSpan();
            for (int b = 0; b < batch; b++)
            {
                for (int part = 0; part < 3; part++)
                {
                    var range = ranges[part];
                    int first = b * total + range[random.Next(range.Length)], at = (b * 3 + part) * chunk;
                    for (int k = 0; k < chunk; k++) span[at + k] = first + k;
                }
            }
        });
        var local = engine.Reshape(
            engine.TensorGather(engine.Reshape(full, new[] { batch * total, mels }), frames, axis: 0),
            new[] { batch, 3, chunk, mels });
        return (engine.TensorConcatenate(new[] { engine.Reshape(global, new[] { batch, 1, chunk, mels }), local }, 1), longer);
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
        int filled = repeats * samples;
        if (batch == 0 || filled == 0) return new Tensor<T>(new[] { batch, clip });

        // The clip repeated `repeats` whole times, then zeros to the clip length
        var engine = AiDotNetEngine.Current;
        var tiled = repeats == 1 ? waveforms : engine.TensorTile(waveforms, new[] { 1, repeats });
        return filled == clip ? tiled : engine.Concat(new[] { tiled, new Tensor<T>(new[] { batch, clip - filled }) }, 1);
    }

    private static Tensor<T> RandomCrop(Tensor<T> waveforms, int clip, Random random)
    {
        int batch = waveforms.Shape[0], overflow = waveforms.Shape[1] - clip;
        if (batch == 0) return new Tensor<T>(new[] { batch, clip });
        var engine = AiDotNetEngine.Current;
        int samples = waveforms.Shape[1];
        // One offset per row, drawn in row order: the only thing this step draws, on the eager tape and in a plan.
        var offsets = AiDotNet.Training.CompiledStepRandom<T>.DrawData<int>(new[] { batch }, drawn =>
        {
            var span = drawn.AsWritableSpan();
            for (int b = 0; b < batch; b++) span[b] = random.Next(overflow + 1);
        });

        if (!AiDotNet.Tensors.Engines.Compilation.GraphMode.IsActive)
        {
            // Nothing is being recorded, so each row is a slice at its offset and no index is built.
            var starts = offsets.AsSpan();
            var crops = new Tensor<T>[batch];
            for (int b = 0; b < batch; b++)
                crops[b] = engine.TensorSlice(waveforms, new[] { b, starts[b] }, new[] { 1, clip });
            return engine.Concat(crops, 0);
        }

        // A recorded slice keeps the offset it was traced with, and indices computed inside the trace from the
        // offsets are not something AiDotNet.Tensors 0.152.1 can record a gather through (the plan's first step reads
        // them as zeros). So a plan takes the samples through one flat index per sample, refilled from this step's
        // offsets before every replay; with traced index arithmetic this becomes the offsets plus an arange.
        var positions = new Tensor<int>(new[] { batch * clip });
        void Fill(Tensor<int> indices)
        {
            var span = indices.AsWritableSpan();
            var starts = offsets.AsSpan();
            for (int b = 0; b < batch; b++)
            {
                int first = b * samples + starts[b], at = b * clip;
                for (int k = 0; k < clip; k++) span[at + k] = first + k;
            }
        }
        Fill(positions);
        AiDotNet.Training.CompiledStepRandom<T>.RecordDerived(positions, Fill);
        return engine.Reshape(engine.TensorTake(engine.Reshape(waveforms, new[] { batch * samples }), positions),
            new[] { batch, clip });
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
        if (outputs <= 0 || mels == 0)
            return new Tensor<T>(new[] { outputs, mels });

        double scale = inputs / (double)outputs;
        double support = scale >= 1.0 ? scale : 1.0;
        double invScale = scale >= 1.0 ? 1.0 / scale : 1.0;

        // Filter window [xmin, xmax) of each output; windows are about scale + 1 rows on each side
        var windowStart = new int[outputs];
        var windowCount = new int[outputs];
        int taps = 0;
        for (int i = 0; i < outputs; i++)
        {
            double center = scale * (i + 0.5);
            int xmin = Math.Max((int)(center - support + 0.5), 0);
            int xmax = Math.Min((int)(center + support + 0.5), inputs);
            windowStart[i] = xmin;
            windowCount[i] = Math.Max(0, xmax - xmin);
            taps = Math.Max(taps, windowCount[i]);
        }
        if (taps == 0)
            return new Tensor<T>(new[] { outputs, mels });

        // Triangle weights 1 - |(x - center + 0.5) / scale| (clamped at 0) over each window, normalized per output
        // (all-zero windows stay zero). The filter depends only on the two sizes, so it is built here on the host,
        // in double (frame positions past 2048 are not exact in Half), and rounded to T once: a compiled training
        // trace reads the weights and the row indices as constants instead of recording their arithmetic.
        var weights = new double[taps][];
        var sources = new int[taps][];
        for (int k = 0; k < taps; k++)
        {
            weights[k] = new double[outputs];
            sources[k] = new int[outputs];
        }
        int lastRow = Math.Max(0, inputs - 1);
        for (int i = 0; i < outputs; i++)
        {
            double center = scale * (i + 0.5), total = 0.0;
            for (int k = 0; k < windowCount[i]; k++)
            {
                double arg = Math.Abs((windowStart[i] + k - center + 0.5) * invScale);
                double weight = arg < 1.0 ? 1.0 - arg : 0.0;
                weights[k][i] = weight;
                total += weight;
            }
            for (int k = 0; k < taps; k++)
            {
                weights[k][i] = total > 0.0 ? weights[k][i] / total : 0.0;
                sources[k][i] = Math.Min(windowStart[i] + k, lastRow);
            }
        }

        // result[i] = sum over the window taps of weight * row; zero-weight taps are skipped (so a
        // non-finite row they would touch does not leak in)
        var engine = AiDotNetEngine.Current;
        var shape = new[] { outputs, mels };
        var outputZeros = new Tensor<T>(shape);
        var result = outputZeros;
        T zero = NumOps.Zero;
        for (int k = 0; k < taps; k++)
        {
            Tensor<T> column;
            using (AiDotNet.Tensors.Engines.Compilation.GraphMode.SuspendRecording())
                column = new Tensor<double>(weights[k], new[] { outputs, 1 }).Cast<T>();
            var tapRows = engine.TensorIndexSelect(rows, new Tensor<int>(sources[k], new[] { outputs }), 0);
            var tapWeight = engine.TensorBroadcastTo(column, shape);
            result = engine.TensorAdd(result, engine.TensorWhere(
                engine.TensorGreaterThan(tapWeight, zero), engine.TensorMultiply(tapWeight, tapRows), outputZeros));
        }
        return result;
    }
}
