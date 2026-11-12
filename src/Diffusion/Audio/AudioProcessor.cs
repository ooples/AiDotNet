using AiDotNet.Interfaces;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Audio;

/// <summary>
/// Complete audio processing pipeline for diffusion-based audio generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class combines STFT, Mel spectrogram, and Griffin-Lim into a unified
/// pipeline for audio analysis and synthesis. It's designed for use with
/// diffusion models like Riffusion that generate spectrograms.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is your one-stop shop for working with audio in
/// diffusion models. It handles:
///
/// - Converting audio waveforms to spectrograms (for training/conditioning)
/// - Converting spectrograms back to audio (for generation)
/// - Normalizing and denormalizing spectrograms
///
/// Typical workflow for Riffusion-style generation:
/// ```csharp
/// var processor = new AudioProcessor&lt;float&gt;(sampleRate: 44100);
///
/// // Encode reference audio to latent space (via spectrogram)
/// var referenceAudio = Tensor&lt;float&gt;.CreateRandom(44100);   // one second at 44.1 kHz
/// var spectrogram = processor.AudioToSpectrogram(referenceAudio);
/// var normalized = processor.NormalizeSpectrogram(spectrogram);
///
/// // ... diffusion model generates new spectrogram ...
/// var generatedSpec = normalized;
///
/// // Decode generated spectrogram back to audio
/// var denormalized = processor.DenormalizeSpectrogram(generatedSpec);
/// var audio = processor.SpectrogramToAudio(denormalized);
/// ```
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class AudioProcessor<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Sample rate in Hz.
    /// </summary>
    private readonly int _sampleRate;

    /// <summary>
    /// FFT size.
    /// </summary>
    private readonly int _nFft;

    /// <summary>
    /// Hop length between frames.
    /// </summary>
    private readonly int _hopLength;

    /// <summary>
    /// Number of Mel bins.
    /// </summary>
    private readonly int _nMels;

    /// <summary>
    /// Minimum dB for normalization.
    /// </summary>
    private readonly double _minDb;

    /// <summary>
    /// Maximum dB for normalization.
    /// </summary>
    private readonly double _maxDb;

    /// <summary>
    /// STFT processor.
    /// </summary>
    private readonly ShortTimeFourierTransform<T> _stft;

    /// <summary>
    /// Mel spectrogram processor.
    /// </summary>
    private readonly MelSpectrogram<T> _melSpec;

    /// <summary>
    /// Griffin-Lim algorithm.
    /// </summary>
    private readonly GriffinLim<T> _griffinLim;

    /// <summary>
    /// Gets the sample rate.
    /// </summary>
    public int SampleRate => _sampleRate;

    /// <summary>
    /// Gets the FFT size.
    /// </summary>
    public int NFft => _nFft;

    /// <summary>
    /// Gets the hop length.
    /// </summary>
    public int HopLength => _hopLength;

    /// <summary>
    /// Gets the number of Mel bins.
    /// </summary>
    public int NumMels => _nMels;

    /// <summary>
    /// Gets the STFT processor.
    /// </summary>
    public ShortTimeFourierTransform<T> STFT => _stft;

    /// <summary>
    /// Gets the Mel spectrogram processor.
    /// </summary>
    public MelSpectrogram<T> MelSpectrogram => _melSpec;

    /// <summary>
    /// Gets the Griffin-Lim processor.
    /// </summary>
    public GriffinLim<T> GriffinLim => _griffinLim;

    /// <summary>
    /// Initializes a new audio processor with Riffusion-compatible defaults.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate in Hz (default: 44100).</param>
    /// <param name="nFft">FFT size (default: 2048).</param>
    /// <param name="hopLength">Hop length (default: 512).</param>
    /// <param name="nMels">Number of Mel bins (default: 512 for Riffusion).</param>
    /// <param name="fMin">Minimum frequency (default: 0).</param>
    /// <param name="fMax">Maximum frequency (default: sampleRate/2).</param>
    /// <param name="minDb">Minimum dB for normalization (default: -100).</param>
    /// <param name="maxDb">Maximum dB for normalization (default: 20).</param>
    /// <param name="griffinLimIterations">Griffin-Lim iterations (default: 60).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Default parameters are optimized for Riffusion-style
    /// generation at 44.1kHz sample rate. For speech processing, you might use:
    /// - sampleRate: 16000 or 22050
    /// - nMels: 80 (common for speech)
    /// - nFft: 1024 or 512
    /// </para>
    /// </remarks>
    public AudioProcessor(
        int sampleRate = 44100,
        int nFft = 2048,
        int hopLength = 512,
        int nMels = 512,
        double fMin = 0.0,
        double? fMax = null,
        double minDb = -100.0,
        double maxDb = 20.0,
        int griffinLimIterations = 60)
    {
        _sampleRate = sampleRate;
        _nFft = nFft;
        _hopLength = hopLength;
        _nMels = nMels;
        _minDb = minDb;
        _maxDb = maxDb;

        // Initialize STFT (uses HanningWindow by default - industry standard for audio)
        _stft = new ShortTimeFourierTransform<T>(
            nFft: nFft,
            hopLength: hopLength,
            center: true);

        // Initialize Mel spectrogram (without log for more control)
        _melSpec = new MelSpectrogram<T>(
            sampleRate: sampleRate,
            nMels: nMels,
            nFft: nFft,
            hopLength: hopLength,
            fMin: fMin,
            fMax: fMax,
            logMel: false);

        // Initialize Griffin-Lim
        _griffinLim = new GriffinLim<T>(
            _stft,
            iterations: griffinLimIterations,
            momentum: 0.99);
    }

    /// <summary>
    /// Converts audio waveform to a normalized Mel spectrogram.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Normalized Mel spectrogram [numFrames, nMels] in range [0, 1].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This converts audio into a 2D image-like representation
    /// that can be processed by image-based diffusion models.
    /// </para>
    /// </remarks>
    public Tensor<T> AudioToSpectrogram(Tensor<T> audio)
    {
        // Compute power spectrogram
        var powerSpec = _stft.Power(audio);

        // Convert to Mel scale
        var melSpec = _melSpec.FromPowerSpectrogram(powerSpec);

        // Convert to dB
        var dbSpec = PowerToDb(melSpec);

        // Normalize to [0, 1]
        return NormalizeSpectrogram(dbSpec);
    }

    /// <summary>
    /// Converts a normalized spectrogram back to audio.
    /// </summary>
    /// <param name="spectrogram">Normalized spectrogram [numFrames, nMels] in range [0, 1].</param>
    /// <param name="length">Expected output length (optional).</param>
    /// <returns>Audio waveform tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This takes a spectrogram (e.g., generated by a diffusion model)
    /// and converts it back to an audio waveform that can be played.
    /// </para>
    /// </remarks>
    public Tensor<T> SpectrogramToAudio(Tensor<T> spectrogram, int? length = null)
    {
        // Denormalize from [0, 1] to dB
        var dbSpec = DenormalizeSpectrogram(spectrogram);

        // Convert dB to power
        var powerSpec = DbToPower(dbSpec);

        // Invert Mel to linear magnitude spectrogram using GPU-accelerated method
        var magnitude = _melSpec.InvertMelToMagnitude(powerSpec);

        // Apply Griffin-Lim with GPU acceleration
        return _griffinLim.Reconstruct(magnitude, length);
    }

    /// <summary>
    /// Normalizes a dB spectrogram to [0, 1] range.
    /// </summary>
    /// <param name="dbSpectrogram">Spectrogram in dB.</param>
    /// <returns>Normalized spectrogram in [0, 1].</returns>
    public Tensor<T> NormalizeSpectrogram(Tensor<T> dbSpectrogram)
    {
        // Clamp to [minDb, maxDb], then normalize to [0, 1]: (db - minDb) / (maxDb - minDb)
        var engine = AiDotNetEngine.Current;
        var minDb = NumOps.FromDouble(_minDb);
        var db = ClampKeepingNaN(dbSpectrogram, minDb, NumOps.FromDouble(_maxDb));
        return engine.TensorDivideScalar(engine.TensorAddScalar(db, NumOps.Negate(minDb)), NumOps.FromDouble(_maxDb - _minDb));
    }

    /// <summary>
    /// Clamps every element to [<paramref name="low"/>, <paramref name="high"/>]; NaN compares false and is kept.
    /// </summary>
    private static Tensor<T> ClampKeepingNaN(Tensor<T> values, T low, T high)
    {
        var engine = AiDotNetEngine.Current;
        var upper = engine.TensorWhere(
            engine.TensorGreaterThan(values, high), Tensor<T>.CreateDefault(values._shape, high), values);
        return engine.TensorWhere(
            engine.TensorLessThan(upper, low), Tensor<T>.CreateDefault(values._shape, low), upper);
    }

    /// <summary>
    /// Denormalizes a [0, 1] spectrogram back to dB.
    /// </summary>
    /// <param name="normalizedSpectrogram">Normalized spectrogram.</param>
    /// <returns>Spectrogram in dB.</returns>
    public Tensor<T> DenormalizeSpectrogram(Tensor<T> normalizedSpectrogram)
    {
        // Clamp to [0, 1], then denormalize to dB: norm * (maxDb - minDb) + minDb
        var engine = AiDotNetEngine.Current;
        var norm = ClampKeepingNaN(normalizedSpectrogram, NumOps.Zero, NumOps.One);
        return engine.TensorAddScalar(
            engine.TensorMultiplyScalar(norm, NumOps.FromDouble(_maxDb - _minDb)), NumOps.FromDouble(_minDb));
    }

    /// <summary>
    /// Converts power spectrogram to dB.
    /// </summary>
    private Tensor<T> PowerToDb(Tensor<T> power)
    {
        // dB = 10 * log10(power + epsilon)
        var engine = AiDotNetEngine.Current;
        var shifted = engine.TensorAddScalar(power, NumOps.FromDouble(1e-10));
        return engine.TensorMultiplyScalar(engine.TensorLog10(shifted), NumOps.FromDouble(10.0));
    }

    /// <summary>
    /// Converts dB spectrogram to power.
    /// </summary>
    private Tensor<T> DbToPower(Tensor<T> db)
    {
        // power = 10^(dB / 10) = exp(dB * ln(10) / 10)
        var engine = AiDotNetEngine.Current;
        return engine.TensorExp(engine.TensorMultiplyScalar(db, NumOps.FromDouble(Math.Log(10.0) / 10.0)));
    }

    /// <summary>
    /// Computes the duration of audio from spectrogram dimensions.
    /// </summary>
    /// <param name="numFrames">Number of spectrogram frames.</param>
    /// <returns>Duration in seconds.</returns>
    public double FramesToDuration(int numFrames)
    {
        int signalLength = _nFft + (numFrames - 1) * _hopLength - _nFft;
        return (double)signalLength / _sampleRate;
    }

    /// <summary>
    /// Computes the number of frames for a given duration.
    /// </summary>
    /// <param name="durationSeconds">Duration in seconds.</param>
    /// <returns>Number of spectrogram frames.</returns>
    public int DurationToFrames(double durationSeconds)
    {
        int signalLength = (int)(durationSeconds * _sampleRate);
        return _stft.CalculateNumFrames(signalLength);
    }

    /// <summary>
    /// Computes the number of samples for a given duration.
    /// </summary>
    /// <param name="durationSeconds">Duration in seconds.</param>
    /// <returns>Number of audio samples.</returns>
    public int DurationToSamples(double durationSeconds)
    {
        return (int)(durationSeconds * _sampleRate);
    }

    /// <summary>
    /// Gets the time axis values for a spectrogram.
    /// </summary>
    /// <param name="numFrames">Number of spectrogram frames.</param>
    /// <returns>Array of time values in seconds for each frame.</returns>
    public double[] GetTimeAxis(int numFrames)
    {
        var times = new double[numFrames];
        for (int i = 0; i < numFrames; i++)
        {
            times[i] = (double)(i * _hopLength) / _sampleRate;
        }
        return times;
    }

    /// <summary>
    /// Gets the frequency axis values for a Mel spectrogram.
    /// </summary>
    /// <returns>Array of center frequencies in Hz for each Mel bin.</returns>
    public double[] GetMelFrequencyAxis()
    {
        return _melSpec.GetMelCenterFrequencies();
    }

    /// <summary>
    /// Pads or truncates audio to a specific length.
    /// </summary>
    /// <param name="audio">Input audio tensor.</param>
    /// <param name="targetLength">Target length in samples.</param>
    /// <returns>Audio tensor of specified length (padded with zeros).</returns>
    public Tensor<T> PadOrTruncate(Tensor<T> audio, int targetLength)
    {
        return PadOrTruncate(audio, targetLength, NumOps.Zero);
    }

    /// <summary>
    /// Pads or truncates audio to a specific length.
    /// </summary>
    /// <param name="audio">Input audio tensor.</param>
    /// <param name="targetLength">Target length in samples.</param>
    /// <param name="padValue">Value to use for padding.</param>
    /// <returns>Audio tensor of specified length.</returns>
    public Tensor<T> PadOrTruncate(Tensor<T> audio, int targetLength, T padValue)
    {
        var result = new Tensor<T>(new[] { targetLength });
        T pad = padValue;

        int copyLength = Math.Min(audio.Data.Length, targetLength);
        audio.Data.Span.Slice(0, copyLength).CopyTo(result.Data.Span);

        // Pad if needed
        for (int i = copyLength; i < targetLength; i++)
        {
            result.Data.Span[i] = pad;
        }

        return result;
    }

    /// <summary>
    /// Normalizes audio to a peak amplitude.
    /// </summary>
    /// <param name="audio">Input audio tensor.</param>
    /// <param name="targetPeak">Target peak amplitude (default: 0.95).</param>
    /// <returns>Normalized audio tensor.</returns>
    public Tensor<T> NormalizeAudio(Tensor<T> audio, double targetPeak = 0.95)
    {
        // Find current peak (NaN samples are ignored)
        var engine = AiDotNetEngine.Current;
        var absolute = engine.TensorNanToNum(engine.TensorAbs(audio), 0.0, double.PositiveInfinity, double.NegativeInfinity);
        T maxAbs = audio.Length == 0 ? NumOps.Zero : engine.TensorMaxValue(absolute);

        if (NumOps.LessThan(maxAbs, NumOps.FromDouble(1e-8)))
            return audio;

        // Scale to target peak
        return engine.TensorMultiplyScalar(audio, NumOps.Divide(NumOps.FromDouble(targetPeak), maxAbs));
    }

    /// <summary>
    /// Creates a spectrogram suitable for image-based diffusion models.
    /// </summary>
    /// <param name="audio">Input audio tensor.</param>
    /// <param name="targetWidth">Target spectrogram width (time dimension).</param>
    /// <param name="targetHeight">Target spectrogram height (frequency dimension).</param>
    /// <returns>Resized spectrogram tensor [height, width].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Diffusion models often expect fixed-size inputs like
    /// 512x512 or 1024x1024. This method creates a spectrogram and resizes it
    /// to match those dimensions.
    /// </para>
    /// </remarks>
    public Tensor<T> AudioToImageSpectrogram(Tensor<T> audio, int targetWidth = 512, int targetHeight = 512)
    {
        var spectrogram = AudioToSpectrogram(audio);

        // Simple bilinear resize
        return ResizeSpectrogram(spectrogram, targetHeight, targetWidth);
    }

    /// <summary>
    /// Resizes a spectrogram using bilinear interpolation.
    /// </summary>
    private Tensor<T> ResizeSpectrogram(Tensor<T> spectrogram, int targetHeight, int targetWidth)
    {
        int srcHeight = spectrogram.Shape[0];
        int srcWidth = spectrogram.Shape[1];
        var engine = AiDotNetEngine.Current;
        var source = engine.Reshape(
            engine.TensorNarrow(engine.Reshape(spectrogram, new[] { spectrogram.Length }), 0, 0, srcHeight * srcWidth),
            new[] { srcHeight, srcWidth });

        // Align-corners sample positions (a single target row/column samples position 0)
        void Axis(int target, int src, out Tensor<int> lower, out Tensor<int> upper, out Tensor<T> fraction)
        {
            var lo = new int[target];
            var hi = new int[target];
            var frac = new double[target];
            for (int i = 0; i < target; i++)
            {
                double pos = target == 1 ? 0 : (double)i * (src - 1) / (target - 1);
                lo[i] = (int)Math.Floor(pos);
                hi[i] = Math.Min(lo[i] + 1, src - 1);
                frac[i] = pos - lo[i];
            }

            lower = new Tensor<int>(new[] { target }, new Vector<int>(lo));
            upper = new Tensor<int>(new[] { target }, new Vector<int>(hi));
            fraction = new Tensor<double>(new[] { target }, new Vector<double>(frac)).Cast<T>();
        }

        Axis(targetHeight, srcHeight, out var y0, out var y1, out var yFrac);
        Axis(targetWidth, srcWidth, out var x0, out var x1, out var xFrac);
        var shape = new[] { targetHeight, targetWidth };
        var xw = engine.TensorBroadcastTo(engine.Reshape(xFrac, new[] { 1, targetWidth }), shape);
        var yw = engine.TensorBroadcastTo(engine.Reshape(yFrac, new[] { targetHeight, 1 }), shape);

        // Bilinear interpolation: v0 = v00 (1 - xf) + v01 xf, v1 = v10 (1 - xf) + v11 xf, value = v0 (1 - yf) + v1 yf
        Tensor<T> Blend(Tensor<T> a, Tensor<T> b, Tensor<T> w) =>
            engine.TensorAdd(engine.TensorMultiply(a, engine.ScalarMinusTensor(NumOps.One, w)), engine.TensorMultiply(b, w));
        var rows0 = engine.TensorIndexSelect(source, y0, 0);
        var rows1 = engine.TensorIndexSelect(source, y1, 0);
        var v0 = Blend(engine.TensorIndexSelect(rows0, x0, 1), engine.TensorIndexSelect(rows0, x1, 1), xw);
        var v1 = Blend(engine.TensorIndexSelect(rows1, x0, 1), engine.TensorIndexSelect(rows1, x1, 1), xw);
        return Blend(v0, v1, yw);
    }
}
