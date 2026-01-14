using AiDotNet.Interfaces;

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
/// var spectrogram = processor.AudioToSpectrogram(referenceAudio);
/// var normalized = processor.NormalizeSpectrogram(spectrogram);
///
/// // ... diffusion model generates new spectrogram ...
///
/// // Decode generated spectrogram back to audio
/// var denormalized = processor.DenormalizeSpectrogram(generatedSpec);
/// var audio = processor.SpectrogramToAudio(denormalized);
/// ```
/// </para>
/// </remarks>
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
        var normalized = new Tensor<T>(dbSpectrogram.Shape);
        double range = _maxDb - _minDb;

        for (int i = 0; i < dbSpectrogram.Data.Length; i++)
        {
            double db = NumOps.ToDouble(dbSpectrogram.Data.Span[i]);

            // Clamp to [minDb, maxDb]
            db = Math.Max(_minDb, Math.Min(_maxDb, db));

            // Normalize to [0, 1]
            double norm = (db - _minDb) / range;
            normalized.Data.Span[i] = NumOps.FromDouble(norm);
        }

        return normalized;
    }

    /// <summary>
    /// Denormalizes a [0, 1] spectrogram back to dB.
    /// </summary>
    /// <param name="normalizedSpectrogram">Normalized spectrogram.</param>
    /// <returns>Spectrogram in dB.</returns>
    public Tensor<T> DenormalizeSpectrogram(Tensor<T> normalizedSpectrogram)
    {
        var denormalized = new Tensor<T>(normalizedSpectrogram.Shape);
        double range = _maxDb - _minDb;

        for (int i = 0; i < normalizedSpectrogram.Data.Length; i++)
        {
            double norm = NumOps.ToDouble(normalizedSpectrogram.Data.Span[i]);

            // Clamp to [0, 1]
            norm = Math.Max(0, Math.Min(1, norm));

            // Denormalize to dB
            double db = norm * range + _minDb;
            denormalized.Data.Span[i] = NumOps.FromDouble(db);
        }

        return denormalized;
    }

    /// <summary>
    /// Converts power spectrogram to dB.
    /// </summary>
    private Tensor<T> PowerToDb(Tensor<T> power)
    {
        var db = new Tensor<T>(power.Shape);
        const double epsilon = 1e-10;

        for (int i = 0; i < power.Data.Length; i++)
        {
            double powerVal = NumOps.ToDouble(power.Data.Span[i]);
            double dbVal = 10.0 * Math.Log10(powerVal + epsilon);
            db.Data.Span[i] = NumOps.FromDouble(dbVal);
        }

        return db;
    }

    /// <summary>
    /// Converts dB spectrogram to power.
    /// </summary>
    private Tensor<T> DbToPower(Tensor<T> db)
    {
        var power = new Tensor<T>(db.Shape);

        for (int i = 0; i < db.Data.Length; i++)
        {
            double dbVal = NumOps.ToDouble(db.Data.Span[i]);
            double powerVal = Math.Pow(10.0, dbVal / 10.0);
            power.Data.Span[i] = NumOps.FromDouble(powerVal);
        }

        return power;
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
        // Find current peak
        double maxAbs = 0;
        for (int i = 0; i < audio.Data.Length; i++)
        {
            double absVal = Math.Abs(NumOps.ToDouble(audio.Data.Span[i]));
            if (absVal > maxAbs)
            {
                maxAbs = absVal;
            }
        }

        if (maxAbs < 1e-8)
            return audio;

        // Scale to target peak
        var result = new Tensor<T>(audio.Shape);
        double scale = targetPeak / maxAbs;

        for (int i = 0; i < audio.Data.Length; i++)
        {
            double scaled = NumOps.ToDouble(audio.Data.Span[i]) * scale;
            result.Data.Span[i] = NumOps.FromDouble(scaled);
        }

        return result;
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

        var resized = new Tensor<T>(new[] { targetHeight, targetWidth });

        for (int y = 0; y < targetHeight; y++)
        {
            // Guard against division by zero when targetHeight is 1
            double srcY = targetHeight == 1 ? 0 : (double)y * (srcHeight - 1) / (targetHeight - 1);
            int y0 = (int)Math.Floor(srcY);
            int y1 = Math.Min(y0 + 1, srcHeight - 1);
            double yFrac = srcY - y0;

            for (int x = 0; x < targetWidth; x++)
            {
                // Guard against division by zero when targetWidth is 1
                double srcX = targetWidth == 1 ? 0 : (double)x * (srcWidth - 1) / (targetWidth - 1);
                int x0 = (int)Math.Floor(srcX);
                int x1 = Math.Min(x0 + 1, srcWidth - 1);
                double xFrac = srcX - x0;

                // Bilinear interpolation
                double v00 = NumOps.ToDouble(spectrogram.Data.Span[y0 * srcWidth + x0]);
                double v01 = NumOps.ToDouble(spectrogram.Data.Span[y0 * srcWidth + x1]);
                double v10 = NumOps.ToDouble(spectrogram.Data.Span[y1 * srcWidth + x0]);
                double v11 = NumOps.ToDouble(spectrogram.Data.Span[y1 * srcWidth + x1]);

                double v0 = v00 * (1 - xFrac) + v01 * xFrac;
                double v1 = v10 * (1 - xFrac) + v11 * xFrac;
                double value = v0 * (1 - yFrac) + v1 * yFrac;

                resized.Data.Span[y * targetWidth + x] = NumOps.FromDouble(value);
            }
        }

        return resized;
    }
}
