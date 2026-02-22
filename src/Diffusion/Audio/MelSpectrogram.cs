using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.WindowFunctions;

namespace AiDotNet.Diffusion.Audio;

/// <summary>
/// Computes Mel spectrograms from audio signals.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Mel spectrogram is a representation of audio that mimics human hearing.
/// It applies the Mel scale, which spaces frequencies according to how humans
/// perceive pitch rather than the physical frequency.
/// </para>
/// <para>
/// <b>For Beginners:</b> Human hearing doesn't perceive pitch linearly - we can
/// tell the difference between 100Hz and 200Hz easily, but 10,000Hz and 10,100Hz
/// sound almost the same to us. The Mel scale accounts for this.
///
/// A Mel spectrogram:
/// 1. Computes the power spectrogram using STFT
/// 2. Applies a bank of triangular filters on the Mel scale
/// 3. Takes the log (optional) to compress dynamic range
///
/// This representation is commonly used for:
/// - Speech recognition (like Whisper)
/// - Music generation (like Riffusion)
/// - Audio classification
/// - Speaker verification
///
/// Usage:
/// ```csharp
/// var melSpec = new MelSpectrogram&lt;float&gt;(
///     sampleRate: 44100,
///     nMels: 128,
///     nFft: 2048
/// );
/// var mel = melSpec.Forward(audioSignal);
/// // mel.Shape = [numFrames, nMels]
/// ```
/// </para>
/// </remarks>
public class MelSpectrogram<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Sample rate of the audio in Hz.
    /// </summary>
    private readonly int _sampleRate;

    /// <summary>
    /// Number of Mel frequency bins.
    /// </summary>
    private readonly int _nMels;

    /// <summary>
    /// Minimum frequency in Hz.
    /// </summary>
    private readonly double _fMin;

    /// <summary>
    /// Maximum frequency in Hz.
    /// </summary>
    private readonly double _fMax;

    /// <summary>
    /// Whether to apply log compression.
    /// </summary>
    private readonly bool _logMel;

    /// <summary>
    /// Reference value for dB conversion.
    /// </summary>
    private readonly double _refDb;

    /// <summary>
    /// Minimum dB value (floor).
    /// </summary>
    private readonly double _minDb;

    /// <summary>
    /// The STFT processor.
    /// </summary>
    private readonly ShortTimeFourierTransform<T> _stft;

    /// <summary>
    /// Mel filterbank matrix [nMels, nFreqs].
    /// </summary>
    private readonly Tensor<T> _melFilterbank;

    /// <summary>
    /// Window tensor for IEngine operations.
    /// </summary>
    private readonly Tensor<T> _windowTensor;

    /// <summary>
    /// FFT size for direct GPU operations.
    /// </summary>
    private readonly int _nFft;

    /// <summary>
    /// Hop length for direct GPU operations.
    /// </summary>
    private readonly int _hopLength;

    /// <summary>
    /// IEngine for GPU-accelerated operations.
    /// </summary>
    private IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets the number of Mel bins.
    /// </summary>
    public int NumMels => _nMels;

    /// <summary>
    /// Gets the sample rate.
    /// </summary>
    public int SampleRate => _sampleRate;

    /// <summary>
    /// Gets the STFT parameters.
    /// </summary>
    public ShortTimeFourierTransform<T> STFT => _stft;

    /// <summary>
    /// Initializes a new Mel spectrogram processor.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate in Hz (default: 22050).</param>
    /// <param name="nMels">Number of Mel frequency bins (default: 128).</param>
    /// <param name="nFft">FFT size (default: 2048).</param>
    /// <param name="hopLength">Hop length between frames (default: nFft/4).</param>
    /// <param name="fMin">Minimum frequency in Hz (default: 0).</param>
    /// <param name="fMax">Maximum frequency in Hz (default: sampleRate/2).</param>
    /// <param name="windowFunction">Window function to use (default: HanningWindow - industry standard for audio).</param>
    /// <param name="logMel">Whether to apply log compression (default: true).</param>
    /// <param name="refDb">Reference value for dB conversion (default: 1.0).</param>
    /// <param name="minDb">Minimum dB value floor (default: -80).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - sampleRate: Must match your audio file's sample rate
    /// - nMels: More bins = more frequency detail (128 is common for music, 80 for speech)
    /// - nFft: Larger = more frequency resolution, less time resolution
    /// - fMin/fMax: Filter out frequencies outside your range of interest
    /// - windowFunction: Reduces spectral leakage. Hann (default) is the industry standard.
    /// - logMel: Log compression makes the representation more perceptually uniform
    /// </para>
    /// </remarks>
    public MelSpectrogram(
        int sampleRate = 22050,
        int nMels = 128,
        int nFft = 2048,
        int? hopLength = null,
        double fMin = 0.0,
        double? fMax = null,
        IWindowFunction<T>? windowFunction = null,
        bool logMel = true,
        double refDb = 1.0,
        double minDb = -80.0)
    {
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "Sample rate must be positive.");
        if (nMels <= 0)
            throw new ArgumentOutOfRangeException(nameof(nMels), "Number of Mel bins must be positive.");

        _sampleRate = sampleRate;
        _nMels = nMels;
        _fMin = fMin;
        _fMax = fMax ?? sampleRate / 2.0;
        _logMel = logMel;
        _refDb = refDb;
        _minDb = minDb;

        if (_fMax > sampleRate / 2.0)
            throw new ArgumentOutOfRangeException(nameof(fMax), "fMax cannot exceed Nyquist frequency.");

        // Store FFT parameters for direct GPU operations
        _nFft = nFft;
        _hopLength = hopLength ?? nFft / 4;

        // Initialize STFT (uses HanningWindow by default - industry standard for audio)
        _stft = new ShortTimeFourierTransform<T>(
            nFft: nFft,
            hopLength: hopLength,
            windowFunction: windowFunction);

        // Create Mel filterbank
        _melFilterbank = CreateMelFilterbank(nMels, nFft, sampleRate, _fMin, _fMax);

        // Create window tensor for direct IEngine operations
        var window = windowFunction ?? new HanningWindow<T>();
        var windowVector = window.Create(nFft);
        var windowData = new T[nFft];
        for (int i = 0; i < nFft; i++)
        {
            windowData[i] = windowVector[i];
        }
        _windowTensor = new Tensor<T>(windowData, new[] { nFft });
    }

    /// <summary>
    /// Computes the Mel spectrogram of an audio signal.
    /// </summary>
    /// <param name="signal">Input audio signal.</param>
    /// <returns>Mel spectrogram tensor [numFrames, nMels].</returns>
    /// <remarks>
    /// <para>
    /// <b>GPU Acceleration:</b> When GPU is available, this method uses IEngine.MelSpectrogram
    /// for hardware-accelerated processing of the entire pipeline (STFT + Mel filterbank + dB conversion).
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> signal)
    {
        // Try GPU-accelerated path using IEngine.MelSpectrogram
        if (Engine.SupportsGpu)
        {
            try
            {
                T fMinT = NumOps.FromDouble(_fMin);
                T fMaxT = NumOps.FromDouble(_fMax);
                return Engine.MelSpectrogram(signal, _sampleRate, _nFft, _hopLength, _nMels, fMinT, fMaxT, _windowTensor, _logMel);
            }
            catch
            {
                // Fall back to CPU implementation on any error
            }
        }

        // CPU fallback path
        // Compute power spectrogram
        var powerSpec = _stft.Power(signal);

        // Apply Mel filterbank
        var melSpec = ApplyMelFilterbank(powerSpec);

        // Apply log compression if requested
        if (_logMel)
        {
            melSpec = PowerToDb(melSpec);
        }

        return melSpec;
    }

    /// <summary>
    /// Computes Mel spectrogram from a pre-computed power spectrogram.
    /// </summary>
    /// <param name="powerSpectrogram">Power spectrogram [numFrames, numFreqs].</param>
    /// <returns>Mel spectrogram tensor [numFrames, nMels].</returns>
    public Tensor<T> FromPowerSpectrogram(Tensor<T> powerSpectrogram)
    {
        var melSpec = ApplyMelFilterbank(powerSpectrogram);

        if (_logMel)
        {
            melSpec = PowerToDb(melSpec);
        }

        return melSpec;
    }

    /// <summary>
    /// Applies the Mel filterbank to a power spectrogram.
    /// </summary>
    private Tensor<T> ApplyMelFilterbank(Tensor<T> powerSpec)
    {
        int numFrames = powerSpec.Shape[0];
        int numFreqs = powerSpec.Shape.Length > 1 ? powerSpec.Shape[1] : powerSpec.Data.Length;
        int nMels = _melFilterbank.Shape[0];
        int filterFreqs = _melFilterbank.Shape[1];

        // Ensure dimensions match
        if (filterFreqs != numFreqs)
        {
            throw new ArgumentException(
                $"Power spectrogram has {numFreqs} frequency bins but filterbank expects {filterFreqs}.");
        }

        var melSpec = new Tensor<T>(new[] { numFrames, nMels });

        // Matrix multiplication: melSpec = powerSpec @ melFilterbank.T
        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int mel = 0; mel < nMels; mel++)
            {
                T sum = NumOps.Zero;
                for (int f = 0; f < numFreqs; f++)
                {
                    var power = powerSpec.Data.Span[frame * numFreqs + f];
                    var filter = _melFilterbank.Data.Span[mel * filterFreqs + f];
                    sum = NumOps.Add(sum, NumOps.Multiply(power, filter));
                }
                melSpec.Data.Span[frame * nMels + mel] = sum;
            }
        }

        return melSpec;
    }

    /// <summary>
    /// Converts power spectrogram to decibels.
    /// </summary>
    /// <param name="power">Power spectrogram.</param>
    /// <returns>dB spectrogram.</returns>
    private Tensor<T> PowerToDb(Tensor<T> power)
    {
        var db = new Tensor<T>(power.Shape);
        T refValue = NumOps.FromDouble(_refDb);
        T minDbT = NumOps.FromDouble(_minDb);
        T epsilon = NumOps.FromDouble(1e-10);
        T factor = NumOps.FromDouble(10.0);

        for (int i = 0; i < power.Data.Length; i++)
        {
            // dB = 10 * log10(power / ref + epsilon)
            var ratio = NumOps.Add(NumOps.Divide(power.Data.Span[i], refValue), epsilon);
            var logVal = NumOps.FromDouble(Math.Log10(NumOps.ToDouble(ratio)));
            var dbVal = NumOps.Multiply(factor, logVal);

            // Clamp to minimum dB
            if (NumOps.ToDouble(dbVal) < NumOps.ToDouble(minDbT))
            {
                dbVal = minDbT;
            }

            db.Data.Span[i] = dbVal;
        }

        return db;
    }

    /// <summary>
    /// Converts dB spectrogram back to power.
    /// </summary>
    /// <param name="db">dB spectrogram.</param>
    /// <returns>Power spectrogram.</returns>
    public Tensor<T> DbToPower(Tensor<T> db)
    {
        var power = new Tensor<T>(db.Shape);
        T refValue = NumOps.FromDouble(_refDb);
        T divisor = NumOps.FromDouble(10.0);

        for (int i = 0; i < db.Data.Length; i++)
        {
            // power = ref * 10^(dB / 10)
            var exponent = NumOps.Divide(db.Data.Span[i], divisor);
            var factor = NumOps.FromDouble(Math.Pow(10.0, NumOps.ToDouble(exponent)));
            power.Data.Span[i] = NumOps.Multiply(refValue, factor);
        }

        return power;
    }

    /// <summary>
    /// Inverts a Mel spectrogram to approximate magnitude spectrogram.
    /// </summary>
    /// <param name="melSpec">Mel spectrogram (linear or dB).</param>
    /// <param name="isDb">Whether the input is in dB (default: true if logMel was enabled).</param>
    /// <returns>Approximate magnitude spectrogram.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is an approximate inversion because the Mel filterbank
    /// is not perfectly invertible. The result can be used with Griffin-Lim to reconstruct audio.
    /// </para>
    /// </remarks>
    public Tensor<T> InvertMelToMagnitude(Tensor<T> melSpec, bool? isDb = null)
    {
        var linearMel = melSpec;

        // Convert from dB if needed
        if (isDb ?? _logMel)
        {
            linearMel = DbToPower(melSpec);
        }

        int numFrames = linearMel.Shape[0];
        int nMels = linearMel.Shape[1];
        int numFreqs = _melFilterbank.Shape[1];

        // Pseudo-inverse of filterbank (simple transpose with normalization)
        var magnitude = new Tensor<T>(new[] { numFrames, numFreqs });

        // Compute normalization factors for each frequency bin
        var filterNorm = new T[numFreqs];
        for (int f = 0; f < numFreqs; f++)
        {
            T sum = NumOps.Zero;
            for (int mel = 0; mel < nMels; mel++)
            {
                var filter = _melFilterbank.Data.Span[mel * numFreqs + f];
                sum = NumOps.Add(sum, NumOps.Multiply(filter, filter));
            }
            filterNorm[f] = NumOps.Add(sum, NumOps.FromDouble(1e-8));
        }

        // Apply pseudo-inverse
        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int f = 0; f < numFreqs; f++)
            {
                T sum = NumOps.Zero;
                for (int mel = 0; mel < nMels; mel++)
                {
                    var melVal = linearMel.Data.Span[frame * nMels + mel];
                    var filter = _melFilterbank.Data.Span[mel * numFreqs + f];
                    sum = NumOps.Add(sum, NumOps.Multiply(melVal, filter));
                }
                var normalized = NumOps.Divide(sum, filterNorm[f]);

                // Take square root to go from power to magnitude
                magnitude.Data.Span[frame * numFreqs + f] = NumOps.FromDouble(
                    Math.Sqrt(Math.Max(0, NumOps.ToDouble(normalized))));
            }
        }

        return magnitude;
    }

    /// <summary>
    /// Creates a Mel filterbank matrix.
    /// </summary>
    /// <param name="nMels">Number of Mel frequency bins.</param>
    /// <param name="nFft">FFT size.</param>
    /// <param name="sampleRate">Sample rate in Hz.</param>
    /// <param name="fMin">Minimum frequency in Hz.</param>
    /// <param name="fMax">Maximum frequency in Hz.</param>
    /// <returns>Filterbank matrix [nMels, nFreqs].</returns>
    private static Tensor<T> CreateMelFilterbank(int nMels, int nFft, int sampleRate, double fMin, double fMax)
    {
        int numFreqs = nFft / 2 + 1;
        var filterbank = new Tensor<T>(new[] { nMels, numFreqs });

        // Compute Mel frequencies for filter centers
        double melMin = HzToMel(fMin);
        double melMax = HzToMel(fMax);

        // nMels + 2 points: left edge, center points, right edge
        var melPoints = new double[nMels + 2];
        for (int i = 0; i < nMels + 2; i++)
        {
            melPoints[i] = melMin + (melMax - melMin) * i / (nMels + 1);
        }

        // Convert Mel points back to Hz
        var hzPoints = melPoints.Select(MelToHz).ToArray();

        // Convert Hz to FFT bin indices
        var binPoints = hzPoints.Select(hz => (int)Math.Floor((nFft + 1) * hz / sampleRate)).ToArray();

        // Create triangular filters
        for (int mel = 0; mel < nMels; mel++)
        {
            int leftBin = binPoints[mel];
            int centerBin = binPoints[mel + 1];
            int rightBin = binPoints[mel + 2];

            // Rising edge: leftBin to centerBin
            for (int f = leftBin; f < centerBin; f++)
            {
                if (f >= 0 && f < numFreqs && centerBin != leftBin)
                {
                    double weight = (double)(f - leftBin) / (centerBin - leftBin);
                    filterbank.Data.Span[mel * numFreqs + f] = NumOps.FromDouble(weight);
                }
            }

            // Falling edge: centerBin to rightBin
            for (int f = centerBin; f < rightBin; f++)
            {
                if (f >= 0 && f < numFreqs && rightBin != centerBin)
                {
                    double weight = (double)(rightBin - f) / (rightBin - centerBin);
                    filterbank.Data.Span[mel * numFreqs + f] = NumOps.FromDouble(weight);
                }
            }
        }

        // Normalize each filter to have unit area (Slaney-style normalization)
        for (int mel = 0; mel < nMels; mel++)
        {
            T sum = NumOps.Zero;
            for (int f = 0; f < numFreqs; f++)
            {
                sum = NumOps.Add(sum, filterbank.Data.Span[mel * numFreqs + f]);
            }

            if (NumOps.ToDouble(sum) > 1e-8)
            {
                for (int f = 0; f < numFreqs; f++)
                {
                    filterbank.Data.Span[mel * numFreqs + f] = NumOps.Divide(
                        filterbank.Data.Span[mel * numFreqs + f], sum);
                }
            }
        }

        return filterbank;
    }

    /// <summary>
    /// Converts frequency in Hz to Mel scale.
    /// </summary>
    /// <param name="hz">Frequency in Hz.</param>
    /// <returns>Frequency in Mels.</returns>
    /// <remarks>
    /// Uses the formula: mel = 2595 * log10(1 + hz / 700)
    /// </remarks>
    public static double HzToMel(double hz)
    {
        return 2595.0 * Math.Log10(1.0 + hz / 700.0);
    }

    /// <summary>
    /// Converts frequency in Mels to Hz.
    /// </summary>
    /// <param name="mel">Frequency in Mels.</param>
    /// <returns>Frequency in Hz.</returns>
    /// <remarks>
    /// Uses the formula: hz = 700 * (10^(mel / 2595) - 1)
    /// </remarks>
    public static double MelToHz(double mel)
    {
        return 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);
    }

    /// <summary>
    /// Gets the Mel filterbank matrix.
    /// </summary>
    /// <returns>Filterbank matrix [nMels, numFreqs].</returns>
    public Tensor<T> GetFilterbank()
    {
        var copy = new Tensor<T>(_melFilterbank.Shape);
        _melFilterbank.Data.Span.CopyTo(copy.Data.Span);
        return copy;
    }

    /// <summary>
    /// Computes the frequency (in Hz) for each Mel bin center.
    /// </summary>
    /// <returns>Array of center frequencies.</returns>
    public double[] GetMelCenterFrequencies()
    {
        double melMin = HzToMel(_fMin);
        double melMax = HzToMel(_fMax);

        var centerFreqs = new double[_nMels];
        for (int i = 0; i < _nMels; i++)
        {
            double mel = melMin + (melMax - melMin) * (i + 1) / (_nMels + 1);
            centerFreqs[i] = MelToHz(mel);
        }

        return centerFreqs;
    }
}
