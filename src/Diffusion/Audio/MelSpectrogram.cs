using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.WindowFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;

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
/// var audioSignal = Tensor&lt;float&gt;.CreateRandom(44100);   // one second at 44.1 kHz
/// var mel = melSpec.Forward(audioSignal);
/// // mel.Shape = [numFrames, nMels]
/// ```
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public partial class MelSpectrogram<T>
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
    /// When set, compression is <c>log(mel + offset)</c> instead of <see cref="PowerToDb"/>.
    /// </summary>
    private readonly double? _logOffset;

    /// <summary>
    /// The STFT processor.
    /// </summary>
    private readonly ShortTimeFourierTransform<T> _stft;

    /// <summary>
    /// Mel filterbank matrix [nMels, nFreqs].
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Tensor<T> _melFilterbank;

    /// <summary>
    /// Window tensor for IEngine operations.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
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
        double minDb = -80.0,
        double? logOffset = null)
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

        // Stabilised log-mel, log(mel + offset), as an alternative to dB. Both compress the same
        // spectrogram and are monotonically related, but they are NOT interchangeable: dB is
        // 10*log10(power/ref) floored at minDb, while this is a natural log with a small additive
        // floor and no clamp, so the two produce different scales and different gradients. Models
        // whose published recipe specifies the offset form need it exactly -- VGGish (AudioSet)
        // uses log(mel + 0.01), and feeding it dB instead changes the input distribution its
        // architecture was designed around.
        // NaN is spelled out rather than folded into `off <= 0.0`. The two are not the same test:
        // every comparison against NaN is false, so `off <= 0.0` ADMITS NaN while the intent here is
        // to reject it. A NaN offset would otherwise reach log(mel + NaN) and turn the whole
        // spectrogram into NaN, far from this constructor and long after the cause is visible.
        if (logOffset is { } off && (double.IsNaN(off) || off <= 0.0))
            throw new ArgumentOutOfRangeException(nameof(logOffset), "Log offset must be positive.");
        _logOffset = logOffset;

        if (_fMax > sampleRate / 2.0)
            throw new ArgumentOutOfRangeException(nameof(fMax), "fMax cannot exceed Nyquist frequency.");

        // Initialize STFT first so every path shares its exact FFT geometry. Arbitrary FFT
        // lengths are supported, including Whisper's paper-faithful 400-sample window.
        _stft = new ShortTimeFourierTransform<T>(
            nFft: nFft,
            hopLength: hopLength,
            windowFunction: windowFunction);

        // Store FFT parameters for direct GPU operations
        _nFft = _stft.NFft;
        // Hop is derived from the REQUESTED nFft, matching librosa's default of nFft/4.
        _hopLength = hopLength ?? nFft / 4;

        // Create Mel filterbank against the effective FFT size
        _melFilterbank = CreateMelFilterbank(nMels, _nFft, sampleRate, _fMin, _fMax);

        // Create window tensor for direct IEngine operations
        var window = windowFunction ?? new HanningWindow<T>();
        var windowVector = window.Create(_nFft);
        var windowData = windowVector.AsSpan().Slice(0, _nFft).ToArray();
        _windowTensor = new Tensor<T>(windowData, new[] { _nFft });
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
                // The fused kernel only knows dB compression, so when the offset form is requested
                // ask it for the raw mel power and apply the compression here. The STFT and
                // filterbank -- the expensive part -- still run on the GPU.
                bool fusedLog = _logMel && _logOffset is null;
                var gpuMel = Engine.MelSpectrogram(signal, _sampleRate, _nFft, _hopLength, _nMels, fMinT, fMaxT, _windowTensor, fusedLog);
                // The engine kernel returns [..., mel bins, frames]; this feature extractor's
                // public contract, shared with the CPU path, is [..., frames, mel bins].
                var axes = Enumerable.Range(0, gpuMel.Rank).ToArray();
                (axes[axes.Length - 2], axes[axes.Length - 1]) = (axes[axes.Length - 1], axes[axes.Length - 2]);
                gpuMel = Engine.TensorPermute(gpuMel, axes);
                return _logOffset is null ? gpuMel : ApplyLogOffset(gpuMel);
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
        melSpec = Compress(melSpec);

        return melSpec;
    }

    /// <summary>
    /// Computes Mel spectrogram from a pre-computed power spectrogram.
    /// </summary>
    /// <param name="powerSpectrogram">Power spectrogram [..., numFrames, numFreqs].</param>
    /// <returns>Mel spectrogram tensor [..., numFrames, nMels].</returns>
    public Tensor<T> FromPowerSpectrogram(Tensor<T> powerSpectrogram)
    {
        var melSpec = ApplyMelFilterbank(powerSpectrogram);

        melSpec = Compress(melSpec);

        return melSpec;
    }

    /// <summary>
    /// Applies the configured compression: <c>log(mel + offset)</c> when an offset was supplied,
    /// otherwise dB, otherwise none.
    /// </summary>
    private Tensor<T> Compress(Tensor<T> melSpec)
    {
        if (_logOffset is not null) return ApplyLogOffset(melSpec);
        return _logMel ? PowerToDb(melSpec) : melSpec;
    }

    /// <summary>
    /// Stabilised log compression, <c>log(mel + offset)</c>, as used by VGGish.
    /// </summary>
    /// <remarks>
    /// Natural log with an additive floor rather than a clamp: the offset alone keeps log(0)
    /// from being taken, so quiet bins stay ordered instead of piling up on a minimum-dB floor.
    /// </remarks>
    private Tensor<T> ApplyLogOffset(Tensor<T> melSpec)
    {
        double offset = _logOffset ?? 0.01;
        return Engine.TensorLog(Engine.TensorAddScalar(melSpec, NumOps.FromDouble(offset)));
    }

    /// <summary>
    /// Applies the Mel filterbank to a power spectrogram.
    /// </summary>
    private Tensor<T> ApplyMelFilterbank(Tensor<T> powerSpec)
    {
        if (powerSpec.Rank < 2)
        {
            throw new ArgumentException(
                $"Power spectrogram must have shape [..., frames, frequencies], got rank {powerSpec.Rank}.",
                nameof(powerSpec));
        }

        int numFrames = powerSpec.Shape[^2];
        int numFreqs = powerSpec.Shape[^1];
        int nMels = _melFilterbank.Shape[0];
        int filterFreqs = _melFilterbank.Shape[1];

        // Ensure dimensions match
        if (filterFreqs != numFreqs)
        {
            throw new ArgumentException(
                $"Power spectrogram has {numFreqs} frequency bins but filterbank expects {filterFreqs}.");
        }

        var outputShape = powerSpec.Shape.ToArray();
        outputShape[^1] = nMels;

        // Batched matrix multiplication: melSpec = powerSpec @ melFilterbank.T.
        // Every leading axis is folded into the frame axis and restored afterwards.
        var engine = Engine;
        var rows = engine.Reshape(powerSpec, new[] { powerSpec.Length / numFreqs, numFreqs });
        var melRows = engine.TensorMatMul(rows, engine.TensorTranspose(_melFilterbank));
        return engine.Reshape(melRows, outputShape);
    }

    /// <summary>
    /// Converts power spectrogram to decibels.
    /// </summary>
    /// <param name="power">Power spectrogram.</param>
    /// <returns>dB spectrogram.</returns>
    private Tensor<T> PowerToDb(Tensor<T> power)
    {
        var engine = Engine;
        T refValue = NumOps.FromDouble(_refDb);
        T minDbT = NumOps.FromDouble(_minDb);

        // dB = 10 * log10(power / ref + epsilon)
        var ratio = engine.TensorAddScalar(engine.TensorDivideScalar(power, refValue), NumOps.FromDouble(1e-10));
        var db = engine.TensorMultiplyScalar(engine.TensorLog10(ratio), NumOps.FromDouble(10.0));

        // Clamp to minimum dB (NaN compares false and is kept)
        return engine.TensorWhere(engine.TensorLessThan(db, minDbT), Tensor<T>.CreateDefault(db._shape, minDbT), db);
    }

    /// <summary>
    /// Converts dB spectrogram back to power.
    /// </summary>
    /// <param name="db">dB spectrogram.</param>
    /// <returns>Power spectrogram.</returns>
    public Tensor<T> DbToPower(Tensor<T> db)
    {
        // power = ref * 10^(dB / 10) = ref * exp(dB * ln(10) / 10)
        var engine = Engine;
        var factor = engine.TensorExp(engine.TensorMultiplyScalar(db, NumOps.FromDouble(Math.Log(10.0) / 10.0)));
        return engine.TensorMultiplyScalar(factor, NumOps.FromDouble(_refDb));
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

        var engine = Engine;
        int numFrames = linearMel.Shape[0];
        int nMels = linearMel.Shape[1];
        var mel = engine.Reshape(
            engine.TensorNarrow(engine.Reshape(linearMel, new[] { linearMel.Length }), 0, 0, numFrames * nMels),
            new[] { numFrames, nMels });

        // Pseudo-inverse of filterbank (simple transpose with normalization):
        // normalization per frequency bin is sum_mel filter^2 + 1e-8
        var filterNorm = engine.TensorAddScalar(
            engine.ReduceSum(engine.TensorSquare(_melFilterbank), new[] { 0 }, true), NumOps.FromDouble(1e-8));
        var projected = engine.TensorMatMul(mel, _melFilterbank);
        var normalized = engine.TensorDivide(projected, engine.TensorBroadcastTo(filterNorm, projected._shape));

        // Take square root of max(0, x) to go from power to magnitude (NaN is kept)
        var nonNegative = engine.TensorWhere(
            engine.TensorLessThan(normalized, NumOps.Zero), new Tensor<T>(normalized._shape), normalized);
        return engine.TensorSqrt(nonNegative);
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
        var engine = AiDotNetEngine.Current;
        int numFreqs = nFft / 2 + 1;
        var shape = new[] { nMels, numFreqs };

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

        // Per-filter edges as [nMels, numFreqs] columns, frequency bins as rows
        Tensor<T> Edges(int offset) => engine.TensorBroadcastTo(
            new Tensor<int>(new[] { nMels, 1 }, new Vector<int>(binPoints.Skip(offset).Take(nMels).ToArray())).Cast<T>(), shape);
        var left = Edges(0);
        var center = Edges(1);
        var right = Edges(2);
        var freqs = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorArange<T>(0, numFreqs, 1), new[] { 1, numFreqs }), shape);

        // Triangular filters: rising edge (f - left) / (center - left) on [left, center),
        // falling edge (right - f) / (right - center) on [center, right), zero elsewhere.
        // A non-empty edge range implies a non-zero denominator; the masked-out quotients are discarded.
        var rising = engine.TensorDivide(engine.TensorSubtract(freqs, left), engine.TensorSubtract(center, left));
        var falling = engine.TensorDivide(engine.TensorSubtract(right, freqs), engine.TensorSubtract(right, center));
        var onRising = engine.TensorMultiply(
            engine.ScalarMinusTensor(NumOps.One, engine.TensorLessThan(freqs, left)), engine.TensorLessThan(freqs, center));
        var onFalling = engine.TensorMultiply(
            engine.ScalarMinusTensor(NumOps.One, engine.TensorLessThan(freqs, center)), engine.TensorLessThan(freqs, right));
        var filterbank = engine.TensorWhere(onRising, rising,
            engine.TensorWhere(onFalling, falling, new Tensor<T>(shape)));

        // Normalize each filter to have unit area (Slaney-style normalization); filters summing to at most
        // 1e-8 are left unnormalized
        var sums = engine.TensorBroadcastTo(engine.ReduceSum(filterbank, new[] { 1 }, true), shape);
        return engine.TensorWhere(
            engine.TensorGreaterThan(sums, NumOps.FromDouble(1e-8)),
            engine.TensorDivide(filterbank, sums),
            filterbank);
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
        var copy = new Tensor<T>(_melFilterbank._shape);
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
