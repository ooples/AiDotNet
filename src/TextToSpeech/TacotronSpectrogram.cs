using AiDotNet.Audio.Features;
using EnginePadMode = AiDotNet.Tensors.Engines.PadMode;

namespace AiDotNet.TextToSpeech;

/// <summary>
/// The mel spectrogram and frame energy that Tacotron 2 and the acoustic models following it are trained on.
/// </summary>
/// <remarks>
/// <para>
/// Tacotron 2 (Shen et al. 2018) defines the target most neural TTS models reuse — FastSpeech 2 states it follows
/// Shen et al. with a 1024-sample frame and 256-sample hop at 22050 Hz (Ren et al. 2021, §3.1). Its reference
/// implementation (<c>TacotronSTFT</c>) takes the magnitude of a centred, reflect-padded, Hann-windowed STFT,
/// applies librosa's mel filterbank (Slaney mel scale, Slaney area normalization), and compresses with
/// <c>log(max(x, 1e-5))</c>. FastSpeech 2's energy is the L2 norm of each frame of that magnitude.
/// </para>
/// <para>The filterbank follows librosa <c>filters.mel</c> exactly, so targets match those produced by the
/// published preprocessing.</para>
/// </remarks>
public sealed class TacotronSpectrogram
{
    private readonly double[,] _melBasis;

    /// <summary>Creates the transform. Defaults are Tacotron 2 / FastSpeech 2's LJSpeech settings.</summary>
    public TacotronSpectrogram(int sampleRate = 22050, int fftSize = 1024, int hopLength = 256, int windowLength = 1024,
        int melChannels = 80, double minFrequency = 0.0, double maxFrequency = 8000.0, double clipValue = 1e-5)
    {
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (fftSize <= 0) throw new ArgumentOutOfRangeException(nameof(fftSize));
        if (hopLength <= 0) throw new ArgumentOutOfRangeException(nameof(hopLength));
        if (windowLength <= 0 || windowLength > fftSize) throw new ArgumentOutOfRangeException(nameof(windowLength));
        if (melChannels <= 0) throw new ArgumentOutOfRangeException(nameof(melChannels));
        if (maxFrequency <= minFrequency || maxFrequency > sampleRate / 2.0)
            throw new ArgumentOutOfRangeException(nameof(maxFrequency));

        SampleRate = sampleRate;
        FftSize = fftSize;
        HopLength = hopLength;
        WindowLength = windowLength;
        MelChannels = melChannels;
        ClipValue = clipValue;

        _melBasis = SlaneyMelBasis(sampleRate, fftSize, melChannels, minFrequency, maxFrequency);
    }

    /// <summary>Sampling rate, in Hz.</summary>
    public int SampleRate { get; }
    /// <summary>FFT size.</summary>
    public int FftSize { get; }
    /// <summary>Hop between frames, in samples.</summary>
    public int HopLength { get; }
    /// <summary>Analysis window length, in samples.</summary>
    public int WindowLength { get; }
    /// <summary>Number of mel channels.</summary>
    public int MelChannels { get; }
    /// <summary>Floor applied before the log.</summary>
    public double ClipValue { get; }

    /// <summary>Frame period in milliseconds (hop / sample rate), the period pitch must be extracted at to align.</summary>
    public double FramePeriodMs => 1000.0 * HopLength / SampleRate;

    /// <summary>
    /// Number of frames for a signal of <paramref name="sampleCount"/> samples: the reference count
    /// <c>1 + (samples + 2 (fftSize / 2) − fftSize) / hop</c> of the reflect-padded STFT (<c>1 + samples / hop</c> for an
    /// even FFT size).
    /// </summary>
    public int FrameCount(int sampleCount) => EngineStft<double>.FrameCount(sampleCount, FftSize, HopLength);

    /// <summary>
    /// The STFT magnitude, <c>[frames, fftSize/2 + 1]</c>: the signal reflect-padded by fftSize / 2 on each side, framed
    /// at the hop, windowed by the centred periodic Hann window and transformed (librosa <c>stft(center=True,
    /// pad_mode='reflect')</c>, Tacotron 2's <c>TacotronSTFT</c>).
    /// </summary>
    /// <typeparam name="T">The numeric type of the recording.</typeparam>
    public Tensor<T> Magnitude<T>(Tensor<T> audio)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        int pad = FftSize / 2;
        if (audio.Length <= pad)
            throw new ArgumentException($"Reflect padding needs more than {pad} samples.", nameof(audio));
        return EngineStft<T>.Magnitude(audio, FftSize, HopLength, EngineStft<T>.CenteredHann(FftSize, WindowLength),
            EnginePadMode.Reflect);
    }

    /// <summary>The log mel spectrogram of a recording, <c>[frames, melChannels]</c>.</summary>
    /// <typeparam name="T">The numeric type of the recording.</typeparam>
    public Tensor<T> LogMel<T>(Tensor<T> audio) => LogMelOfMagnitude(Magnitude(audio));

    /// <summary>The log mel spectrogram <c>log(max(M S, clip))</c> of an STFT magnitude <c>[frames, fftSize/2 + 1]</c>,
    /// <c>[frames, melChannels]</c>.</summary>
    /// <typeparam name="T">The numeric type of the magnitude.</typeparam>
    public Tensor<T> LogMelOfMagnitude<T>(Tensor<T> magnitude)
    {
        if (magnitude is null) throw new ArgumentNullException(nameof(magnitude));
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var mel = engine.TensorMatMul(magnitude, engine.TensorTranspose(EngineStft<T>.Table(_melBasis)));
        return engine.TensorLog(engine.TensorMax(mel, ops.FromDouble(ClipValue)));
    }

    /// <summary>
    /// FastSpeech 2's frame energy: the L2 norm of each STFT magnitude frame, <c>[frames]</c>. The magnitude is scaled
    /// by a power of two before it is squared (exact in binary floating point), so a 16-bit T cannot overflow.
    /// </summary>
    /// <typeparam name="T">The numeric type of the magnitude.</typeparam>
    public Tensor<T> Energy<T>(Tensor<T> magnitude)
    {
        if (magnitude is null) throw new ArgumentNullException(nameof(magnitude));
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        double scale = Math.Pow(2.0, Math.Ceiling(Math.Log(FftSize, 2.0)));
        var scaled = engine.TensorMultiplyScalar(magnitude, ops.FromDouble(1.0 / scale));
        var sumOfSquares = engine.ReduceSum(engine.TensorSquare(scaled), new[] { 1 }, false);
        return engine.TensorMultiplyScalar(engine.TensorSqrt(sumOfSquares), ops.FromDouble(scale));
    }

    /// <summary>The mel filterbank, <c>[melChannels, fftSize/2 + 1]</c> (librosa <c>filters.mel</c>, Slaney).</summary>
    public double[,] MelBasis => (double[,])_melBasis.Clone();

    private static double[,] SlaneyMelBasis(int sampleRate, int fftSize, int melChannels, double fMin, double fMax)
    {
        int bins = 1 + fftSize / 2;
        var fftFreqs = new double[bins];
        for (int k = 0; k < bins; k++) fftFreqs[k] = (double)k * sampleRate / fftSize;

        double melMin = HzToMel(fMin), melMax = HzToMel(fMax);
        var melF = new double[melChannels + 2];
        for (int i = 0; i < melF.Length; i++)
            melF[i] = MelToHz(melMin + (melMax - melMin) * i / (melF.Length - 1));

        var weights = new double[melChannels, bins];
        for (int i = 0; i < melChannels; i++)
        {
            double lowerDiff = melF[i + 1] - melF[i];
            double upperDiff = melF[i + 2] - melF[i + 1];
            double enorm = 2.0 / (melF[i + 2] - melF[i]);
            for (int k = 0; k < bins; k++)
            {
                double lower = -(melF[i] - fftFreqs[k]) / lowerDiff;
                double upper = (melF[i + 2] - fftFreqs[k]) / upperDiff;
                weights[i, k] = Math.Max(0, Math.Min(lower, upper)) * enorm;
            }
        }
        return weights;
    }

    /// <summary>
    /// librosa's <c>filters.mel</c> exactly as it computes its default float32 result (Slaney scale and norm): the bin
    /// frequencies by <c>np.fft.rfftfreq</c>, the band edges by <c>np.linspace</c>, each triangle stored to float32 before
    /// the area normalization multiplies it, and the product stored to float32 again. <c>[melChannels, fftSize/2 + 1]</c>.
    /// </summary>
    internal static double[,] LibrosaFloat32MelBasis(int sampleRate, int fftSize, int melChannels, double fMin, double fMax)
    {
        int bins = 1 + fftSize / 2;
        double spacing = 1.0 / (fftSize * (1.0 / sampleRate));
        var fftFreqs = new double[bins];
        for (int k = 0; k < bins; k++) fftFreqs[k] = k * spacing;
        double melMin = HzToMel(fMin), melMax = HzToMel(fMax);
        int count = melChannels + 2;
        double step = (melMax - melMin) / (count - 1);
        var melF = new double[count];
        for (int i = 0; i < count; i++) melF[i] = MelToHz(i == count - 1 ? melMax : i * step + melMin);
        var weights = new double[melChannels, bins];
        for (int i = 0; i < melChannels; i++)
        {
            double lowerDiff = melF[i + 1] - melF[i];
            double upperDiff = melF[i + 2] - melF[i + 1];
            double enorm = 2.0 / (melF[i + 2] - melF[i]);
            for (int k = 0; k < bins; k++)
            {
                double lower = -(melF[i] - fftFreqs[k]) / lowerDiff;
                double upper = (melF[i + 2] - fftFreqs[k]) / upperDiff;
                float triangle = (float)Math.Max(0, Math.Min(lower, upper));
                weights[i, k] = (float)(triangle * enorm);
            }
        }
        return weights;
    }

    // librosa hz_to_mel / mel_to_hz with htk=False (Slaney): linear below 1 kHz, logarithmic above.
    private const double FSp = 200.0 / 3;
    private const double MinLogHz = 1000.0;
    private const double MinLogMel = MinLogHz / FSp;
    private static readonly double LogStep = Math.Log(6.4) / 27.0;

    private static double HzToMel(double hz)
        => hz >= MinLogHz ? MinLogMel + Math.Log(hz / MinLogHz) / LogStep : hz / FSp;

    private static double MelToHz(double mel)
        => mel >= MinLogMel ? MinLogHz * Math.Exp(LogStep * (mel - MinLogMel)) : FSp * mel;
}
