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
    private readonly double[] _window;
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

        // librosa 'hann' = scipy get_window('hann', N, fftbins=True): periodic, zero-padded to fftSize centred.
        _window = new double[fftSize];
        int offset = (fftSize - windowLength) / 2;
        for (int i = 0; i < windowLength; i++)
            _window[offset + i] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / windowLength);

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

    /// <summary>Number of frames for a signal of <paramref name="sampleCount"/> samples.</summary>
    public int FrameCount(int sampleCount) => 1 + sampleCount / HopLength;

    /// <summary>The STFT magnitude, <c>[frames, fftSize/2 + 1]</c>.</summary>
    public double[,] Magnitude(double[] audio)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        int pad = FftSize / 2;
        if (audio.Length <= pad)
            throw new ArgumentException($"Reflect padding needs more than {pad} samples.", nameof(audio));

        var padded = new double[audio.Length + 2 * pad];
        for (int i = 0; i < padded.Length; i++)
        {
            int j = i - pad;
            if (j < 0) j = -j;
            else if (j >= audio.Length) j = 2 * (audio.Length - 1) - j;
            padded[i] = audio[j];
        }

        int frames = FrameCount(audio.Length);
        int bins = FftSize / 2 + 1;
        var re = new double[frames * FftSize];
        for (int f = 0; f < frames; f++)
            for (int k = 0; k < FftSize; k++)
                re[f * FftSize + k] = padded[f * HopLength + k] * _window[k];

        var tRe = new Tensor<double>(new[] { frames, FftSize }, new Vector<double>(re));
        var tIm = new Tensor<double>(new[] { frames, FftSize });
        AiDotNet.Tensors.Engines.AiDotNetEngine.Current.FFT(tRe, tIm, out var outRe, out var outIm);
        var sRe = outRe.AsSpan();
        var sIm = outIm.AsSpan();

        var magnitude = new double[frames, bins];
        for (int f = 0; f < frames; f++)
            for (int k = 0; k < bins; k++)
            {
                double a = sRe[f * FftSize + k], b = sIm[f * FftSize + k];
                magnitude[f, k] = Math.Sqrt(a * a + b * b);
            }
        return magnitude;
    }

    /// <summary>The log mel spectrogram, <c>[frames, melChannels]</c>.</summary>
    public double[,] LogMel(double[] audio) => LogMel(Magnitude(audio));

    /// <summary>The log mel spectrogram of an STFT magnitude.</summary>
    public double[,] LogMel(double[,] magnitude)
    {
        int frames = magnitude.GetLength(0), bins = magnitude.GetLength(1);
        var mel = new double[frames, MelChannels];
        for (int f = 0; f < frames; f++)
            for (int m = 0; m < MelChannels; m++)
            {
                double acc = 0;
                for (int k = 0; k < bins; k++) acc += _melBasis[m, k] * magnitude[f, k];
                mel[f, m] = Math.Log(Math.Max(acc, ClipValue));
            }
        return mel;
    }

    /// <summary>FastSpeech 2's frame energy: the L2 norm of each STFT magnitude frame.</summary>
    public double[] Energy(double[,] magnitude)
    {
        int frames = magnitude.GetLength(0), bins = magnitude.GetLength(1);
        var energy = new double[frames];
        for (int f = 0; f < frames; f++)
        {
            double acc = 0;
            for (int k = 0; k < bins; k++) acc += magnitude[f, k] * magnitude[f, k];
            energy[f] = Math.Sqrt(acc);
        }
        return energy;
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
