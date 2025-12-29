using System;
using System.Numerics;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Features;

/// <summary>
/// Constant-Q Transform (CQT) for music analysis with logarithmic frequency resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Unlike the FFT which has linear frequency spacing, the CQT uses logarithmic spacing
/// where each frequency bin is a constant ratio (Q factor) above the previous one.
/// This matches how humans perceive pitch: octaves are equally spaced.
/// </para>
/// <para><b>For Beginners:</b> The CQT is perfect for music analysis because:
/// <list type="bullet">
/// <item>Musical notes are logarithmically spaced (each octave doubles frequency)</item>
/// <item>Low notes get wide bins, high notes get narrow bins (matches perception)</item>
/// <item>Makes chord/key detection much easier than FFT</item>
/// </list>
///
/// Usage:
/// <code>
/// var cqt = new ConstantQTransform&lt;float&gt;(
///     sampleRate: 22050,
///     fMin: 32.7, // C1
///     binsPerOctave: 12, // Semitones
///     numOctaves: 7);
///
/// var audio = LoadAudio("music.wav");
/// var cqtSpectrum = cqt.Transform(audio);
/// // cqtSpectrum has shape [time_frames, num_bins] where num_bins = 12 * 7 = 84
/// </code>
/// </para>
/// </remarks>
public class ConstantQTransform<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _sampleRate;
    private readonly double _fMin;
    private readonly int _binsPerOctave;
    private readonly int _numOctaves;
    private readonly int _hopLength;
    private readonly int _numBins;
    private readonly double[] _frequencies;
    private readonly int[] _windowLengths;
    private readonly Complex[,] _kernelBank;
    private readonly double _q;

    /// <summary>
    /// Gets the number of frequency bins in the CQT output.
    /// </summary>
    public int NumBins => _numBins;

    /// <summary>
    /// Gets the center frequencies for each bin.
    /// </summary>
    public double[] Frequencies => _frequencies;

    /// <summary>
    /// Gets the sample rate.
    /// </summary>
    public int SampleRate => _sampleRate;

    /// <summary>
    /// Gets the Q factor (quality factor) for this CQT.
    /// </summary>
    public double QFactor => _q;

    /// <summary>
    /// Creates a new Constant-Q Transform instance.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate in Hz.</param>
    /// <param name="fMin">Minimum frequency in Hz (default: C1 = 32.7 Hz).</param>
    /// <param name="binsPerOctave">Number of bins per octave (default: 12 for semitones).</param>
    /// <param name="numOctaves">Number of octaves to cover (default: 7).</param>
    /// <param name="hopLength">Hop length between frames (default: 512).</param>
    /// <param name="windowType">Window type for analysis (default: Hann).</param>
    public ConstantQTransform(
        int sampleRate = 22050,
        double fMin = 32.70,
        int binsPerOctave = 12,
        int numOctaves = 7,
        int hopLength = 512,
        WindowType windowType = WindowType.Hann)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sampleRate = sampleRate;
        _fMin = fMin;
        _binsPerOctave = binsPerOctave;
        _numOctaves = numOctaves;
        _hopLength = hopLength;
        _numBins = binsPerOctave * numOctaves;

        // Q factor: quality factor determines the filter bandwidth
        // For equal-tempered tuning: Q = 1 / (2^(1/binsPerOctave) - 1)
        _q = 1.0 / (Math.Pow(2.0, 1.0 / binsPerOctave) - 1);

        // Calculate center frequencies for each bin
        _frequencies = new double[_numBins];
        for (int k = 0; k < _numBins; k++)
        {
            _frequencies[k] = fMin * Math.Pow(2.0, (double)k / binsPerOctave);
        }

        // Calculate window lengths for each bin
        _windowLengths = new int[_numBins];
        for (int k = 0; k < _numBins; k++)
        {
            _windowLengths[k] = (int)Math.Ceiling(_q * sampleRate / _frequencies[k]);
        }

        // Build the CQT kernel bank
        _kernelBank = BuildKernelBank(windowType);
    }

    private Complex[,] BuildKernelBank(WindowType windowType)
    {
        // Find maximum window length
        int maxWindowLength = 0;
        for (int k = 0; k < _numBins; k++)
        {
            if (_windowLengths[k] > maxWindowLength)
                maxWindowLength = _windowLengths[k];
        }

        // Create kernel bank
        var kernelBank = new Complex[_numBins, maxWindowLength];

        for (int k = 0; k < _numBins; k++)
        {
            int N = _windowLengths[k];
            double freq = _frequencies[k];
            double[] window = CreateWindow(N, windowType);

            for (int n = 0; n < N; n++)
            {
                // Complex sinusoid * window, normalized
                double phase = 2.0 * Math.PI * freq * n / _sampleRate;
                double normalizer = 1.0 / N;
                double windowValue = window[n];

                kernelBank[k, n] = new Complex(
                    Math.Cos(phase) * windowValue * normalizer,
                    -Math.Sin(phase) * windowValue * normalizer);
            }
        }

        return kernelBank;
    }

    private static double[] CreateWindow(int length, WindowType windowType)
    {
        var window = new double[length];

        for (int n = 0; n < length; n++)
        {
            window[n] = windowType switch
            {
                WindowType.Hann => 0.5 * (1 - Math.Cos(2 * Math.PI * n / (length - 1))),
                WindowType.Hamming => 0.54 - 0.46 * Math.Cos(2 * Math.PI * n / (length - 1)),
                WindowType.Blackman => 0.42 - 0.5 * Math.Cos(2 * Math.PI * n / (length - 1))
                    + 0.08 * Math.Cos(4 * Math.PI * n / (length - 1)),
                WindowType.Rectangular => 1.0,
                _ => 0.5 * (1 - Math.Cos(2 * Math.PI * n / (length - 1)))
            };
        }

        return window;
    }

    /// <summary>
    /// Computes the Constant-Q Transform of an audio signal.
    /// </summary>
    /// <param name="audio">Input audio waveform.</param>
    /// <returns>CQT magnitude spectrogram with shape [time_frames, num_bins].</returns>
    public Tensor<T> Transform(Tensor<T> audio)
    {
        int numSamples = audio.Shape[0];
        int numFrames = (numSamples - _windowLengths[_numBins - 1]) / _hopLength + 1;
        numFrames = Math.Max(1, numFrames);

        var result = new Tensor<T>([numFrames, _numBins]);

        // Convert audio to double for processing
        var audioDouble = new double[numSamples];
        for (int i = 0; i < numSamples; i++)
        {
            audioDouble[i] = _numOps.ToDouble(audio[i]);
        }

        // For each time frame
        for (int frame = 0; frame < numFrames; frame++)
        {
            int frameStart = frame * _hopLength;

            // For each frequency bin
            for (int k = 0; k < _numBins; k++)
            {
                int windowLen = _windowLengths[k];

                // Compute inner product with kernel
                Complex sum = Complex.Zero;
                for (int n = 0; n < windowLen && (frameStart + n) < numSamples; n++)
                {
                    int sampleIdx = frameStart + n;
                    sum += audioDouble[sampleIdx] * _kernelBank[k, n];
                }

                // Store magnitude
                result[frame, k] = _numOps.FromDouble(sum.Magnitude);
            }
        }

        return result;
    }

    /// <summary>
    /// Computes the complex CQT (with phase information).
    /// </summary>
    /// <param name="audio">Input audio waveform.</param>
    /// <returns>Complex CQT with shape [time_frames, num_bins, 2] where last dim is [real, imag].</returns>
    public Tensor<T> TransformComplex(Tensor<T> audio)
    {
        int numSamples = audio.Shape[0];
        int numFrames = (numSamples - _windowLengths[_numBins - 1]) / _hopLength + 1;
        numFrames = Math.Max(1, numFrames);

        var result = new Tensor<T>([numFrames, _numBins, 2]);

        var audioDouble = new double[numSamples];
        for (int i = 0; i < numSamples; i++)
        {
            audioDouble[i] = _numOps.ToDouble(audio[i]);
        }

        for (int frame = 0; frame < numFrames; frame++)
        {
            int frameStart = frame * _hopLength;

            for (int k = 0; k < _numBins; k++)
            {
                int windowLen = _windowLengths[k];
                Complex sum = Complex.Zero;

                for (int n = 0; n < windowLen && (frameStart + n) < numSamples; n++)
                {
                    int sampleIdx = frameStart + n;
                    sum += audioDouble[sampleIdx] * _kernelBank[k, n];
                }

                result[frame, k, 0] = _numOps.FromDouble(sum.Real);
                result[frame, k, 1] = _numOps.FromDouble(sum.Imaginary);
            }
        }

        return result;
    }

    /// <summary>
    /// Computes CQT with power spectrum (magnitude squared).
    /// </summary>
    /// <param name="audio">Input audio waveform.</param>
    /// <param name="power">Power exponent (default: 2.0 for power spectrum).</param>
    /// <returns>Power CQT spectrogram.</returns>
    public Tensor<T> TransformPower(Tensor<T> audio, double power = 2.0)
    {
        var cqt = Transform(audio);

        for (int i = 0; i < cqt.Shape[0]; i++)
        {
            for (int j = 0; j < cqt.Shape[1]; j++)
            {
                double mag = _numOps.ToDouble(cqt[i, j]);
                cqt[i, j] = _numOps.FromDouble(Math.Pow(mag, power));
            }
        }

        return cqt;
    }

    /// <summary>
    /// Computes CQT in decibels (log scale).
    /// </summary>
    /// <param name="audio">Input audio waveform.</param>
    /// <param name="refValue">Reference value for dB conversion.</param>
    /// <param name="minDb">Minimum dB value (floor).</param>
    /// <returns>CQT spectrogram in decibels.</returns>
    public Tensor<T> TransformDb(Tensor<T> audio, double refValue = 1.0, double minDb = -80.0)
    {
        var cqt = TransformPower(audio, 2.0);

        for (int i = 0; i < cqt.Shape[0]; i++)
        {
            for (int j = 0; j < cqt.Shape[1]; j++)
            {
                double power = _numOps.ToDouble(cqt[i, j]);
                double db = 10.0 * Math.Log10(Math.Max(power, 1e-10) / (refValue * refValue));
                db = Math.Max(db, minDb);
                cqt[i, j] = _numOps.FromDouble(db);
            }
        }

        return cqt;
    }

    /// <summary>
    /// Gets the MIDI note number for a given bin index.
    /// </summary>
    /// <param name="binIndex">The CQT bin index.</param>
    /// <returns>The corresponding MIDI note number.</returns>
    public double GetMidiNote(int binIndex)
    {
        double frequency = _frequencies[binIndex];
        return 69 + 12 * (Math.Log(frequency / 440.0) / Math.Log(2));
    }

    /// <summary>
    /// Gets the note name for a given bin index.
    /// </summary>
    /// <param name="binIndex">The CQT bin index.</param>
    /// <returns>The note name (e.g., "C4", "A#3").</returns>
    public string GetNoteName(int binIndex)
    {
        string[] noteNames = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
        int midiNote = (int)Math.Round(GetMidiNote(binIndex));
        int noteIndex = midiNote % 12;
        int octave = (midiNote / 12) - 1;
        return $"{noteNames[noteIndex]}{octave}";
    }
}

/// <summary>
/// Window types for spectral analysis.
/// </summary>
public enum WindowType
{
    /// <summary>Rectangular window (no tapering).</summary>
    Rectangular,

    /// <summary>Hann window (cosine tapering).</summary>
    Hann,

    /// <summary>Hamming window (raised cosine).</summary>
    Hamming,

    /// <summary>Blackman window (higher sidelobe attenuation).</summary>
    Blackman
}
