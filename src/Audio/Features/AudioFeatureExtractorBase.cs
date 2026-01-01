using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Features;

/// <summary>
/// Base class for audio feature extractors providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This base class provides:
/// <list type="bullet">
/// <item>Common audio processing utilities (windowing, framing)</item>
/// <item>Numeric operations through INumericOperations&lt;T&gt;</item>
/// <item>Sample rate and FFT configuration</item>
/// </list>
/// </para>
/// </remarks>
public abstract class AudioFeatureExtractorBase<T> : IAudioFeatureExtractor<T>
{
    /// <summary>
    /// Numeric operations for the current type.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// The audio feature extraction options.
    /// </summary>
    protected readonly AudioFeatureOptions Options;

    /// <inheritdoc/>
    public abstract string Name { get; }

    /// <inheritdoc/>
    public abstract int FeatureDimension { get; }

    /// <inheritdoc/>
    public int SampleRate => Options.SampleRate;

    /// <summary>
    /// Gets the FFT size.
    /// </summary>
    protected int FftSize => Options.FftSize;

    /// <summary>
    /// Gets the hop length between frames.
    /// </summary>
    protected int HopLength => Options.HopLength;

    /// <summary>
    /// Gets the window length.
    /// </summary>
    protected int WindowLength => Options.EffectiveWindowLength;

    /// <summary>
    /// Initializes a new instance of the AudioFeatureExtractorBase class.
    /// </summary>
    /// <param name="options">The feature extraction options.</param>
    protected AudioFeatureExtractorBase(AudioFeatureOptions? options = null)
    {
        Options = options ?? new AudioFeatureOptions();
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public abstract Tensor<T> Extract(Tensor<T> audio);

    /// <inheritdoc/>
    public virtual Matrix<T> Extract(Vector<T> audio)
    {
        var audioTensor = new Tensor<T>([audio.Length]);
        for (int i = 0; i < audio.Length; i++)
        {
            audioTensor[i] = audio[i];
        }

        var features = Extract(audioTensor);

        // Convert to Matrix
        int frames = features.Shape[0];
        int featureDim = features.Shape[1];
        var matrix = new Matrix<T>(frames, featureDim);

        for (int f = 0; f < frames; f++)
        {
            for (int d = 0; d < featureDim; d++)
            {
                matrix[f, d] = features[f, d];
            }
        }

        return matrix;
    }

    /// <inheritdoc/>
    public virtual Task<Tensor<T>> ExtractAsync(Tensor<T> audio, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Extract(audio), cancellationToken);
    }

    /// <summary>
    /// Computes the number of frames that will be produced for a given audio length.
    /// </summary>
    /// <param name="audioLength">The number of audio samples.</param>
    /// <returns>The number of frames.</returns>
    protected int ComputeNumFrames(int audioLength)
    {
        if (Options.CenterPad)
        {
            return (audioLength + HopLength - 1) / HopLength;
        }

        return Math.Max(0, (audioLength - WindowLength) / HopLength + 1);
    }

    /// <summary>
    /// Creates a Hann window of the specified length.
    /// </summary>
    /// <param name="length">The window length.</param>
    /// <returns>The Hann window coefficients.</returns>
    protected T[] CreateHannWindow(int length)
    {
        var window = new T[length];

        for (int i = 0; i < length; i++)
        {
            double coefficient = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (length - 1)));
            window[i] = NumOps.FromDouble(coefficient);
        }

        return window;
    }

    /// <summary>
    /// Creates a Hamming window of the specified length.
    /// </summary>
    /// <param name="length">The window length.</param>
    /// <returns>The Hamming window coefficients.</returns>
    protected T[] CreateHammingWindow(int length)
    {
        var window = new T[length];

        for (int i = 0; i < length; i++)
        {
            double coefficient = 0.54 - 0.46 * Math.Cos(2 * Math.PI * i / (length - 1));
            window[i] = NumOps.FromDouble(coefficient);
        }

        return window;
    }

    /// <summary>
    /// Extracts a single frame from the audio signal.
    /// </summary>
    /// <param name="audio">The audio data.</param>
    /// <param name="startIndex">The start index of the frame.</param>
    /// <param name="window">The window function to apply.</param>
    /// <returns>The windowed frame.</returns>
    protected T[] ExtractFrame(T[] audio, int startIndex, T[] window)
    {
        var frame = new T[window.Length];

        for (int i = 0; i < window.Length; i++)
        {
            int sampleIndex = startIndex + i;

            if (sampleIndex >= 0 && sampleIndex < audio.Length)
            {
                frame[i] = NumOps.Multiply(audio[sampleIndex], window[i]);
            }
            else
            {
                frame[i] = NumOps.Zero;
            }
        }

        return frame;
    }

    /// <summary>
    /// Pads audio for center-aligned frames.
    /// </summary>
    /// <param name="audio">The audio data.</param>
    /// <returns>The padded audio.</returns>
    protected T[] PadAudioCenter(T[] audio)
    {
        int padAmount = WindowLength / 2;
        var padded = new T[audio.Length + 2 * padAmount];

        // Initialize with zeros
        for (int i = 0; i < padded.Length; i++)
        {
            padded[i] = NumOps.Zero;
        }

        // Copy audio to center
        Array.Copy(audio, 0, padded, padAmount, audio.Length);

        return padded;
    }

    /// <summary>
    /// Converts frequency in Hz to mel scale.
    /// </summary>
    /// <param name="hz">Frequency in Hz.</param>
    /// <returns>Frequency in mel scale.</returns>
    protected static double HzToMel(double hz)
    {
        return 2595 * Math.Log10(1 + hz / 700);
    }

    /// <summary>
    /// Converts mel scale frequency to Hz.
    /// </summary>
    /// <param name="mel">Frequency in mel scale.</param>
    /// <returns>Frequency in Hz.</returns>
    protected static double MelToHz(double mel)
    {
        return 700 * (Math.Pow(10, mel / 2595) - 1);
    }

    /// <summary>
    /// Creates mel filterbank.
    /// </summary>
    /// <param name="numMels">Number of mel filters.</param>
    /// <param name="fftSize">FFT size.</param>
    /// <param name="sampleRate">Sample rate.</param>
    /// <param name="fMin">Minimum frequency.</param>
    /// <param name="fMax">Maximum frequency.</param>
    /// <returns>The mel filterbank matrix [numMels x (fftSize/2+1)].</returns>
    protected T[,] CreateMelFilterbank(int numMels, int fftSize, int sampleRate, double fMin = 0, double? fMax = null)
    {
        double maxFreq = fMax ?? sampleRate / 2.0;
        int numBins = fftSize / 2 + 1;

        var filterbank = new T[numMels, numBins];

        double melMin = HzToMel(fMin);
        double melMax = HzToMel(maxFreq);

        // Create evenly spaced mel points
        var melPoints = new double[numMels + 2];
        for (int i = 0; i < melPoints.Length; i++)
        {
            melPoints[i] = melMin + (melMax - melMin) * i / (numMels + 1);
        }

        // Convert to Hz and then to FFT bin indices
        var binIndices = new int[melPoints.Length];
        for (int i = 0; i < melPoints.Length; i++)
        {
            double hz = MelToHz(melPoints[i]);
            binIndices[i] = (int)Math.Floor((fftSize + 1) * hz / sampleRate);
        }

        // Create triangular filters
        for (int m = 0; m < numMels; m++)
        {
            int fStart = binIndices[m];
            int fCenter = binIndices[m + 1];
            int fEnd = binIndices[m + 2];

            // Rising slope
            for (int k = fStart; k < fCenter && k < numBins; k++)
            {
                double weight = (double)(k - fStart) / (fCenter - fStart);
                filterbank[m, k] = NumOps.FromDouble(weight);
            }

            // Falling slope
            for (int k = fCenter; k < fEnd && k < numBins; k++)
            {
                double weight = (double)(fEnd - k) / (fEnd - fCenter);
                filterbank[m, k] = NumOps.FromDouble(weight);
            }
        }

        return filterbank;
    }
}
