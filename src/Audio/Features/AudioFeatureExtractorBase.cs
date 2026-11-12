using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
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
/// <para><b>For Beginners:</b> for provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public abstract class AudioFeatureExtractorBase<T> : IAudioFeatureExtractor<T>
{
    /// <summary>
    /// Gets the hardware-accelerated computation engine for vectorized operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

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
        var audioTensor = new Tensor<T>([audio.Length], audio);

        var features = Extract(audioTensor);

        // Convert to Matrix [frames, featureDim]
        return features.ToMatrix();
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
        return AudioEngineOps.SymmetricHannWindow<T>(Engine, length).ToArray();
    }

    /// <summary>
    /// Creates a Hamming window of the specified length.
    /// </summary>
    /// <param name="length">The window length.</param>
    /// <returns>The Hamming window coefficients.</returns>
    protected T[] CreateHammingWindow(int length)
    {
        // 0.54 - 0.46 cos(2 pi i / (length - 1)); a one-sample window is defined as [1] (NumPy/SciPy
        // hamming(1)), not the 0/0 the formula would give.
        if (length < 0) throw new OverflowException("Window length must be non-negative.");
        if (length == 0) return new T[0];
        if (length == 1) return new[] { NumOps.One };
        return Engine.TensorHammingWindow<T>(length, periodic: false).ToArray();
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
        // window[i] * audio[startIndex + i] where the sample exists, else zero.
        int length = window.Length;
        int first = Math.Max(0, startIndex);
        int last = Math.Min(audio.Length, startIndex + length);
        if (last <= first)
            return new T[length];

        int count = last - first;
        var product = Engine.TensorMultiply(
            Engine.TensorSlice(new Tensor<T>(audio, [audio.Length]), [first], [count]),
            Engine.TensorSlice(new Tensor<T>(window, [length]), [first - startIndex], [count]));

        var parts = new List<Tensor<T>>(3);
        if (first > startIndex) parts.Add(new Tensor<T>([first - startIndex]));
        parts.Add(product);
        if (startIndex + length > last) parts.Add(new Tensor<T>([startIndex + length - last]));
        return (parts.Count == 1 ? product : Engine.Concat(parts, 0)).ToArray();
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

        // Create evenly spaced mel points, convert to Hz and then to FFT bin indices
        var binIndices = new int[numMels + 2];
        for (int i = 0; i < binIndices.Length; i++)
        {
            double melPoint = melMin + (melMax - melMin) * i / (numMels + 1);
            double hz = MelToHz(melPoint);
            binIndices[i] = (int)Math.Floor((fftSize + 1) * hz / sampleRate);
        }

        if (numMels <= 0 || numBins <= 0)
            return filterbank;

        // Triangular filters: rising (k - start) / (center - start) on [start, center), falling
        // (end - k) / (end - center) on [center, end).
        var edges = new int[numMels * 3];
        for (int m = 0; m < numMels; m++)
        {
            edges[m * 3] = binIndices[m];
            edges[m * 3 + 1] = binIndices[m + 1];
            edges[m * 3 + 2] = binIndices[m + 2];
        }
        var edgeTensor = new Tensor<int>(edges, [numMels, 3]).Cast<T>();
        var shape = new[] { numMels, numBins };
        var start = Engine.TensorBroadcastTo(Engine.TensorSlice(edgeTensor, [0, 0], [numMels, 1]), shape);
        var center = Engine.TensorBroadcastTo(Engine.TensorSlice(edgeTensor, [0, 1], [numMels, 1]), shape);
        var end = Engine.TensorBroadcastTo(Engine.TensorSlice(edgeTensor, [0, 2], [numMels, 1]), shape);
        var bin = Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorArange<T>(0, numBins, 1), [1, numBins]), shape);

        var zero = new Tensor<T>(shape);
        var rising = Engine.TensorMultiply(
            Engine.TensorGreaterEqual(bin, start), Engine.TensorLessThan(bin, center));
        var falling = Engine.TensorMultiply(
            Engine.TensorGreaterEqual(bin, center), Engine.TensorLessThan(bin, end));
        var weights = Engine.TensorWhere(
            falling,
            Engine.TensorDivide(Engine.TensorSubtract(end, bin), Engine.TensorSubtract(end, center)),
            Engine.TensorWhere(
                rising,
                Engine.TensorDivide(Engine.TensorSubtract(bin, start), Engine.TensorSubtract(center, start)),
                zero));

        var flat = weights.ToArray();
        int offset = 0;
        for (int m = 0; m < numMels; m++)
        {
            for (int k = 0; k < numBins; k++)
                filterbank[m, k] = flat[offset++];
        }

        return filterbank;
    }
}
