using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Diffusion.Audio;

/// <summary>
/// Short-Time Fourier Transform (STFT) for analyzing audio signals over time.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The STFT breaks a signal into short overlapping segments and computes the
/// Fourier transform of each segment. This reveals how the frequency content
/// of a signal changes over time.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio signals like music or speech change over time.
/// While a regular FFT tells you which frequencies are in the entire signal,
/// it doesn't tell you WHEN those frequencies occur.
///
/// The STFT solves this by:
/// 1. Cutting the audio into small overlapping pieces (frames)
/// 2. Applying a window function to each frame (reduces edge artifacts)
/// 3. Computing FFT on each windowed frame
/// 4. Stacking the results to form a spectrogram (time vs. frequency)
///
/// Usage:
/// ```csharp
/// var stft = new ShortTimeFourierTransform&lt;float&gt;(nFft: 2048, hopLength: 512);
/// var spectrogram = stft.Forward(audioSignal);
/// // spectrogram.Shape = [numFrames, nFft/2 + 1] (complex values)
///
/// // To reconstruct audio from spectrogram:
/// var reconstructed = stft.Inverse(spectrogram);
/// ```
/// </para>
/// </remarks>
public class ShortTimeFourierTransform<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// FFT size (number of frequency bins).
    /// </summary>
    private readonly int _nFft;

    /// <summary>
    /// Number of samples between successive frames.
    /// </summary>
    private readonly int _hopLength;

    /// <summary>
    /// Length of the window (defaults to nFft).
    /// </summary>
    private readonly int _windowLength;

    /// <summary>
    /// Window function coefficients.
    /// </summary>
    private readonly T[] _window;

    /// <summary>
    /// Whether to center the signal by padding.
    /// </summary>
    private readonly bool _center;

    /// <summary>
    /// Padding mode when centering.
    /// </summary>
    private readonly PaddingMode _padMode;

    /// <summary>
    /// FFT implementation.
    /// </summary>
    private readonly FastFourierTransform<T> _fft;

    /// <summary>
    /// Gets the FFT size.
    /// </summary>
    public int NFft => _nFft;

    /// <summary>
    /// Gets the hop length.
    /// </summary>
    public int HopLength => _hopLength;

    /// <summary>
    /// Gets the number of frequency bins (nFft / 2 + 1).
    /// </summary>
    public int NumFrequencyBins => _nFft / 2 + 1;

    /// <summary>
    /// Initializes a new STFT processor.
    /// </summary>
    /// <param name="nFft">FFT size (default: 2048). Should be a power of 2.</param>
    /// <param name="hopLength">Hop length between frames (default: nFft/4).</param>
    /// <param name="windowLength">Window length (default: nFft).</param>
    /// <param name="windowType">Type of window function (default: Hann).</param>
    /// <param name="center">Whether to pad signal so frames are centered (default: true).</param>
    /// <param name="padMode">Padding mode when centering (default: Reflect).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - nFft: Determines frequency resolution. Larger = more frequency detail but less time detail
    /// - hopLength: How much to slide between frames. Smaller = more overlap = smoother output
    ///   Common: hopLength = nFft/4 gives 75% overlap
    /// - window: Reduces spectral leakage. Hann is the most common for audio
    /// </para>
    /// </remarks>
    public ShortTimeFourierTransform(
        int nFft = 2048,
        int? hopLength = null,
        int? windowLength = null,
        WindowType windowType = WindowType.Hann,
        bool center = true,
        PaddingMode padMode = PaddingMode.Reflect)
    {
        if (nFft <= 0)
            throw new ArgumentOutOfRangeException(nameof(nFft), "FFT size must be positive.");
        if ((nFft & (nFft - 1)) != 0)
            throw new ArgumentException("FFT size must be a power of 2.", nameof(nFft));

        _nFft = nFft;
        _hopLength = hopLength ?? nFft / 4;
        _windowLength = windowLength ?? nFft;
        _center = center;
        _padMode = padMode;
        _fft = new FastFourierTransform<T>();

        if (_hopLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(hopLength), "Hop length must be positive.");
        if (_windowLength <= 0 || _windowLength > nFft)
            throw new ArgumentOutOfRangeException(nameof(windowLength), "Window length must be positive and <= nFft.");

        // Create window function
        _window = WindowFunctions<T>.Create(windowType, _windowLength);

        // Pad window to nFft if shorter
        if (_windowLength < nFft)
        {
            var paddedWindow = new T[nFft];
            int padStart = (nFft - _windowLength) / 2;
            Array.Copy(_window, 0, paddedWindow, padStart, _windowLength);
            _window = paddedWindow;
        }
    }

    /// <summary>
    /// Computes the Short-Time Fourier Transform of a signal.
    /// </summary>
    /// <param name="signal">Input signal as a tensor [length] or [batch, length].</param>
    /// <returns>Complex spectrogram tensor [numFrames, numFreqs] or [batch, numFrames, numFreqs].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes your audio waveform and produces a spectrogram
    /// showing which frequencies are present at each point in time.
    /// </para>
    /// </remarks>
    public Tensor<Complex<T>> Forward(Tensor<T> signal)
    {
        if (signal == null)
            throw new ArgumentNullException(nameof(signal));

        // Handle batched input
        if (signal.Shape.Length == 2)
        {
            return ForwardBatched(signal);
        }

        // Flatten to 1D if needed
        var flatSignal = signal.Shape.Length == 1 ? signal.Data : signal.Data;
        return ForwardSingle(flatSignal);
    }

    /// <summary>
    /// Computes STFT for a single signal.
    /// </summary>
    private Tensor<Complex<T>> ForwardSingle(T[] signal)
    {
        T[] paddedSignal;

        if (_center)
        {
            // Pad signal by nFft/2 on each side
            int padLength = _nFft / 2;
            paddedSignal = PadSignal(signal, padLength, padLength, _padMode);
        }
        else
        {
            paddedSignal = signal;
        }

        // Calculate number of frames
        int numFrames = 1 + (paddedSignal.Length - _nFft) / _hopLength;
        int numFreqs = _nFft / 2 + 1;

        var output = new Tensor<Complex<T>>(new[] { numFrames, numFreqs });

        // Process each frame
        for (int frame = 0; frame < numFrames; frame++)
        {
            int startSample = frame * _hopLength;

            // Extract frame and apply window
            var windowedFrame = new Vector<T>(_nFft);
            for (int i = 0; i < _nFft; i++)
            {
                int sampleIdx = startSample + i;
                if (sampleIdx < paddedSignal.Length)
                {
                    windowedFrame[i] = NumOps.Multiply(paddedSignal[sampleIdx], _window[i]);
                }
            }

            // Compute FFT
            var spectrum = _fft.Forward(windowedFrame);

            // Store only positive frequencies (DC to Nyquist)
            for (int f = 0; f < numFreqs; f++)
            {
                int outputIdx = frame * numFreqs + f;
                output.Data[outputIdx] = spectrum[f];
            }
        }

        return output;
    }

    /// <summary>
    /// Computes STFT for a batch of signals.
    /// </summary>
    private Tensor<Complex<T>> ForwardBatched(Tensor<T> signals)
    {
        int batchSize = signals.Shape[0];
        int signalLength = signals.Shape[1];

        // Calculate output dimensions
        int padLength = _center ? _nFft / 2 : 0;
        int paddedLength = signalLength + 2 * padLength;
        int numFrames = 1 + (paddedLength - _nFft) / _hopLength;
        int numFreqs = _nFft / 2 + 1;

        var output = new Tensor<Complex<T>>(new[] { batchSize, numFrames, numFreqs });

        for (int b = 0; b < batchSize; b++)
        {
            // Extract single signal
            var singleSignal = new T[signalLength];
            for (int i = 0; i < signalLength; i++)
            {
                singleSignal[i] = signals.Data[b * signalLength + i];
            }

            // Compute STFT
            var singleOutput = ForwardSingle(singleSignal);

            // Copy to batched output
            int offset = b * numFrames * numFreqs;
            Array.Copy(singleOutput.Data, 0, output.Data, offset, numFrames * numFreqs);
        }

        return output;
    }

    /// <summary>
    /// Computes the Inverse Short-Time Fourier Transform (overlap-add reconstruction).
    /// </summary>
    /// <param name="spectrogram">Complex spectrogram [numFrames, numFreqs] or [batch, numFrames, numFreqs].</param>
    /// <param name="length">Expected output length (optional, otherwise computed from spectrogram).</param>
    /// <returns>Reconstructed signal tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes a spectrogram and converts it back to an audio waveform.
    /// It uses the "overlap-add" method, where each frame is inverse-FFT'd and the overlapping
    /// portions are added together.
    ///
    /// Note: Perfect reconstruction requires the same STFT parameters used for analysis.
    /// </para>
    /// </remarks>
    public Tensor<T> Inverse(Tensor<Complex<T>> spectrogram, int? length = null)
    {
        if (spectrogram == null)
            throw new ArgumentNullException(nameof(spectrogram));

        // Handle batched input
        if (spectrogram.Shape.Length == 3)
        {
            return InverseBatched(spectrogram, length);
        }

        return InverseSingle(spectrogram, length);
    }

    /// <summary>
    /// Computes ISTFT for a single spectrogram.
    /// </summary>
    private Tensor<T> InverseSingle(Tensor<Complex<T>> spectrogram, int? targetLength)
    {
        int numFrames = spectrogram.Shape[0];
        int numFreqs = spectrogram.Shape[1];

        // Compute output length
        int outputLength = _nFft + (numFrames - 1) * _hopLength;
        if (_center)
        {
            outputLength -= _nFft; // Remove center padding
        }
        outputLength = targetLength ?? outputLength;

        // Allocate output and window sum for normalization
        var output = new T[outputLength + _nFft]; // Extra padding for overlap-add
        var windowSum = new T[outputLength + _nFft];

        // Process each frame
        for (int frame = 0; frame < numFrames; frame++)
        {
            // Extract spectrum for this frame
            var spectrum = new Vector<Complex<T>>(_nFft);

            // Copy positive frequencies
            for (int f = 0; f < numFreqs; f++)
            {
                spectrum[f] = spectrogram.Data[frame * numFreqs + f];
            }

            // Reconstruct negative frequencies (conjugate symmetry)
            for (int f = 1; f < _nFft / 2; f++)
            {
                spectrum[_nFft - f] = spectrum[f].Conjugate();
            }

            // Inverse FFT
            var timeDomain = _fft.Inverse(spectrum);

            // Apply window and overlap-add
            int startSample = frame * _hopLength;
            if (_center)
            {
                startSample -= _nFft / 2;
            }

            for (int i = 0; i < _nFft; i++)
            {
                int idx = startSample + i;
                if (idx >= 0 && idx < output.Length)
                {
                    // Window the reconstructed frame
                    var windowed = NumOps.Multiply(timeDomain[i], _window[i]);
                    output[idx] = NumOps.Add(output[idx], windowed);

                    // Accumulate window squared for normalization
                    var windowSquared = NumOps.Multiply(_window[i], _window[i]);
                    windowSum[idx] = NumOps.Add(windowSum[idx], windowSquared);
                }
            }
        }

        // Normalize by window sum (COLA normalization)
        var result = new T[outputLength];
        T epsilon = NumOps.FromDouble(1e-8);

        for (int i = 0; i < outputLength; i++)
        {
            var denom = NumOps.Add(windowSum[i], epsilon);
            result[i] = NumOps.Divide(output[i], denom);
        }

        return new Tensor<T>(result, new[] { outputLength });
    }

    /// <summary>
    /// Computes ISTFT for a batch of spectrograms.
    /// </summary>
    private Tensor<T> InverseBatched(Tensor<Complex<T>> spectrograms, int? targetLength)
    {
        int batchSize = spectrograms.Shape[0];
        int numFrames = spectrograms.Shape[1];
        int numFreqs = spectrograms.Shape[2];

        // Compute output length
        int outputLength = _nFft + (numFrames - 1) * _hopLength;
        if (_center)
        {
            outputLength -= _nFft;
        }
        outputLength = targetLength ?? outputLength;

        var output = new Tensor<T>(new[] { batchSize, outputLength });

        for (int b = 0; b < batchSize; b++)
        {
            // Extract single spectrogram
            var singleSpec = new Tensor<Complex<T>>(new[] { numFrames, numFreqs });
            int offset = b * numFrames * numFreqs;
            Array.Copy(spectrograms.Data, offset, singleSpec.Data, 0, numFrames * numFreqs);

            // Compute ISTFT
            var singleOutput = InverseSingle(singleSpec, outputLength);

            // Copy to batched output
            int outOffset = b * outputLength;
            Array.Copy(singleOutput.Data, 0, output.Data, outOffset, outputLength);
        }

        return output;
    }

    /// <summary>
    /// Computes the magnitude spectrogram.
    /// </summary>
    /// <param name="signal">Input signal.</param>
    /// <returns>Magnitude spectrogram [numFrames, numFreqs].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The magnitude spectrogram shows how loud each frequency is
    /// at each time, discarding phase information. This is often used for visualization
    /// and audio processing where phase isn't needed.
    /// </para>
    /// </remarks>
    public Tensor<T> Magnitude(Tensor<T> signal)
    {
        var complex = Forward(signal);
        return ComputeMagnitude(complex);
    }

    /// <summary>
    /// Computes the power spectrogram (magnitude squared).
    /// </summary>
    /// <param name="signal">Input signal.</param>
    /// <returns>Power spectrogram [numFrames, numFreqs].</returns>
    public Tensor<T> Power(Tensor<T> signal)
    {
        var magnitude = Magnitude(signal);
        var power = new Tensor<T>(magnitude.Shape);

        for (int i = 0; i < magnitude.Data.Length; i++)
        {
            power.Data[i] = NumOps.Multiply(magnitude.Data[i], magnitude.Data[i]);
        }

        return power;
    }

    /// <summary>
    /// Computes magnitude from complex spectrogram.
    /// </summary>
    private static Tensor<T> ComputeMagnitude(Tensor<Complex<T>> complex)
    {
        var magnitude = new Tensor<T>(complex.Shape);

        for (int i = 0; i < complex.Data.Length; i++)
        {
            magnitude.Data[i] = complex.Data[i].Magnitude;
        }

        return magnitude;
    }

    /// <summary>
    /// Extracts phase from complex spectrogram.
    /// </summary>
    /// <param name="complex">Complex spectrogram.</param>
    /// <returns>Phase tensor in radians.</returns>
    public static Tensor<T> ExtractPhase(Tensor<Complex<T>> complex)
    {
        var phase = new Tensor<T>(complex.Shape);

        for (int i = 0; i < complex.Data.Length; i++)
        {
            phase.Data[i] = complex.Data[i].Phase;
        }

        return phase;
    }

    /// <summary>
    /// Creates complex spectrogram from magnitude and phase.
    /// </summary>
    /// <param name="magnitude">Magnitude tensor.</param>
    /// <param name="phase">Phase tensor in radians.</param>
    /// <returns>Complex spectrogram.</returns>
    public static Tensor<Complex<T>> PolarToComplex(Tensor<T> magnitude, Tensor<T> phase)
    {
        if (!magnitude.Shape.SequenceEqual(phase.Shape))
            throw new ArgumentException("Magnitude and phase must have the same shape.");

        var complex = new Tensor<Complex<T>>(magnitude.Shape);

        for (int i = 0; i < magnitude.Data.Length; i++)
        {
            complex.Data[i] = Complex<T>.FromPolarCoordinates(magnitude.Data[i], phase.Data[i]);
        }

        return complex;
    }

    /// <summary>
    /// Pads a signal according to the specified mode.
    /// </summary>
    private static T[] PadSignal(T[] signal, int padBefore, int padAfter, PaddingMode mode)
    {
        int newLength = signal.Length + padBefore + padAfter;
        var padded = new T[newLength];

        // Copy original signal
        Array.Copy(signal, 0, padded, padBefore, signal.Length);

        // Pad before
        for (int i = 0; i < padBefore; i++)
        {
            int sourceIdx = mode switch
            {
                PaddingMode.Reflect => padBefore - i,
                PaddingMode.Replicate => 0,
                PaddingMode.Zero => -1,
                _ => -1
            };

            padded[i] = sourceIdx >= 0 && sourceIdx < signal.Length
                ? signal[sourceIdx]
                : NumOps.Zero;
        }

        // Pad after
        for (int i = 0; i < padAfter; i++)
        {
            int sourceIdx = mode switch
            {
                PaddingMode.Reflect => signal.Length - 2 - i,
                PaddingMode.Replicate => signal.Length - 1,
                PaddingMode.Zero => -1,
                _ => -1
            };

            padded[padBefore + signal.Length + i] = sourceIdx >= 0 && sourceIdx < signal.Length
                ? signal[sourceIdx]
                : NumOps.Zero;
        }

        return padded;
    }

    /// <summary>
    /// Calculates the number of frames for a given signal length.
    /// </summary>
    /// <param name="signalLength">Length of the input signal.</param>
    /// <returns>Number of STFT frames.</returns>
    public int CalculateNumFrames(int signalLength)
    {
        int padLength = _center ? _nFft / 2 : 0;
        int paddedLength = signalLength + 2 * padLength;
        return 1 + (paddedLength - _nFft) / _hopLength;
    }

    /// <summary>
    /// Calculates signal length from number of frames.
    /// </summary>
    /// <param name="numFrames">Number of STFT frames.</param>
    /// <returns>Approximate signal length.</returns>
    public int CalculateSignalLength(int numFrames)
    {
        int length = _nFft + (numFrames - 1) * _hopLength;
        if (_center)
        {
            length -= _nFft;
        }
        return length;
    }
}

/// <summary>
/// Padding mode for STFT centering.
/// </summary>
public enum PaddingMode
{
    /// <summary>
    /// Reflect padding: abcde -> edcba|abcde|edcba
    /// </summary>
    Reflect,

    /// <summary>
    /// Replicate padding: abcde -> aaaaa|abcde|eeeee
    /// </summary>
    Replicate,

    /// <summary>
    /// Zero padding: abcde -> 00000|abcde|00000
    /// </summary>
    Zero
}
