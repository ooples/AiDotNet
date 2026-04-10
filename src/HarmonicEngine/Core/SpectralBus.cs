using AiDotNet.LinearAlgebra;

namespace AiDotNet.HarmonicEngine.Core;

/// <summary>
/// OFDM-style spectral bus that encodes features onto orthogonal frequency carriers
/// and decodes them back. This replaces the traditional weight matrix (W * x) with a
/// spectral broadcast operation at O(N log N) complexity.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In a traditional neural network, a dense layer multiplies inputs by a weight matrix:
/// output = W * input. This costs O(N^2) for N features. The spectral bus does something fundamentally different:
///
/// 1. Encode: Each input feature is placed as the amplitude of a unique frequency carrier
/// 2. Combine: All carriers are summed into a single composite time-domain signal via inverse FFT
/// 3. This composite signal contains ALL feature information simultaneously
/// 4. Decode: FFT extracts the features back from specific carrier frequencies
///
/// The encode-decode round trip costs O(N log N) via FFT, and between encode and decode,
/// a nonlinearity can create intermodulation products that represent cross-feature interactions.
/// </para>
/// </remarks>
public class SpectralBus<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly INumericOperations<Complex<T>> _complexOps;
    private readonly FastFourierTransform<T> _fft;
    private readonly int[] _carriers;
    private readonly int _fftSize;

    /// <summary>
    /// Gets the carrier frequency bin indices used by this bus.
    /// </summary>
    public int[] Carriers => _carriers;

    /// <summary>
    /// Gets the FFT size used by this bus.
    /// </summary>
    public int FftSize => _fftSize;

    /// <summary>
    /// Gets the number of carriers (feature channels) in this bus.
    /// </summary>
    public int NumCarriers => _carriers.Length;

    /// <summary>
    /// Initializes a new SpectralBus with pre-allocated carrier positions.
    /// </summary>
    /// <param name="carriers">Frequency bin indices for each carrier (from CarrierAllocator).</param>
    /// <param name="fftSize">FFT size. Must be a power of 2 and large enough for all carriers.</param>
    public SpectralBus(int[] carriers, int fftSize)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _complexOps = MathHelper.GetNumericOperations<Complex<T>>();
        _fft = new FastFourierTransform<T>();
        _carriers = carriers;
        _fftSize = fftSize;

        if ((fftSize & (fftSize - 1)) != 0)
            throw new ArgumentException("FFT size must be a power of 2.", nameof(fftSize));

        foreach (int c in carriers)
        {
            if (c < 0 || c >= fftSize / 2)
                throw new ArgumentException(
                    $"Carrier index {c} is out of range for FFT size {fftSize}.", nameof(carriers));
        }
    }

    /// <summary>
    /// Encodes feature amplitudes onto frequency carriers and converts to a time-domain signal via IFFT.
    /// </summary>
    /// <param name="features">Feature amplitudes, one per carrier. Length must equal NumCarriers.</param>
    /// <returns>Time-domain signal of length FftSize containing all features as superimposed carriers.</returns>
    public Vector<T> Encode(Vector<T> features)
    {
        if (features.Length != _carriers.Length)
            throw new ArgumentException(
                $"Feature count ({features.Length}) must equal carrier count ({_carriers.Length}).");

        // Build frequency-domain representation
        var spectrum = new Vector<Complex<T>>(_fftSize);
        var zero = new Complex<T>(_numOps.Zero, _numOps.Zero);

        for (int i = 0; i < _fftSize; i++)
        {
            spectrum[i] = zero;
        }

        // Place each feature as the amplitude of its assigned carrier
        for (int i = 0; i < _carriers.Length; i++)
        {
            int bin = _carriers[i];
            spectrum[bin] = new Complex<T>(features[i], _numOps.Zero);

            // Mirror to negative frequency for real-valued output
            int mirrorBin = _fftSize - bin;
            if (mirrorBin != bin && mirrorBin < _fftSize)
            {
                spectrum[mirrorBin] = new Complex<T>(features[i], _numOps.Zero);
            }
        }

        // IFFT to get real-valued time-domain signal
        return _fft.Inverse(spectrum);
    }

    /// <summary>
    /// Decodes feature amplitudes from a time-domain signal by reading carrier positions in the FFT.
    /// </summary>
    /// <param name="signal">Time-domain signal of length FftSize.</param>
    /// <returns>Feature amplitudes, one per carrier.</returns>
    public Vector<T> Decode(Vector<T> signal)
    {
        var spectrum = _fft.Forward(signal);
        var features = new Vector<T>(_carriers.Length);

        for (int i = 0; i < _carriers.Length; i++)
        {
            // Read the magnitude at the carrier frequency
            features[i] = spectrum[_carriers[i]].Magnitude;
        }

        return features;
    }

    /// <summary>
    /// Decodes the full complex spectrum at carrier positions (preserving phase information).
    /// </summary>
    /// <param name="signal">Time-domain signal of length FftSize.</param>
    /// <returns>Complex values at carrier frequencies.</returns>
    public Vector<Complex<T>> DecodeComplex(Vector<T> signal)
    {
        var spectrum = _fft.Forward(signal);
        var features = new Vector<Complex<T>>(_carriers.Length);

        for (int i = 0; i < _carriers.Length; i++)
        {
            features[i] = spectrum[_carriers[i]];
        }

        return features;
    }

    /// <summary>
    /// Encodes features with specified phases (not just amplitudes).
    /// </summary>
    /// <param name="amplitudes">Feature amplitudes.</param>
    /// <param name="phases">Feature phases (in radians).</param>
    /// <returns>Time-domain signal.</returns>
    public Vector<T> EncodeWithPhase(Vector<T> amplitudes, Vector<T> phases)
    {
        if (amplitudes.Length != _carriers.Length || phases.Length != _carriers.Length)
            throw new ArgumentException("Amplitude and phase vectors must have length equal to carrier count.");

        var spectrum = new Vector<Complex<T>>(_fftSize);
        var zero = new Complex<T>(_numOps.Zero, _numOps.Zero);

        for (int i = 0; i < _fftSize; i++)
        {
            spectrum[i] = zero;
        }

        for (int i = 0; i < _carriers.Length; i++)
        {
            int bin = _carriers[i];
            spectrum[bin] = Complex<T>.FromPolarCoordinates(amplitudes[i], phases[i]);

            // Conjugate mirror for real-valued output
            int mirrorBin = _fftSize - bin;
            if (mirrorBin != bin && mirrorBin < _fftSize)
            {
                spectrum[mirrorBin] = new Complex<T>(
                    spectrum[bin].Real,
                    _numOps.Negate(spectrum[bin].Imaginary));
            }
        }

        return _fft.Inverse(spectrum);
    }
}
