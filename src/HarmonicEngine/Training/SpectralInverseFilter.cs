using AiDotNet.HarmonicEngine.Core;

namespace AiDotNet.HarmonicEngine.Training;

/// <summary>
/// Computes and applies the Tikhonov-regularized spectral inverse of a complex
/// filter: <c>H⁻¹(k) = conj(H(k)) / (|H(k)|² + ε)</c>. This is the core DSP
/// operation behind the spectral target propagation training strategy — given
/// a trained Hebbian filter H that maps inputs to outputs in the frequency
/// domain, the inverse maps targets from the output space back to the input
/// space, giving each earlier layer a local target without backpropagation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Suppose a spectral filter <c>H</c> takes an input
/// signal <c>x</c> and produces output <c>y = IFFT(H · FFT(x))</c>. The
/// inverse filter <c>H⁻¹</c> goes the other way: given <c>y</c>, it recovers
/// (approximately) <c>x = IFFT(H⁻¹ · FFT(y))</c>. In pure math this inverse
/// is <c>1/H(k)</c>, but that blows up wherever <c>H(k)</c> is near zero.
/// Tikhonov regularization fixes this by adding a small ε to the denominator:
/// </para>
/// <para>
/// <c>H⁻¹(k) = conj(H(k)) / (|H(k)|² + ε)</c>
/// </para>
/// <para>
/// When <c>|H(k)|</c> is large (filter is active at that frequency) the
/// inverse is approximately <c>1/H(k)</c> as expected. When <c>|H(k)|</c>
/// is small (filter is inactive) the inverse is damped toward zero instead
/// of blowing up. The paper's choice of <c>ε</c> is <i>adaptive</i>: we set
/// <c>ε = 0.01 · mean(|H|²)</c>, so it scales with the filter's typical
/// magnitude. This removes ε as a hand-tuned hyperparameter.
/// </para>
/// </remarks>
public class SpectralInverseFilter<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Vector<Complex<T>> _inverseFilter;
    private readonly int _length;

    /// <summary>
    /// Gets the underlying inverse filter coefficients H⁻¹(k) as a complex vector.
    /// </summary>
    public Vector<Complex<T>> Coefficients => _inverseFilter;

    /// <summary>
    /// Gets the filter length (number of frequency bins).
    /// </summary>
    public int Length => _length;

    /// <summary>
    /// Constructs the Tikhonov-regularized spectral inverse of a complex filter.
    /// </summary>
    /// <param name="filter">The forward filter <c>H(k)</c> to invert. Each entry
    /// is a complex coefficient in the frequency domain.</param>
    /// <param name="regularizationFactor">The fraction of the filter's mean squared
    /// magnitude to use as ε. Default 0.01 — small enough to leave active bins
    /// nearly unchanged while damping near-zero bins. Higher values give more
    /// aggressive smoothing / more stability; lower values give tighter inversion.</param>
    public SpectralInverseFilter(Vector<Complex<T>> filter, double regularizationFactor = 0.01)
    {
        if (filter.Length == 0)
            throw new ArgumentException("Filter must be non-empty.", nameof(filter));
        if (regularizationFactor <= 0.0 || !double.IsFinite(regularizationFactor))
            throw new ArgumentOutOfRangeException(nameof(regularizationFactor),
                $"Regularization factor must be positive and finite, got {regularizationFactor}.");

        _numOps = MathHelper.GetNumericOperations<T>();
        _length = filter.Length;
        _inverseFilter = new Vector<Complex<T>>(_length);

        // Compute |H(k)|² for each bin and the mean across bins.
        // This is used both for ε = c · mean(|H|²) and for the per-bin denominator.
        var magSq = new double[_length];
        double sumMagSq = 0;
        for (int k = 0; k < _length; k++)
        {
            double re = _numOps.ToDouble(filter[k].Real);
            double im = _numOps.ToDouble(filter[k].Imaginary);
            magSq[k] = re * re + im * im;
            sumMagSq += magSq[k];
        }
        double meanMagSq = sumMagSq / _length;

        // Adaptive ε: scales with the filter's typical magnitude.
        // If the filter is zero (meanMagSq = 0), fall back to a small absolute floor
        // so the inverse is well-defined even for an untrained (all-zero) filter.
        double epsilon = Math.Max(regularizationFactor * meanMagSq, 1e-12);

        // Build H⁻¹(k) = conj(H(k)) / (|H(k)|² + ε)
        for (int k = 0; k < _length; k++)
        {
            double denom = magSq[k] + epsilon;
            double re = _numOps.ToDouble(filter[k].Real) / denom;
            double im = -_numOps.ToDouble(filter[k].Imaginary) / denom;
            _inverseFilter[k] = new Complex<T>(_numOps.FromDouble(re), _numOps.FromDouble(im));
        }
    }

    /// <summary>
    /// Applies the inverse filter to a real-valued signal in the frequency domain.
    /// Computes <c>IFFT(H⁻¹(k) · FFT(input))</c> — that is, FFT the signal, multiply
    /// by the inverse filter's spectrum, and IFFT back to the time domain. Used to
    /// propagate a target signal from one layer's output space back to the previous
    /// layer's output space during training.
    /// </summary>
    /// <param name="input">The real-valued input signal, length <see cref="Length"/>.</param>
    /// <returns>The filtered (target-propagated) signal, same length as input.</returns>
    public Vector<T> ApplyReal(Vector<T> input)
    {
        if (input.Length != _length)
            throw new ArgumentException(
                $"Input length ({input.Length}) must match filter length ({_length}).",
                nameof(input));

        // Step 1: FFT the input. Uses the Engine-accelerated spectral helper.
        var spectrum = SpectralEngineHelper.FFT(input);
        var spectrumVec = SpectralEngineHelper.ToComplexVector(spectrum);

        // Step 2: Per-bin complex multiply H⁻¹ · X.
        var filtered = new Vector<Complex<T>>(_length);
        for (int k = 0; k < _length; k++)
        {
            double hRe = _numOps.ToDouble(_inverseFilter[k].Real);
            double hIm = _numOps.ToDouble(_inverseFilter[k].Imaginary);
            double xRe = _numOps.ToDouble(spectrumVec[k].Real);
            double xIm = _numOps.ToDouble(spectrumVec[k].Imaginary);

            // Complex multiply: (h_re + i·h_im)(x_re + i·x_im)
            double re = hRe * xRe - hIm * xIm;
            double im = hRe * xIm + hIm * xRe;
            filtered[k] = new Complex<T>(_numOps.FromDouble(re), _numOps.FromDouble(im));
        }

        // Step 3: IFFT back to the time domain. We reuse the FastFourierTransform
        // since SpectralEngineHelper.IFFTReal expects a tensor rather than a vector.
        var fft = new FastFourierTransform<T>();
        return fft.Inverse(filtered);
    }

    /// <summary>
    /// Applies the inverse filter to a complex-valued signal already in the
    /// frequency domain (no FFT needed). Useful when the caller has already
    /// computed the FFT and wants to avoid redundant transforms.
    /// </summary>
    /// <param name="spectrum">The complex input spectrum, length <see cref="Length"/>.</param>
    /// <returns>The filtered spectrum <c>H⁻¹(k) · spectrum(k)</c>, same length.</returns>
    public Vector<Complex<T>> ApplySpectral(Vector<Complex<T>> spectrum)
    {
        if (spectrum.Length != _length)
            throw new ArgumentException(
                $"Spectrum length ({spectrum.Length}) must match filter length ({_length}).",
                nameof(spectrum));

        var filtered = new Vector<Complex<T>>(_length);
        for (int k = 0; k < _length; k++)
        {
            double hRe = _numOps.ToDouble(_inverseFilter[k].Real);
            double hIm = _numOps.ToDouble(_inverseFilter[k].Imaginary);
            double xRe = _numOps.ToDouble(spectrum[k].Real);
            double xIm = _numOps.ToDouble(spectrum[k].Imaginary);

            double re = hRe * xRe - hIm * xIm;
            double im = hRe * xIm + hIm * xRe;
            filtered[k] = new Complex<T>(_numOps.FromDouble(re), _numOps.FromDouble(im));
        }
        return filtered;
    }
}
