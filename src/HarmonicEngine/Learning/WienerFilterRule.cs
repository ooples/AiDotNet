using AiDotNet.HarmonicEngine.Transforms;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.HarmonicEngine.Learning;

/// <summary>
/// Computes the Wiener optimal filter directly from input and target signals.
/// The Wiener filter H_opt(k) = S_xy(k) / S_xx(k) minimizes the mean squared error
/// between filtered input and target in a single computation (no iteration needed).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Wiener filter is the mathematically optimal answer to the question:
/// "What is the best linear filter to apply to signal X to produce signal Y?"
///
/// It computes this optimum directly from two quantities:
/// 1. Cross-spectral density S_xy: how input and target correlate at each frequency
/// 2. Auto-spectral density S_xx: how much energy the input has at each frequency
///
/// The ratio H_opt = S_xy / S_xx tells you exactly how much to scale and phase-shift
/// each frequency to best reconstruct the target from the input.
///
/// This serves as the ground truth for validating that Hebbian learning converges correctly.
/// If Hebbian learning works, the filter it produces should approach the Wiener filter.
/// </para>
/// </remarks>
public class WienerFilterRule<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly CrossSpectralDensity<T> _csd;
    private readonly FastFourierTransform<T> _fft;

    /// <summary>
    /// Initializes a new WienerFilterRule.
    /// </summary>
    public WienerFilterRule()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _csd = new CrossSpectralDensity<T>();
        _fft = new FastFourierTransform<T>();
    }

    /// <summary>
    /// Computes the Wiener optimal filter from input and target signals.
    /// </summary>
    /// <param name="input">The input signal x(t).</param>
    /// <param name="target">The target signal y(t).</param>
    /// <returns>The optimal filter H_opt(k) = S_xy(k) / S_xx(k) as complex coefficients.</returns>
    public Vector<Complex<T>> ComputeOptimal(Vector<T> input, Vector<T> target)
    {
        int n = input.Length;

        // Compute spectra
        var inputSpectrum = _fft.Forward(input);
        var targetSpectrum = _fft.Forward(target);

        // Cross-spectral density: S_xy(k) = X(k) * conj(Y(k))
        // Note: We want Y * conj(X) for H = S_yx / S_xx convention
        var crossSpectral = _csd.ComputeFromSpectra(targetSpectrum, inputSpectrum);

        // Auto-spectral density: S_xx(k) = |X(k)|^2
        var autoSpectral = _csd.AutoSpectralFromSpectrum(inputSpectrum);

        // H_opt(k) = S_yx(k) / S_xx(k)
        var filter = new Vector<Complex<T>>(n);
        var epsilon = _numOps.FromDouble(1e-10);

        for (int k = 0; k < n; k++)
        {
            var denom = _numOps.Add(autoSpectral[k], epsilon);
            filter[k] = new Complex<T>(
                _numOps.Divide(crossSpectral[k].Real, denom),
                _numOps.Divide(crossSpectral[k].Imaginary, denom));
        }

        return filter;
    }

    /// <summary>
    /// Applies a spectral filter to an input signal and returns the filtered output.
    /// </summary>
    /// <param name="input">The input signal.</param>
    /// <param name="filter">The spectral filter H(k).</param>
    /// <returns>The filtered signal.</returns>
    public Vector<T> Apply(Vector<T> input, Vector<Complex<T>> filter)
    {
        var spectrum = _fft.Forward(input);
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();
        int n = input.Length;

        var filteredSpectrum = new Vector<Complex<T>>(n);
        for (int k = 0; k < n; k++)
        {
            filteredSpectrum[k] = complexOps.Multiply(filter[k], spectrum[k]);
        }

        return _fft.Inverse(filteredSpectrum);
    }

    /// <summary>
    /// Computes the MSE between filtered input and target to validate filter quality.
    /// </summary>
    /// <param name="input">The input signal.</param>
    /// <param name="target">The target signal.</param>
    /// <param name="filter">The filter to evaluate.</param>
    /// <returns>Mean squared error between filtered input and target.</returns>
    public T ComputeMSE(Vector<T> input, Vector<T> target, Vector<Complex<T>> filter)
    {
        var filtered = Apply(input, filter);
        int n = input.Length;
        T totalError = _numOps.Zero;

        for (int i = 0; i < n; i++)
        {
            var diff = _numOps.Subtract(filtered[i], target[i]);
            totalError = _numOps.Add(totalError, _numOps.Multiply(diff, diff));
        }

        return _numOps.Divide(totalError, _numOps.FromDouble(n));
    }
}
