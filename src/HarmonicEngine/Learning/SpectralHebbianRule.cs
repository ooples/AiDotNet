using AiDotNet.HarmonicEngine.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.HarmonicEngine.Learning;

/// <summary>
/// Implements the spectral Hebbian learning rule with anti-Hebbian decorrelation.
/// This updates a spectral filter H(k) based on input-output coherence at each frequency,
/// converging to the Wiener optimal filter without backpropagation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Hebbian learning follows the principle "what fires together, wires together."
/// In the spectral domain, this means:
///
/// For each frequency k:
/// - If input X(k) and target Y(k) are in phase (coherent), strengthen H(k)
/// - If they are out of phase, weaken H(k)
///
/// The update rule is: deltaH(k) = eta * Y(k) * conj(X(k))
/// This is the cross-spectral density at frequency k.
///
/// The anti-Hebbian term prevents collapse:
/// deltaH(k) -= alpha * H(k) * |X(k)|^2
/// This decorrelation forces different frequency bins to capture different features.
///
/// Together, these converge to: H_opt(k) = S_xy(k) / S_xx(k) — the Wiener filter,
/// which is the optimal linear filter in the mean-squared-error sense.
/// </para>
/// </remarks>
public class SpectralHebbianRule<T> : ISpectralLearningRule<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly INumericOperations<Complex<T>> _complexOps;
    private readonly T _learningRate;
    private readonly T _antiHebbianAlpha;

    /// <inheritdoc/>
    public string Name => "SpectralHebbian";

    /// <inheritdoc/>
    public double LearningRate => _numOps.ToDouble(_learningRate);

    /// <summary>
    /// Initializes a new spectral Hebbian learning rule.
    /// </summary>
    /// <param name="learningRate">Learning rate eta for the Hebbian update.</param>
    /// <param name="antiHebbianAlpha">Strength of anti-Hebbian decorrelation (prevents collapse).</param>
    public SpectralHebbianRule(double learningRate = 0.01, double antiHebbianAlpha = 0.1)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _complexOps = MathHelper.GetNumericOperations<Complex<T>>();
        _learningRate = _numOps.FromDouble(learningRate);
        _antiHebbianAlpha = _numOps.FromDouble(antiHebbianAlpha);
    }

    /// <summary>
    /// Updates the spectral filter using the Hebbian rule with anti-Hebbian decorrelation.
    /// </summary>
    /// <param name="filter">The spectral filter H(k) to update (modified in place).</param>
    /// <param name="inputSpectrum">FFT of the input signal X(k).</param>
    /// <param name="targetSpectrum">FFT of the target signal Y(k).</param>
    /// <remarks>
    /// <para>
    /// Update rule per frequency bin k:
    ///   deltaH(k) = eta * [Y(k) * conj(X(k)) - alpha * H(k) * |X(k)|^2]
    ///
    /// The first term (Hebbian): strengthens H at frequencies where input and target correlate
    /// The second term (anti-Hebbian): prevents unbounded growth and encourages decorrelation
    /// </para>
    /// </remarks>
    public void Update(Vector<Complex<T>> filter, Vector<Complex<T>> inputSpectrum, Vector<Complex<T>> targetSpectrum)
    {
        int n = filter.Length;

        for (int k = 0; k < n; k++)
        {
            // Y(k) * conj(X(k)) — cross-spectral density at bin k
            var xr = inputSpectrum[k].Real;
            var xi = inputSpectrum[k].Imaginary;
            var yr = targetSpectrum[k].Real;
            var yi = targetSpectrum[k].Imaginary;

            // conj(X) = (xr, -xi)
            // Y * conj(X) = (yr*xr + yi*xi) + i*(yi*xr - yr*xi)
            var hebbianReal = _numOps.Add(
                _numOps.Multiply(yr, xr),
                _numOps.Multiply(yi, xi));
            var hebbianImag = _numOps.Subtract(
                _numOps.Multiply(yi, xr),
                _numOps.Multiply(yr, xi));

            // |X(k)|^2 = xr^2 + xi^2
            var powerX = _numOps.Add(
                _numOps.Multiply(xr, xr),
                _numOps.Multiply(xi, xi));

            // Normalize by input power to prevent divergence on high-energy bins
            var epsilon = _numOps.FromDouble(1e-10);
            var normFactor = _numOps.Divide(_numOps.One, _numOps.Add(powerX, epsilon));

            // Anti-Hebbian: alpha * H(k) * |X(k)|^2 (normalized)
            var antiReal = _numOps.Multiply(_antiHebbianAlpha,
                _numOps.Multiply(filter[k].Real, powerX));
            var antiImag = _numOps.Multiply(_antiHebbianAlpha,
                _numOps.Multiply(filter[k].Imaginary, powerX));

            // deltaH = eta * normFactor * (hebbian - anti-hebbian)
            var deltaReal = _numOps.Multiply(_learningRate,
                _numOps.Multiply(normFactor, _numOps.Subtract(hebbianReal, antiReal)));
            var deltaImag = _numOps.Multiply(_learningRate,
                _numOps.Multiply(normFactor, _numOps.Subtract(hebbianImag, antiImag)));

            // Update filter
            filter[k] = new Complex<T>(
                _numOps.Add(filter[k].Real, deltaReal),
                _numOps.Add(filter[k].Imaginary, deltaImag));
        }
    }

    /// <summary>
    /// Computes the distance between the current filter and the Wiener optimal filter.
    /// Used to measure convergence.
    /// </summary>
    /// <param name="filter">Current filter H(k).</param>
    /// <param name="wienerFilter">Optimal Wiener filter H_opt(k).</param>
    /// <returns>Mean squared error between current and optimal filter.</returns>
    public T ConvergenceError(Vector<Complex<T>> filter, Vector<Complex<T>> wienerFilter)
    {
        int n = filter.Length;
        T totalError = _numOps.Zero;

        for (int k = 0; k < n; k++)
        {
            var diffReal = _numOps.Subtract(filter[k].Real, wienerFilter[k].Real);
            var diffImag = _numOps.Subtract(filter[k].Imaginary, wienerFilter[k].Imaginary);
            totalError = _numOps.Add(totalError,
                _numOps.Add(
                    _numOps.Multiply(diffReal, diffReal),
                    _numOps.Multiply(diffImag, diffImag)));
        }

        return _numOps.Divide(totalError, _numOps.FromDouble(n));
    }
}
