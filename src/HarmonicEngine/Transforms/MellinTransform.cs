using AiDotNet.LinearAlgebra;

namespace AiDotNet.HarmonicEngine.Transforms;

/// <summary>
/// Computes the Mellin transform for scale-invariant signal analysis.
/// The Mellin transform converts scale changes in the time domain to shifts in the transform domain,
/// enabling recognition of patterns regardless of their amplitude scaling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Imagine you see a pattern in a signal that repeats at different rates.
/// For example, a chirp that takes 10 samples to complete vs. the same chirp stretched to 20 samples.
/// The Mellin transform produces the same magnitude spectrum for both versions — it is invariant
/// to <b>temporal scaling</b> (dilation of the time axis), not to amplitude scaling.
///
/// Mathematically: M{f(ax)}(s) = a^(-s) · M{f(x)}(s), so the magnitude |M{f(ax)}| = |a|^(-Re(s)) · |M{f(x)}|,
/// which means the shape of the magnitude spectrum is preserved across time-scaled versions of the same pattern.
///
/// How it works:
/// 1. Resample the signal from linear time to logarithmic time (log-resampling)
/// 2. Apply a standard FFT to the log-resampled signal
/// 3. The resulting spectrum is scale-invariant — time-axis scaling in the original domain
///    becomes a mere phase shift in the Mellin domain
/// </para>
/// </remarks>
public class MellinTransform<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly FastFourierTransform<T> _fft;

    /// <summary>
    /// Initializes a new instance of the MellinTransform class.
    /// </summary>
    public MellinTransform()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _fft = new FastFourierTransform<T>();
    }

    /// <summary>
    /// Computes the forward Mellin transform of a real-valued signal.
    /// </summary>
    /// <param name="signal">The input signal. Length must be a power of 2.</param>
    /// <returns>The Mellin transform coefficients as complex numbers.</returns>
    /// <remarks>
    /// <para>
    /// The Mellin transform M{f}(s) = integral[0 to inf] f(x) * x^(s-1) dx
    /// is computed via the substitution x = e^t, which converts it to a Fourier transform
    /// of f(e^t) * e^t: M{f}(s) = FT{f(e^t) * e^t}(s).
    ///
    /// Implementation:
    /// 1. Resample signal from linear domain [1, N] to exponential domain
    /// 2. Apply Jacobian weighting (multiply by e^t)
    /// 3. Compute FFT of the resampled, weighted signal
    /// </para>
    /// </remarks>
    public Vector<Complex<T>> Forward(Vector<T> signal)
    {
        int n = signal.Length;

        if (n < 2)
            throw new ArgumentException("Signal must have at least 2 samples.", nameof(signal));

        if ((n & (n - 1)) != 0)
            throw new ArgumentException(
                $"Signal length ({n}) must be a power of 2 for FFT.", nameof(signal));

        // Resample to logarithmic time grid
        // Map uniform indices [0, n-1] to exponential positions in [0, n-1]
        var logResampled = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            // Map i in [0, n-1] to t in [0, ln(n)]
            double t = (double)i / (n - 1) * Math.Log(n);
            // Exponential position in original signal
            double expPos = Math.Exp(t) - 1.0; // maps [0, ln(n)] -> [0, n-1]

            // Linear interpolation in the original signal
            int idx0 = (int)Math.Floor(expPos);
            int idx1 = Math.Min(idx0 + 1, n - 1);
            idx0 = Math.Max(0, Math.Min(idx0, n - 1));
            double frac = expPos - idx0;

            var val0 = _numOps.ToDouble(signal[idx0]);
            var val1 = _numOps.ToDouble(signal[idx1]);
            double interpolated = val0 * (1.0 - frac) + val1 * frac;

            // Apply Jacobian weighting: multiply by e^t (derivative of the substitution)
            double jacobian = Math.Exp(t);
            logResampled[i] = _numOps.FromDouble(interpolated * jacobian);
        }

        // Apply FFT to get the Mellin transform
        return _fft.Forward(logResampled);
    }

    /// <summary>
    /// Computes the magnitude spectrum of the Mellin transform, which is scale-invariant.
    /// </summary>
    /// <param name="signal">The input signal.</param>
    /// <returns>The scale-invariant magnitude spectrum.</returns>
    /// <remarks>
    /// <para>
    /// The magnitude of the Mellin transform |M{f}(s)| is invariant to scaling of the input signal.
    /// If g(x) = f(ax) for some scale factor a, then |M{g}(s)| = |M{f}(s)| for all s.
    /// This is the "scale-invariant fingerprint" of the signal.
    /// </para>
    /// </remarks>
    public Vector<T> ScaleInvariantFingerprint(Vector<T> signal)
    {
        var mellin = Forward(signal);
        int n = mellin.Length;
        var fingerprint = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            fingerprint[i] = mellin[i].Magnitude;
        }

        return fingerprint;
    }

    /// <summary>
    /// Computes the Mellin-Fourier fingerprint that is invariant to both scaling and time-shifting.
    /// </summary>
    /// <param name="signal">The input signal.</param>
    /// <returns>A fingerprint that is invariant to both scale and shift transformations.</returns>
    /// <remarks>
    /// <para>
    /// Combining Mellin (scale-invariant) with a second Fourier magnitude (shift-invariant)
    /// produces a representation that is unchanged under both scaling and time-shifting.
    /// This is the complete invariant fingerprint used in the MellinFourierLayer.
    /// </para>
    /// </remarks>
    public Vector<T> ScaleShiftInvariantFingerprint(Vector<T> signal)
    {
        // Step 1: Mellin magnitude (scale-invariant)
        var scaleInvariant = ScaleInvariantFingerprint(signal);

        // Step 2: FFT of the magnitude spectrum, then take magnitude again (shift-invariant)
        var secondFft = _fft.Forward(scaleInvariant);
        int n = secondFft.Length;
        var fingerprint = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            fingerprint[i] = secondFft[i].Magnitude;
        }

        return fingerprint;
    }
}
