using AiDotNet.LinearAlgebra;

namespace AiDotNet.HarmonicEngine.Transforms;

/// <summary>
/// Computes the analytic signal via the Hilbert transform, enabling extraction of
/// instantaneous amplitude, phase, and frequency from real-valued signals.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A real-valued signal (like a stock price over time) only tells you the value at each point.
/// The analytic signal is a complex-valued version that also encodes the "hidden" phase information.
/// From this complex signal, you can extract:
/// - Instantaneous amplitude (envelope): how strong the signal is at each moment
/// - Instantaneous phase: where in its cycle the signal is at each moment
/// - Instantaneous frequency: how fast the signal is oscillating at each moment
///
/// The Hilbert transform creates the analytic signal by:
/// 1. Computing the FFT of the real signal
/// 2. Zeroing out negative frequency components
/// 3. Doubling positive frequency components
/// 4. Computing the inverse FFT
/// </para>
/// </remarks>
public class AnalyticSignal<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly FastFourierTransform<T> _fft;

    /// <summary>
    /// Initializes a new instance of the AnalyticSignal class.
    /// </summary>
    public AnalyticSignal()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _fft = new FastFourierTransform<T>();
    }

    /// <summary>
    /// Computes the analytic signal of a real-valued input using the Hilbert transform.
    /// </summary>
    /// <param name="signal">The real-valued input signal. Length must be a power of 2.</param>
    /// <returns>The complex-valued analytic signal.</returns>
    /// <remarks>
    /// <para>
    /// The analytic signal z(t) = x(t) + i*H[x(t)] where H is the Hilbert transform.
    /// For a cosine input cos(wt), the analytic signal is e^(iwt) = cos(wt) + i*sin(wt).
    /// </para>
    /// </remarks>
    public Vector<Complex<T>> Compute(Vector<T> signal)
    {
        int n = signal.Length;

        if (n < 2)
            throw new ArgumentException("Signal must have at least 2 samples.", nameof(signal));

        if ((n & (n - 1)) != 0)
            throw new ArgumentException(
                $"Signal length ({n}) must be a power of 2 for FFT.", nameof(signal));

        var spectrum = _fft.Forward(signal);

        // Build the analytic signal in frequency domain:
        // - DC component (k=0): unchanged (multiply by 1)
        // - Positive frequencies (1 to N/2-1): doubled (multiply by 2)
        // - Nyquist (k=N/2): unchanged (multiply by 1)
        // - Negative frequencies (N/2+1 to N-1): zeroed (multiply by 0)
        var analyticSpectrum = new Vector<Complex<T>>(n);
        var two = _numOps.FromDouble(2.0);

        // DC component — keep as-is
        analyticSpectrum[0] = spectrum[0];

        // Positive frequencies — double them
        for (int k = 1; k < n / 2; k++)
        {
            analyticSpectrum[k] = new Complex<T>(
                _numOps.Multiply(two, spectrum[k].Real),
                _numOps.Multiply(two, spectrum[k].Imaginary));
        }

        // Nyquist frequency — keep as-is
        if (n > 1)
        {
            analyticSpectrum[n / 2] = spectrum[n / 2];
        }

        // Negative frequencies — zero them out (already zero from initialization)
        // analyticSpectrum[n/2+1 .. n-1] = 0 (default)

        // Inverse FFT to get the analytic signal in time domain
        // We need to do the inverse manually since _fft.Inverse returns Vector<T>
        var result = new Vector<Complex<T>>(n);
        var scale = _numOps.FromDouble(n);

        // Compute inverse FFT by conjugating, forward FFT, conjugating, and dividing by N
        var conjugated = new Vector<Complex<T>>(n);
        for (int i = 0; i < n; i++)
        {
            conjugated[i] = new Complex<T>(
                analyticSpectrum[i].Real,
                _numOps.Negate(analyticSpectrum[i].Imaginary));
        }

        // Treat conjugated complex as input to Forward via manual complex FFT
        // We reuse the FFT by converting: Forward of conjugate, then conjugate and scale
        var forwardInput = new Vector<T>(n);
        var forwardInputImag = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            forwardInput[i] = conjugated[i].Real;
            forwardInputImag[i] = conjugated[i].Imaginary;
        }

        // Since FFT only accepts real input, we compute IFFT manually:
        // IFFT(X) = conj(FFT(conj(X))) / N
        // We already have conj(X) in 'conjugated'. Now we need FFT of a complex input.
        // Use linearity: FFT(a + ib) = FFT(a) + i*FFT(b)
        var fftReal = _fft.Forward(forwardInput);
        var fftImag = _fft.Forward(forwardInputImag);

        for (int i = 0; i < n; i++)
        {
            // Combined FFT result for complex input
            var combinedReal = _numOps.Subtract(fftReal[i].Real, fftImag[i].Imaginary);
            var combinedImag = _numOps.Add(fftReal[i].Imaginary, fftImag[i].Real);

            // Conjugate and divide by N
            result[i] = new Complex<T>(
                _numOps.Divide(combinedReal, scale),
                _numOps.Divide(_numOps.Negate(combinedImag), scale));
        }

        return result;
    }

    /// <summary>
    /// Computes the instantaneous amplitude (envelope) of the signal.
    /// </summary>
    /// <param name="signal">The real-valued input signal.</param>
    /// <returns>The instantaneous amplitude |z(t)| at each time step.</returns>
    public Vector<T> InstantaneousAmplitude(Vector<T> signal)
    {
        var analytic = Compute(signal);
        int n = signal.Length;
        var amplitude = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            amplitude[i] = analytic[i].Magnitude;
        }

        return amplitude;
    }

    /// <summary>
    /// Computes the instantaneous phase of the signal.
    /// </summary>
    /// <param name="signal">The real-valued input signal.</param>
    /// <returns>The instantaneous phase angle (in radians) at each time step.</returns>
    public Vector<T> InstantaneousPhase(Vector<T> signal)
    {
        var analytic = Compute(signal);
        int n = signal.Length;
        var phase = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            var real = _numOps.ToDouble(analytic[i].Real);
            var imag = _numOps.ToDouble(analytic[i].Imaginary);
            phase[i] = _numOps.FromDouble(Math.Atan2(imag, real));
        }

        return phase;
    }

    /// <summary>
    /// Computes the instantaneous frequency of the signal as the derivative of phase.
    /// </summary>
    /// <param name="signal">The real-valued input signal.</param>
    /// <param name="sampleRate">The sampling rate of the signal (default 1.0).</param>
    /// <returns>
    /// The instantaneous frequency at each time step (in Hz if sampleRate is provided).
    /// Output length is signal.Length - 1 due to finite differencing.
    /// </returns>
    /// <remarks>
    /// <para>
    /// Instantaneous frequency is computed as: f_inst(t) = (1 / 2pi) * d(phase)/dt.
    /// Phase unwrapping is applied to handle 2pi discontinuities.
    /// </para>
    /// </remarks>
    public Vector<T> InstantaneousFrequency(Vector<T> signal, double sampleRate = 1.0)
    {
        var phase = InstantaneousPhase(signal);
        int n = phase.Length;
        var freq = new Vector<T>(n - 1);
        var invTwoPi = _numOps.FromDouble(sampleRate / (2.0 * Math.PI));

        for (int i = 0; i < n - 1; i++)
        {
            // Phase difference
            var diff = _numOps.Subtract(phase[i + 1], phase[i]);

            // Unwrap: bring diff into [-pi, pi]
            var diffDouble = _numOps.ToDouble(diff);
            while (diffDouble > Math.PI) diffDouble -= 2.0 * Math.PI;
            while (diffDouble < -Math.PI) diffDouble += 2.0 * Math.PI;
            diff = _numOps.FromDouble(diffDouble);

            // Frequency = (1 / 2pi) * d(phase)/dt * sampleRate
            freq[i] = _numOps.Multiply(diff, invTwoPi);
        }

        return freq;
    }
}
