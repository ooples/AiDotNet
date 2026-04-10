using AiDotNet.LinearAlgebra;

namespace AiDotNet.HarmonicEngine.Transforms;

/// <summary>
/// Computes the cross-spectral density (CSD) and auto-spectral density (ASD) of signals.
/// These are the core building blocks for spectral Hebbian learning and Wiener filter computation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Cross-spectral density measures how two signals are correlated at each frequency.
/// If two signals both have strong energy at 10 Hz and they are in phase, the CSD at 10 Hz will be large.
/// If they are unrelated at 10 Hz, the CSD will be near zero.
///
/// This is used in the HRE for:
/// - Hebbian learning: strengthening frequency connections that show input-output coherence
/// - Wiener filter: computing the optimal linear filter H(f) = CSD_xy(f) / ASD_xx(f)
/// - Phase alignment: measuring how well internal oscillators match the data
///
/// The CSD is computed as: S_xy(f) = FFT(x) * conj(FFT(y))
/// The ASD is computed as: S_xx(f) = FFT(x) * conj(FFT(x)) = |FFT(x)|^2
/// </para>
/// </remarks>
public class CrossSpectralDensity<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly INumericOperations<Complex<T>> _complexOps;
    private readonly FastFourierTransform<T> _fft;

    /// <summary>
    /// Initializes a new instance of the CrossSpectralDensity class.
    /// </summary>
    public CrossSpectralDensity()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _complexOps = MathHelper.GetNumericOperations<Complex<T>>();
        _fft = new FastFourierTransform<T>();
    }

    /// <summary>
    /// Computes the cross-spectral density S_xy(f) = X(f) * conj(Y(f)) between two signals.
    /// </summary>
    /// <param name="x">The first signal (input).</param>
    /// <param name="y">The second signal (output/target).</param>
    /// <returns>The cross-spectral density as complex values at each frequency bin.</returns>
    public Vector<Complex<T>> Compute(Vector<T> x, Vector<T> y)
    {
        var spectrumX = _fft.Forward(x);
        var spectrumY = _fft.Forward(y);
        int n = spectrumX.Length;
        var csd = new Vector<Complex<T>>(n);

        for (int k = 0; k < n; k++)
        {
            // S_xy(k) = X(k) * conj(Y(k))
            // conj(Y) = (Y.Real, -Y.Imaginary)
            // X * conj(Y) = (X.Re*Y.Re + X.Im*Y.Im) + i*(X.Im*Y.Re - X.Re*Y.Im)
            var xr = spectrumX[k].Real;
            var xi = spectrumX[k].Imaginary;
            var yr = spectrumY[k].Real;
            var yi = spectrumY[k].Imaginary;

            var realPart = _numOps.Add(
                _numOps.Multiply(xr, yr),
                _numOps.Multiply(xi, yi));
            var imagPart = _numOps.Subtract(
                _numOps.Multiply(xi, yr),
                _numOps.Multiply(xr, yi));

            csd[k] = new Complex<T>(realPart, imagPart);
        }

        return csd;
    }

    /// <summary>
    /// Computes the cross-spectral density from pre-computed spectra.
    /// </summary>
    /// <param name="spectrumX">FFT of the first signal.</param>
    /// <param name="spectrumY">FFT of the second signal.</param>
    /// <returns>The cross-spectral density.</returns>
    public Vector<Complex<T>> ComputeFromSpectra(Vector<Complex<T>> spectrumX, Vector<Complex<T>> spectrumY)
    {
        int n = spectrumX.Length;
        var csd = new Vector<Complex<T>>(n);

        for (int k = 0; k < n; k++)
        {
            var xr = spectrumX[k].Real;
            var xi = spectrumX[k].Imaginary;
            var yr = spectrumY[k].Real;
            var yi = spectrumY[k].Imaginary;

            csd[k] = new Complex<T>(
                _numOps.Add(_numOps.Multiply(xr, yr), _numOps.Multiply(xi, yi)),
                _numOps.Subtract(_numOps.Multiply(xi, yr), _numOps.Multiply(xr, yi)));
        }

        return csd;
    }

    /// <summary>
    /// Computes the auto-spectral density (power spectrum) S_xx(f) = |X(f)|^2.
    /// </summary>
    /// <param name="x">The input signal.</param>
    /// <returns>The power spectral density (real-valued) at each frequency bin.</returns>
    public Vector<T> AutoSpectral(Vector<T> x)
    {
        var spectrum = _fft.Forward(x);
        int n = spectrum.Length;
        var psd = new Vector<T>(n);

        for (int k = 0; k < n; k++)
        {
            // |X(k)|^2 = Re^2 + Im^2
            var re = spectrum[k].Real;
            var im = spectrum[k].Imaginary;
            psd[k] = _numOps.Add(
                _numOps.Multiply(re, re),
                _numOps.Multiply(im, im));
        }

        return psd;
    }

    /// <summary>
    /// Computes the auto-spectral density from a pre-computed spectrum.
    /// </summary>
    /// <param name="spectrum">FFT of the signal.</param>
    /// <returns>The power spectral density.</returns>
    public Vector<T> AutoSpectralFromSpectrum(Vector<Complex<T>> spectrum)
    {
        int n = spectrum.Length;
        var psd = new Vector<T>(n);

        for (int k = 0; k < n; k++)
        {
            var re = spectrum[k].Real;
            var im = spectrum[k].Imaginary;
            psd[k] = _numOps.Add(
                _numOps.Multiply(re, re),
                _numOps.Multiply(im, im));
        }

        return psd;
    }

    /// <summary>
    /// Computes the coherence between two signals: gamma^2(f) = |S_xy(f)|^2 / (S_xx(f) * S_yy(f)).
    /// Values range from 0 (no correlation) to 1 (perfect correlation) at each frequency.
    /// </summary>
    /// <param name="x">The first signal.</param>
    /// <param name="y">The second signal.</param>
    /// <returns>The coherence (real-valued, [0, 1]) at each frequency bin.</returns>
    public Vector<T> Coherence(Vector<T> x, Vector<T> y)
    {
        var specX = _fft.Forward(x);
        var specY = _fft.Forward(y);
        int n = specX.Length;
        var coherence = new Vector<T>(n);
        var epsilon = _numOps.FromDouble(1e-12);

        for (int k = 0; k < n; k++)
        {
            // |S_xy|^2
            var csdReal = _numOps.Add(
                _numOps.Multiply(specX[k].Real, specY[k].Real),
                _numOps.Multiply(specX[k].Imaginary, specY[k].Imaginary));
            var csdImag = _numOps.Subtract(
                _numOps.Multiply(specX[k].Imaginary, specY[k].Real),
                _numOps.Multiply(specX[k].Real, specY[k].Imaginary));
            var csdMagSq = _numOps.Add(
                _numOps.Multiply(csdReal, csdReal),
                _numOps.Multiply(csdImag, csdImag));

            // S_xx * S_yy
            var sxx = _numOps.Add(
                _numOps.Multiply(specX[k].Real, specX[k].Real),
                _numOps.Multiply(specX[k].Imaginary, specX[k].Imaginary));
            var syy = _numOps.Add(
                _numOps.Multiply(specY[k].Real, specY[k].Real),
                _numOps.Multiply(specY[k].Imaginary, specY[k].Imaginary));

            var denominator = _numOps.Add(_numOps.Multiply(sxx, syy), epsilon);
            coherence[k] = _numOps.Divide(csdMagSq, denominator);
        }

        return coherence;
    }
}
