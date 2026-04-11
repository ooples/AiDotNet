using AiDotNet.HarmonicEngine.Core;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.HarmonicEngine.Transforms;

/// <summary>
/// Computes the cross-spectral density (CSD) and auto-spectral density (ASD) of signals
/// using Engine-accelerated operations (SIMD on CPU, GPU when available).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
public class CrossSpectralDensity<T>
{
    private readonly INumericOperations<T> _numOps;
    private static IEngine Engine => AiDotNetEngine.Current;

    public CrossSpectralDensity()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes the cross-spectral density S_xy(f) = X(f) * conj(Y(f)) between two signals.
    /// Uses Engine.NativeComplexCrossSpectral for SIMD/GPU acceleration.
    /// </summary>
    public Vector<Complex<T>> Compute(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException(
                $"Signals must have equal length. X: {x.Length}, Y: {y.Length}.");

        var specX = SpectralEngineHelper.FFT(x);
        var specY = SpectralEngineHelper.FFT(y);
        var csd = SpectralEngineHelper.CrossSpectral(specX, specY);
        return SpectralEngineHelper.ToComplexVector(csd);
    }

    /// <summary>
    /// Computes the cross-spectral density from pre-computed spectra.
    /// </summary>
    public Vector<Complex<T>> ComputeFromSpectra(Vector<Complex<T>> spectrumX, Vector<Complex<T>> spectrumY)
    {
        if (spectrumX.Length != spectrumY.Length)
            throw new ArgumentException(
                $"Spectra must have equal length. X: {spectrumX.Length}, Y: {spectrumY.Length}.");

        var tx = SpectralEngineHelper.ToComplexTensor(spectrumX);
        var ty = SpectralEngineHelper.ToComplexTensor(spectrumY);
        var csd = SpectralEngineHelper.CrossSpectral(tx, ty);
        return SpectralEngineHelper.ToComplexVector(csd);
    }

    /// <summary>
    /// Computes the auto-spectral density (power spectrum) S_xx(f) = |X(f)|^2.
    /// Uses Engine.NativeComplexMagnitudeSquared for SIMD/GPU acceleration.
    /// </summary>
    public Vector<T> AutoSpectral(Vector<T> x)
    {
        var spectrum = SpectralEngineHelper.FFT(x);
        var magSq = SpectralEngineHelper.MagnitudeSquared(spectrum);
        return SpectralEngineHelper.ToVector(magSq);
    }

    /// <summary>
    /// Computes the auto-spectral density from a pre-computed spectrum.
    /// </summary>
    public Vector<T> AutoSpectralFromSpectrum(Vector<Complex<T>> spectrum)
    {
        var tensor = SpectralEngineHelper.ToComplexTensor(spectrum);
        var magSq = SpectralEngineHelper.MagnitudeSquared(tensor);
        return SpectralEngineHelper.ToVector(magSq);
    }

    /// <summary>
    /// Computes the magnitude-squared coherence between two signals using Welch's method
    /// with overlapping segments, Hann windowing, and spectral averaging.
    /// </summary>
    /// <param name="x">First signal.</param>
    /// <param name="y">Second signal.</param>
    /// <param name="segmentLength">Length of each segment (must be power of 2). Auto-selected if 0.</param>
    /// <param name="overlap">Fractional overlap between segments in [0, 1). Default 0.5 (50%).</param>
    /// <returns>Coherence in [0, 1] for each frequency bin.</returns>
    /// <remarks>
    /// <para>
    /// The magnitude-squared coherence is defined as:
    ///   γ²_xy(f) = |S_xy(f)|² / (S_xx(f) · S_yy(f))
    ///
    /// where S_xy, S_xx, S_yy are cross- and auto-spectral densities averaged over
    /// multiple segments. With a single segment, this ratio is identically 1, so
    /// meaningful coherence requires averaging (Welch's method).
    ///
    /// For input shorter than ~4 * segmentLength, this falls back to a single-segment
    /// computation which will return values near 1 everywhere (not meaningful).
    /// </para>
    /// </remarks>
    public Vector<T> Coherence(Vector<T> x, Vector<T> y, int segmentLength = 0, double overlap = 0.5)
    {
        if (x.Length != y.Length)
            throw new ArgumentException(
                $"Signals must have equal length. X: {x.Length}, Y: {y.Length}.");

        if (overlap < 0.0 || overlap >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(overlap), "Overlap must be in [0, 1).");

        int n = x.Length;

        // Auto-select segment length: 1/8 of input rounded down to power of 2, min 32
        if (segmentLength <= 0)
        {
            segmentLength = Math.Max(32, NextPowerOfTwoBelow(n / 8));
        }

        if ((segmentLength & (segmentLength - 1)) != 0)
            throw new ArgumentException(
                $"Segment length must be a power of 2, got {segmentLength}.", nameof(segmentLength));

        if (segmentLength > n)
            throw new ArgumentException(
                $"Segment length ({segmentLength}) cannot exceed signal length ({n}).", nameof(segmentLength));

        int step = Math.Max(1, (int)(segmentLength * (1.0 - overlap)));
        int numSegments = (n - segmentLength) / step + 1;

        // Precompute Hann window and its power normalization factor
        var window = new double[segmentLength];
        double windowPower = 0;
        for (int i = 0; i < segmentLength; i++)
        {
            window[i] = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (segmentLength - 1)));
            windowPower += window[i] * window[i];
        }

        // Averaged spectral densities
        var avgSxx = new double[segmentLength];
        var avgSyy = new double[segmentLength];
        var avgSxyReal = new double[segmentLength];
        var avgSxyImag = new double[segmentLength];

        var segX = new Vector<T>(segmentLength);
        var segY = new Vector<T>(segmentLength);

        for (int s = 0; s < numSegments; s++)
        {
            int offset = s * step;

            // Apply window to segments
            for (int i = 0; i < segmentLength; i++)
            {
                double xi = _numOps.ToDouble(x[offset + i]) * window[i];
                double yi = _numOps.ToDouble(y[offset + i]) * window[i];
                segX[i] = _numOps.FromDouble(xi);
                segY[i] = _numOps.FromDouble(yi);
            }

            // FFT each windowed segment
            var specX = SpectralEngineHelper.FFT(segX);
            var specY = SpectralEngineHelper.FFT(segY);
            var specXVec = SpectralEngineHelper.ToComplexVector(specX);
            var specYVec = SpectralEngineHelper.ToComplexVector(specY);

            // Accumulate S_xx, S_yy, S_xy = X * conj(Y)
            for (int k = 0; k < segmentLength; k++)
            {
                double xr = _numOps.ToDouble(specXVec[k].Real);
                double xi = _numOps.ToDouble(specXVec[k].Imaginary);
                double yr = _numOps.ToDouble(specYVec[k].Real);
                double yi = _numOps.ToDouble(specYVec[k].Imaginary);

                avgSxx[k] += xr * xr + xi * xi;
                avgSyy[k] += yr * yr + yi * yi;
                // S_xy(k) = X(k) * conj(Y(k)) = (xr + i*xi)(yr - i*yi)
                avgSxyReal[k] += xr * yr + xi * yi;
                avgSxyImag[k] += xi * yr - xr * yi;
            }
        }

        // Divide by number of segments × window power for Welch normalization
        double norm = numSegments * windowPower;
        var coherence = new Vector<T>(segmentLength);

        for (int k = 0; k < segmentLength; k++)
        {
            avgSxx[k] /= norm;
            avgSyy[k] /= norm;
            avgSxyReal[k] /= norm;
            avgSxyImag[k] /= norm;

            double csdMagSq = avgSxyReal[k] * avgSxyReal[k] + avgSxyImag[k] * avgSxyImag[k];
            double denom = avgSxx[k] * avgSyy[k] + 1e-12;
            double c = csdMagSq / denom;

            // Clamp to [0, 1] to handle numerical roundoff
            if (c < 0) c = 0;
            if (c > 1) c = 1;
            coherence[k] = _numOps.FromDouble(c);
        }

        return coherence;
    }

    private static int NextPowerOfTwoBelow(int n)
    {
        if (n < 2) return 1;
        int p = 1;
        while (p * 2 <= n) p *= 2;
        return p;
    }
}
