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
    /// Computes the coherence between two signals.
    /// Uses Engine operations for all spectral computations.
    /// </summary>
    public Vector<T> Coherence(Vector<T> x, Vector<T> y)
    {
        var specX = SpectralEngineHelper.FFT(x);
        var specY = SpectralEngineHelper.FFT(y);
        int n = (int)specX.Length;

        // CSD = X * conj(Y), then |CSD|^2
        var csd = SpectralEngineHelper.CrossSpectral(specX, specY);
        var csdMagSq = SpectralEngineHelper.MagnitudeSquared(csd);

        // S_xx = |X|^2, S_yy = |Y|^2
        var sxx = SpectralEngineHelper.MagnitudeSquared(specX);
        var syy = SpectralEngineHelper.MagnitudeSquared(specY);

        // Coherence = |S_xy|^2 / (S_xx * S_yy + epsilon)
        var denom = Engine.TensorMultiply(sxx, syy);
        var epsilon = _numOps.FromDouble(1e-12);
        denom = Engine.TensorAddScalar(denom, epsilon);
        var coherenceTensor = Engine.TensorDivide(csdMagSq, denom);

        return SpectralEngineHelper.ToVector(coherenceTensor);
    }
}
