using AiDotNet.HarmonicEngine.Core;
using AiDotNet.HarmonicEngine.Transforms;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.HarmonicEngine.Learning;

/// <summary>
/// Computes the Wiener optimal filter using Engine-accelerated spectral operations.
/// H_opt(k) = S_xy(k) / S_xx(k) — the minimum MSE linear filter.
/// </summary>
public class WienerFilterRule<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly CrossSpectralDensity<T> _csd;
    private static IEngine Engine => AiDotNetEngine.Current;

    public WienerFilterRule()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _csd = new CrossSpectralDensity<T>();
    }

    /// <summary>
    /// Computes the Wiener optimal filter using Engine-accelerated CSD and magnitude.
    /// </summary>
    public Vector<Complex<T>> ComputeOptimal(Vector<T> input, Vector<T> target)
    {
        int n = input.Length;

        // Engine-accelerated FFT
        var inputSpec = SpectralEngineHelper.FFT(input);
        var targetSpec = SpectralEngineHelper.FFT(target);

        // Engine-accelerated cross-spectral density: Y * conj(X)
        var crossSpectral = SpectralEngineHelper.CrossSpectral(targetSpec, inputSpec);

        // Engine-accelerated power spectrum: |X|^2
        var autoSpectral = SpectralEngineHelper.MagnitudeSquared(inputSpec);

        // H_opt(k) = S_yx(k) / S_xx(k) — per-element complex/real division
        var epsilon = _numOps.FromDouble(1e-10);
        var filter = new Vector<Complex<T>>(n);

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
    /// Applies a spectral filter using Engine-accelerated complex multiply + IFFT.
    /// </summary>
    public Vector<T> Apply(Vector<T> input, Vector<Complex<T>> filter)
    {
        var spectrum = SpectralEngineHelper.FFT(input);
        var filterT = SpectralEngineHelper.ToComplexTensor(filter);

        // Engine-accelerated complex multiply: H(k) * X(k)
        var filtered = Engine.NativeComplexMultiply(filterT, spectrum);

        // Engine-accelerated IFFT
        return SpectralEngineHelper.IFFTReal(filtered);
    }

    /// <summary>
    /// Computes MSE using Engine-accelerated operations.
    /// </summary>
    public T ComputeMSE(Vector<T> input, Vector<T> target, Vector<Complex<T>> filter)
    {
        var filtered = Apply(input, filter);
        int n = input.Length;

        // Engine: diff = filtered - target, then sum of squares
        var filteredT = SpectralEngineHelper.ToTensor(filtered);
        var targetT = SpectralEngineHelper.ToTensor(target);
        var diff = Engine.TensorSubtract(filteredT, targetT);
        var sqDiff = Engine.TensorMultiply(diff, diff);
        var totalError = Engine.TensorSum(sqDiff);

        return _numOps.Divide(totalError, _numOps.FromDouble(n));
    }
}
