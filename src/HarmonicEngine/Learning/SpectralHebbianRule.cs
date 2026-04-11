using AiDotNet.HarmonicEngine.Core;
using AiDotNet.HarmonicEngine.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.HarmonicEngine.Learning;

/// <summary>
/// Implements the spectral Hebbian learning rule with anti-Hebbian decorrelation.
/// Uses Engine-accelerated operations for cross-spectral density and magnitude computation.
/// </summary>
public class SpectralHebbianRule<T> : ISpectralLearningRule<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _learningRate;
    private readonly T _antiHebbianAlpha;
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <inheritdoc/>
    public string Name => "SpectralHebbian";

    /// <inheritdoc/>
    public double LearningRate => _numOps.ToDouble(_learningRate);

    public SpectralHebbianRule(double learningRate = 0.01, double antiHebbianAlpha = 0.1)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _learningRate = _numOps.FromDouble(learningRate);
        _antiHebbianAlpha = _numOps.FromDouble(antiHebbianAlpha);
    }

    /// <summary>
    /// Updates the spectral filter using Engine-accelerated Hebbian rule.
    /// </summary>
    public void Update(Vector<Complex<T>> filter, Vector<Complex<T>> inputSpectrum, Vector<Complex<T>> targetSpectrum)
    {
        int n = filter.Length;

        // Convert to tensors for Engine acceleration
        var inputT = SpectralEngineHelper.ToComplexTensor(inputSpectrum);
        var targetT = SpectralEngineHelper.ToComplexTensor(targetSpectrum);
        var filterT = SpectralEngineHelper.ToComplexTensor(filter);

        // Hebbian term: Y * conj(X) — cross-spectral density (Engine-accelerated)
        var hebbian = Engine.NativeComplexCrossSpectral(targetT, inputT);

        // Power spectrum: |X|^2 (Engine-accelerated)
        var powerX = Engine.NativeComplexMagnitudeSquared(inputT);

        // Per-element update still uses scalar since anti-Hebbian combines real*complex
        // which isn't a single Engine op. The cross-spectral and magnitude are the hot paths.
        var epsilon = _numOps.FromDouble(1e-10);

        for (int k = 0; k < n; k++)
        {
            var px = powerX[k];
            var normFactor = _numOps.Divide(_numOps.One, _numOps.Add(px, epsilon));

            // Anti-Hebbian: alpha * H(k) * |X(k)|^2
            var antiReal = _numOps.Multiply(_antiHebbianAlpha, _numOps.Multiply(filter[k].Real, px));
            var antiImag = _numOps.Multiply(_antiHebbianAlpha, _numOps.Multiply(filter[k].Imaginary, px));

            // deltaH = eta * normFactor * (hebbian - anti)
            var deltaReal = _numOps.Multiply(_learningRate,
                _numOps.Multiply(normFactor, _numOps.Subtract(hebbian[k].Real, antiReal)));
            var deltaImag = _numOps.Multiply(_learningRate,
                _numOps.Multiply(normFactor, _numOps.Subtract(hebbian[k].Imaginary, antiImag)));

            filter[k] = new Complex<T>(
                _numOps.Add(filter[k].Real, deltaReal),
                _numOps.Add(filter[k].Imaginary, deltaImag));
        }
    }

    /// <summary>
    /// Computes convergence error using Engine-accelerated magnitude.
    /// </summary>
    public T ConvergenceError(Vector<Complex<T>> filter, Vector<Complex<T>> wienerFilter)
    {
        int n = filter.Length;

        // Compute difference tensor
        var filterT = SpectralEngineHelper.ToComplexTensor(filter);
        var wienerT = SpectralEngineHelper.ToComplexTensor(wienerFilter);

        // diff = filter - wiener (no Engine complex subtract, use element-wise)
        var diff = new Tensor<Complex<T>>([n]);
        for (int i = 0; i < n; i++)
            diff[i] = new Complex<T>(
                _numOps.Subtract(filter[i].Real, wienerFilter[i].Real),
                _numOps.Subtract(filter[i].Imaginary, wienerFilter[i].Imaginary));

        // |diff|^2 via Engine, then sum
        var magSq = Engine.NativeComplexMagnitudeSquared(diff);
        var totalError = Engine.TensorSum(magSq);

        return _numOps.Divide(totalError, _numOps.FromDouble(n));
    }
}
