using AiDotNet.HarmonicEngine.Learning;
using AiDotNet.HarmonicEngine.Transforms;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Tests Experiment 3: Verify that the spectral Hebbian rule converges to the Wiener filter.
/// </summary>
public class SpectralHebbianTests
{
    [Fact]
    public void WienerFilter_KnownSignals_ProducesOptimalFilter()
    {
        var wiener = new WienerFilterRule<double>();
        int n = 64;

        // Input: cosine at frequency 5
        var input = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            input[i] = Math.Cos(2 * Math.PI * 5 * i / n);
        }

        // Target: same cosine scaled by 2
        var target = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            target[i] = 2.0 * Math.Cos(2 * Math.PI * 5 * i / n);
        }

        // Compute optimal filter
        var optimalFilter = wiener.ComputeOptimal(input, target);

        // Apply filter and check reconstruction
        var filtered = wiener.Apply(input, optimalFilter);

        // Should closely match target
        double mse = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = filtered[i] - target[i];
            mse += diff * diff;
        }
        mse /= n;

        Assert.True(mse < 0.01, $"Wiener filter MSE = {mse}, expected < 0.01");
    }

    [Fact]
    public void HebbianRule_UpdateReducesConvergenceError()
    {
        int n = 64;
        var rule = new SpectralHebbianRule<double>(learningRate: 0.01, antiHebbianAlpha: 0.001);
        var wiener = new WienerFilterRule<double>();
        var fft = new FastFourierTransform<double>();

        // Simple signal pair: target is scaled version of input
        var input = new Vector<double>(n);
        var target = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            input[i] = Math.Cos(2 * Math.PI * 3 * i / n);
            target[i] = 1.5 * Math.Cos(2 * Math.PI * 3 * i / n);
        }

        // Compute Wiener optimal
        var optimalFilter = wiener.ComputeOptimal(input, target);

        // Initialize filter to zero (far from optimal) so we can see convergence toward it
        var filter = new Vector<Complex<double>>(n);
        for (int k = 0; k < n; k++)
        {
            filter[k] = new Complex<double>(0.0, 0.0);
        }

        // Compute initial error (should be large since filter starts at zero)
        double initialError = rule.ConvergenceError(filter, optimalFilter);

        // Apply Hebbian updates — normalized by input power for stability
        var inputSpectrum = fft.Forward(input);
        var targetSpectrum = fft.Forward(target);

        for (int iter = 0; iter < 50; iter++)
        {
            rule.Update(filter, inputSpectrum, targetSpectrum);
        }

        // Compute final error
        double finalError = rule.ConvergenceError(filter, optimalFilter);

        // Error should decrease
        Assert.True(finalError < initialError,
            $"Hebbian update should reduce convergence error. Initial: {initialError}, Final: {finalError}");
    }

    [Fact]
    public void WienerFilter_ComputeMSE_LowForOptimalFilter()
    {
        var wiener = new WienerFilterRule<double>();
        int n = 64;

        var input = new Vector<double>(n);
        var target = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * 4 * i / n) + 0.3 * Math.Sin(2 * Math.PI * 11 * i / n);
            target[i] = 0.8 * Math.Sin(2 * Math.PI * 4 * i / n) + 0.6 * Math.Sin(2 * Math.PI * 11 * i / n);
        }

        var optimalFilter = wiener.ComputeOptimal(input, target);
        double mse = wiener.ComputeMSE(input, target, optimalFilter);

        Assert.True(mse < 0.1, $"Optimal Wiener filter MSE = {mse}, expected < 0.1");
    }
}
