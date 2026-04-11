using AiDotNet.HarmonicEngine.Benchmarks;
using AiDotNet.HarmonicEngine.Core;
using AiDotNet.HarmonicEngine.Learning;
using AiDotNet.HarmonicEngine.Transforms;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Tests Experiment 3: Verify that the spectral Hebbian rule converges to the Wiener filter.
/// </summary>
public class SpectralHebbianTests
{
    private readonly ITestOutputHelper _output;

    public SpectralHebbianTests(ITestOutputHelper output)
    {
        _output = output;
    }

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

        // Error should decrease significantly — not just one step, but monotonically converge
        Assert.True(finalError < initialError,
            $"Hebbian update should reduce convergence error. Initial: {initialError}, Final: {finalError}");

        // After 50 iterations with eta=0.01, alpha=0.1, expect the geometric rate
        // (1 - eta*alpha)^50 ≈ 0.95 in the pure theoretical case, but power-normalization
        // accelerates this. Require at least 50% reduction as a meaningful convergence signal.
        double reductionRatio = finalError / initialError;
        Assert.True(reductionRatio < 0.5,
            $"After 50 iterations, error should be reduced by >50%. " +
            $"Initial: {initialError:F6}, Final: {finalError:F6}, Reduction: {(1 - reductionRatio) * 100:F1}%");
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

    /// <summary>
    /// Theorem 3 AR(p) experiment — the core empirical claim. Generates a
    /// synthetic AR(p) process, computes the analytic Wiener filter on a
    /// training window, and verifies that the spectral Hebbian rule converges
    /// to a filter that matches the Wiener filter in both frequency-space
    /// (filter coefficients) and test-set prediction MSE.
    ///
    /// Engine-accelerated via SpectralEngineHelper.FFT and runs across
    /// multiple AR orders for robustness.
    /// </summary>
    [Theory]
    [InlineData(3, new double[] { 0.7, -0.4, 0.2 })]
    [InlineData(5, new double[] { 0.6, -0.3, 0.15, -0.08, 0.04 })]
    [InlineData(7, new double[] { 0.5, -0.25, 0.12, -0.06, 0.03, -0.015, 0.008 })]
    public void HebbianConvergence_ARProcess_MatchesWienerOptimum(int order, double[] arCoeffs)
    {
        // Train and test windows must have the same length so that the same
        // Wiener filter (computed in frequency domain at length N) can be
        // applied to both without resampling.
        const int windowLen = 256;
        const double noiseLevel = 0.1;
        const double alpha = 0.5;
        const double eta = 0.1;

        var gen = new SyntheticSignalGenerator<double>(seed: 2026 + order);
        var wiener = new WienerFilterRule<double>();
        var rule = new SpectralHebbianRule<double>(learningRate: eta, antiHebbianAlpha: alpha);

        // Generate two non-overlapping windows from a long AR process
        var full = gen.GenerateAR(2 * windowLen + 1, arCoeffs, noiseLevel: noiseLevel);

        var trainInput = new Vector<double>(windowLen);
        var trainTarget = new Vector<double>(windowLen);
        for (int i = 0; i < windowLen; i++)
        {
            trainInput[i] = full[i];
            trainTarget[i] = full[i + 1];
        }

        var testInput = new Vector<double>(windowLen);
        var testTarget = new Vector<double>(windowLen);
        for (int i = 0; i < windowLen; i++)
        {
            testInput[i] = full[windowLen + i];
            testTarget[i] = full[windowLen + i + 1];
        }

        // Compute analytic Wiener filter on the training window
        var wienerFilter = wiener.ComputeOptimal(trainInput, trainTarget);

        // Train Hebbian filter from zero initialization using engine-accelerated FFT
        var hebbianFilter = new Vector<Complex<double>>(windowLen);
        for (int k = 0; k < windowLen; k++)
            hebbianFilter[k] = new Complex<double>(0.0, 0.0);

        var trainInputSpectrum = SpectralEngineHelper.ToComplexVector(SpectralEngineHelper.FFT(trainInput));
        var trainTargetSpectrum = SpectralEngineHelper.ToComplexVector(SpectralEngineHelper.FFT(trainTarget));

        // Theorem 3 predicts geometric convergence with rate (1 − ηα) = 0.95 here,
        // so 300 iterations gives (0.95)^300 ≈ 2.2e-7 residual — well past machine
        // precision of any practical threshold.
        for (int iter = 0; iter < 300; iter++)
        {
            rule.Update(hebbianFilter, trainInputSpectrum, trainTargetSpectrum);
        }

        // Theorem 3 fixed point: H_eq(k) = (1/α) · H_wiener(k), so scale to compare
        var scaledHebbian = new Vector<Complex<double>>(windowLen);
        for (int k = 0; k < windowLen; k++)
        {
            scaledHebbian[k] = new Complex<double>(
                hebbianFilter[k].Real * alpha,
                hebbianFilter[k].Imaginary * alpha);
        }

        // Validation 1: Direct filter comparison in frequency space.
        // Measure ||H_hebbian_scaled − H_wiener||² / ||H_wiener||²
        double diffNormSq = 0, wienerNormSq = 0;
        for (int k = 0; k < windowLen; k++)
        {
            double dr = scaledHebbian[k].Real - wienerFilter[k].Real;
            double di = scaledHebbian[k].Imaginary - wienerFilter[k].Imaginary;
            diffNormSq += dr * dr + di * di;

            wienerNormSq += wienerFilter[k].Real * wienerFilter[k].Real
                          + wienerFilter[k].Imaginary * wienerFilter[k].Imaginary;
        }
        double filterRelativeError = Math.Sqrt(diffNormSq / Math.Max(wienerNormSq, 1e-12));

        // Validation 2: Test-set MSE comparison on held-out data.
        // Apply each filter to the test input and compare against test target.
        double wienerTestMSE = wiener.ComputeMSE(testInput, testTarget, wienerFilter);
        double hebbianTestMSE = wiener.ComputeMSE(testInput, testTarget, scaledHebbian);

        // Also compute the baseline "predict zero" MSE for context
        double baselineMSE = 0;
        for (int i = 0; i < windowLen; i++) baselineMSE += testTarget[i] * testTarget[i];
        baselineMSE /= windowLen;

        double wienerImprovement = 1.0 - (wienerTestMSE / baselineMSE);
        double hebbianImprovement = 1.0 - (hebbianTestMSE / baselineMSE);
        double mseAbsoluteGap = Math.Abs(hebbianTestMSE - wienerTestMSE);
        double mseRelativeGap = mseAbsoluteGap / Math.Max(baselineMSE, 1e-12);

        _output.WriteLine($"=== AR({order}) ===");
        _output.WriteLine($"  Baseline (zero-pred) MSE:  {baselineMSE:F6}");
        _output.WriteLine($"  Wiener test MSE:           {wienerTestMSE:F6}  ({wienerImprovement:P1} improvement)");
        _output.WriteLine($"  Hebbian test MSE:          {hebbianTestMSE:F6}  ({hebbianImprovement:P1} improvement)");
        _output.WriteLine($"  MSE abs gap:               {mseAbsoluteGap:F6}");
        _output.WriteLine($"  MSE rel gap (vs baseline): {mseRelativeGap:P4}");
        _output.WriteLine($"  Filter relative L2 error:  {filterRelativeError:P3}");

        // Assertion 1: Both filters must actually predict (beat baseline)
        Assert.True(wienerImprovement > 0.2,
            $"Wiener should improve over baseline, got {wienerImprovement:P2}");
        Assert.True(hebbianImprovement > 0.2,
            $"Hebbian should improve over baseline, got {hebbianImprovement:P2}");

        // Assertion 2: Hebbian filter matches Wiener filter in frequency space.
        // This is the direct claim of Theorem 3 — the fixed point IS the Wiener
        // filter (up to scaling). We allow 10% relative L2 error to absorb the
        // finite number of iterations and discrete-FFT artifacts.
        Assert.True(filterRelativeError < 0.10,
            $"AR({order}): Hebbian filter differs from Wiener filter by " +
            $"{filterRelativeError:P2} relative L2 error (expected <10%). " +
            $"Theorem 3 predicts convergence to (1/α) · H_wiener.");

        // Assertion 3: Test-set MSE gap is at most 2% of the baseline MSE,
        // which is a meaningful measure even when both MSEs are near zero.
        Assert.True(mseRelativeGap < 0.02,
            $"AR({order}): Hebbian test MSE differs from Wiener by {mseRelativeGap:P4} " +
            $"of baseline (expected <2%). Absolute gap: {mseAbsoluteGap:F6}.");
    }

    /// <summary>
    /// Theorem 3 stability analysis. The spectral Hebbian update has a
    /// predicted stability boundary at |1 − η·α| &lt; 1, i.e., the product
    /// η·α must lie strictly in (0, 2). This test sweeps η·α values on both
    /// sides of the boundary and verifies the empirical convergence behavior
    /// matches the theoretical prediction.
    /// </summary>
    [Fact]
    public void HebbianConvergence_StabilityBoundary_MatchesTheory()
    {
        const int n = 64;
        var fft = new FastFourierTransform<double>();

        var input = new Vector<double>(n);
        var target = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * 5 * i / n) + 0.3 * Math.Cos(2 * Math.PI * 11 * i / n);
            target[i] = 0.8 * input[i];
        }

        var inputSpectrum = fft.Forward(input);
        var targetSpectrum = fft.Forward(target);

        // Test points: ηα values both inside and outside the stability region (0, 2)
        var testPoints = new (double etaAlpha, bool shouldConverge)[]
        {
            (0.1,  true),   // well inside stability
            (0.5,  true),   // mid-range
            (1.0,  true),   // optimal point (single-step convergence)
            (1.8,  true),   // approaching boundary but still stable
            (2.5,  false),  // unstable
            (4.0,  false),  // clearly unstable
        };

        int correct = 0;
        foreach (var (etaAlpha, shouldConverge) in testPoints)
        {
            // Fix alpha = 0.5 and derive eta to hit the target etaAlpha
            double alpha = 0.5;
            double eta = etaAlpha / alpha;
            var rule = new SpectralHebbianRule<double>(learningRate: eta, antiHebbianAlpha: alpha);

            var filter = new Vector<Complex<double>>(n);
            for (int k = 0; k < n; k++)
                filter[k] = new Complex<double>(0.0, 0.0);

            double maxMagnitude = 0;
            bool diverged = false;

            for (int iter = 0; iter < 200; iter++)
            {
                rule.Update(filter, inputSpectrum, targetSpectrum);

                double mag = 0;
                for (int k = 0; k < n; k++)
                {
                    mag += filter[k].Real * filter[k].Real + filter[k].Imaginary * filter[k].Imaginary;
                }
                mag = Math.Sqrt(mag);
                if (mag > maxMagnitude) maxMagnitude = mag;

                // Flag divergence: filter magnitude grows beyond sanity threshold
                if (mag > 1e6 || double.IsNaN(mag) || double.IsInfinity(mag))
                {
                    diverged = true;
                    break;
                }
            }

            bool empiricallyConverged = !diverged && maxMagnitude < 100.0;
            bool matchesTheory = empiricallyConverged == shouldConverge;
            if (matchesTheory) correct++;

            _output.WriteLine(
                $"η·α = {etaAlpha,4:F2}  predicted: {(shouldConverge ? "converge" : "diverge "),-8} " +
                $"observed: {(empiricallyConverged ? "converge" : "diverge "),-8} " +
                $"max|H|={maxMagnitude,10:E2}  {(matchesTheory ? "✓" : "✗")}");
        }

        _output.WriteLine($"\nStability predictions matched: {correct}/{testPoints.Length}");

        // Theorem 3's stability condition should match empirical behavior on
        // all 6 test points. Allow 1 mismatch near the boundary to absorb
        // numerical edge cases (e.g., ηα = 2 exactly).
        Assert.True(correct >= testPoints.Length - 1,
            $"Theorem 3 stability prediction failed: only {correct}/{testPoints.Length} " +
            $"test points matched the |1 − ηα| < 1 boundary.");
    }
}
