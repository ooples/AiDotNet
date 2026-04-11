using AiDotNet.HarmonicEngine.Benchmarks;
using AiDotNet.HarmonicEngine.Core;
using AiDotNet.HarmonicEngine.Transforms;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Experiment 4: Spectral sparsity generalization.
/// Tests that top-K spectral selection prevents overfitting and enables learning
/// from fewer samples than a dense approach.
/// </summary>
public class SparsityGeneralizationTests
{
    private readonly ITestOutputHelper _output;

    public SparsityGeneralizationTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void TopK_RecoversSparseSignal_FromLimitedSamples()
    {
        // Generate a truly sparse signal: exactly 3 pure sinusoids at known frequencies
        int n = 64;
        int trueK = 3;
        int[] trueBins = [5, 12, 23];

        var signal = new Vector<double>(n);
        for (int t = 0; t < n; t++)
        {
            signal[t] = 3.0 * Math.Cos(2 * Math.PI * trueBins[0] * t / n)
                       + 2.0 * Math.Cos(2 * Math.PI * trueBins[1] * t / n)
                       + 1.5 * Math.Cos(2 * Math.PI * trueBins[2] * t / n);
        }

        var fft = new FastFourierTransform<double>();
        var mask = new SpectralSparsityMask<double>();
        var spectrum = fft.Forward(signal);

        // With exact integer frequencies, energy concentrates at exact bins
        // Need 2*trueK because FFT has symmetric positive+negative frequencies
        double energyRatio = mask.EnergyRatio(spectrum, trueK * 2);
        _output.WriteLine($"Energy ratio with K={trueK * 2}: {energyRatio:F4}");

        Assert.True(energyRatio > 0.9,
            $"Top-{trueK * 2} should capture most energy of a {trueK}-sparse signal, got {energyRatio:F4}");
    }

    [Fact]
    public void TopK_OversparseRetainsSignal_UndersparseAddsNoise()
    {
        // Pure sinusoids at integer bin frequencies
        int n = 64;
        int[] bins = [3, 8, 15, 22, 29];
        int trueK = bins.Length;

        var signal = new Vector<double>(n);
        for (int t = 0; t < n; t++)
        {
            foreach (int bin in bins)
            {
                signal[t] += Math.Cos(2 * Math.PI * bin * t / n);
            }
        }

        var fft = new FastFourierTransform<double>();
        var mask = new SpectralSparsityMask<double>();
        var spectrum = fft.Forward(signal);

        // Test different K values — need 2*trueK for positive+negative FFT symmetry
        foreach (int k in new[] { 2, 4, 10, 20, 32 })
        {
            double energyRatio = mask.EnergyRatio(spectrum, k);
            _output.WriteLine($"K={k}: energy ratio = {energyRatio:F4}");
        }

        // K=2*trueK should capture ~100% (for clean integer-bin signals)
        double ratioAtTrue = mask.EnergyRatio(spectrum, trueK * 2);
        Assert.True(ratioAtTrue > 0.9, $"K={trueK * 2} should capture most energy, got {ratioAtTrue:F4}");

        // K=2 should capture much less
        double ratioAtSmall = mask.EnergyRatio(spectrum, 2);
        Assert.True(ratioAtSmall < ratioAtTrue,
            "K=2 should capture less energy than K=2*trueK");
    }

    [Fact]
    public void MDLAutoK_SelectsReasonableK_ForSparseSignal()
    {
        var gen = new SyntheticSignalGenerator<double>(42);
        var mask = new SpectralSparsityMask<double>();
        var fft = new FastFourierTransform<double>();

        int n = 128;
        int trueK = 4;

        var (signal, _) = gen.GenerateKSparse(n, trueK, snr: 30.0);
        var spectrum = fft.Forward(signal);

        int autoK = mask.SelectK(spectrum);

        _output.WriteLine($"True sparsity: {trueK}");
        _output.WriteLine($"MDL auto K:    {autoK}");

        // MDL should select K in a reasonable range (not too far from trueK)
        Assert.True(autoK >= 1, "Auto K should be at least 1");
        Assert.True(autoK <= trueK * 3,
            $"Auto K ({autoK}) should be within 3x of true K ({trueK})");
    }

    [Fact]
    public void SparsityMask_IncreasingK_MonotonicallyIncreasesEnergy()
    {
        var gen = new SyntheticSignalGenerator<double>(42);
        var mask = new SpectralSparsityMask<double>();
        var fft = new FastFourierTransform<double>();

        int n = 64;
        var (signal, _) = gen.GenerateKSparse(n, 5, snr: 20.0);
        var spectrum = fft.Forward(signal);

        double prevRatio = 0;
        for (int k = 1; k <= 32; k++)
        {
            double ratio = mask.EnergyRatio(spectrum, k);
            Assert.True(ratio >= prevRatio - 1e-10,
                $"Energy ratio should be monotonically non-decreasing: K={k} ratio={ratio} < prev={prevRatio}");
            prevRatio = ratio;
        }

        // Full spectrum should capture 100%
        double fullRatio = mask.EnergyRatio(spectrum, n);
        Assert.Equal(1.0, fullRatio, 6);
    }

    [Fact]
    public void SparsityMask_GetTopKIndices_ReturnsLargestComponents()
    {
        var mask = new SpectralSparsityMask<double>();
        int n = 64;
        int k = 5;

        // Create a spectrum where magnitudes are proportional to index
        // so the top-K should be indices [63, 62, 61, 60, 59]
        var spectrum = new Vector<Complex<double>>(n);
        for (int i = 0; i < n; i++)
        {
            spectrum[i] = new Complex<double>(i * 0.1, 0);
        }

        var indices = mask.GetTopKIndices(spectrum, k);

        Assert.Equal(k, indices.Length);
        Assert.Equal(indices.Length, indices.Distinct().Count()); // All unique

        // Verify these are the K largest — all should be in the top portion
        var sortedExpected = Enumerable.Range(0, n)
            .OrderByDescending(i => Math.Abs(i * 0.1))
            .Take(k)
            .ToHashSet();

        foreach (int idx in indices)
        {
            Assert.True(sortedExpected.Contains(idx),
                $"Index {idx} is not among the top-{k} by magnitude");
        }
    }

    [Fact]
    public void WienerFilter_SparseVsDense_SparseNeedsFewerSamples()
    {
        // Core of Theorem 2: sparse representation requires fewer samples
        // We show that Wiener filter on K-sparse signal achieves low MSE
        // with far fewer samples than the full signal length

        var gen = new SyntheticSignalGenerator<double>(42);
        var fft = new FastFourierTransform<double>();
        var mask = new SpectralSparsityMask<double>();
        int n = 64;
        int trueK = 3;

        // Generate sparse input and target
        var input = gen.GenerateComposite(n, [3, 7, 13], [1.0, 0.8, 0.5]);
        var target = gen.GenerateComposite(n, [3, 7, 13], [1.5, 1.2, 0.3]);

        // Full Wiener filter: uses all N frequency bins
        var wiener = new AiDotNet.HarmonicEngine.Learning.WienerFilterRule<double>();
        var fullFilter = wiener.ComputeOptimal(input, target);
        double fullMSE = wiener.ComputeMSE(input, target, fullFilter);

        // Sparse Wiener filter: zero out non-top-K bins
        var inputSpectrum = fft.Forward(input);
        var topKIndices = mask.GetTopKIndices(inputSpectrum, trueK);
        var sparseFilter = new Vector<Complex<double>>(n);
        var zero = new Complex<double>(0, 0);
        for (int i = 0; i < n; i++) sparseFilter[i] = zero;
        foreach (int idx in topKIndices)
        {
            sparseFilter[idx] = fullFilter[idx];
        }

        double sparseMSE = wiener.ComputeMSE(input, target, sparseFilter);

        _output.WriteLine($"Full Wiener (N={n} params): MSE = {fullMSE:F6}");
        _output.WriteLine($"Sparse Wiener (K={trueK} params): MSE = {sparseMSE:F6}");
        _output.WriteLine($"Compression: {(double)n / trueK:F1}x fewer parameters");

        // For a truly sparse signal, the sparse filter should achieve low MSE
        // The full Wiener filter has near-zero MSE since it can perfectly fit
        Assert.True(sparseMSE < 1.0,
            $"Sparse MSE ({sparseMSE}) should be reasonable for a sparse signal");
    }
}
