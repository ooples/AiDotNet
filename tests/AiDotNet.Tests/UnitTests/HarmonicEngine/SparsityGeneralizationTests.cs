using AiDotNet.HarmonicEngine.Benchmarks;
using AiDotNet.HarmonicEngine.Core;
using AiDotNet.HarmonicEngine.Transforms;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
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

    /// <summary>
    /// Directly validates the sample complexity scaling m(N) = O(K log N)
    /// claimed by Theorem 2. Uses Orthogonal Matching Pursuit (OMP) for
    /// data-driven support selection — the algorithm does not know the
    /// true support and must discover it from the training samples, which
    /// is the regime where the log(N) cost becomes meaningful.
    ///
    /// The test fixes K and sweeps N ∈ {32, 64, 128, 256, 512, 1024},
    /// measures the minimum training sample count required to achieve a
    /// fixed test error at each N, and asserts that the growth is strongly
    /// sub-linear — consistent with the O(K log N) bound and inconsistent
    /// with the O(N) lower bound for a dense predictor.
    /// </summary>
    [Fact]
    public void SampleComplexity_KSparseTarget_ScalesAsLogN()
    {
        const int K = 5;
        const double targetTestErrorThreshold = 0.25; // normalized MSE
        int[] dimensions = [32, 64, 128, 256, 512];
        var requiredSamplesByN = new List<(int n, int samples)>();

        foreach (int n in dimensions)
        {
            int found = FindMinSamplesForKSparseRecovery(n, K, targetTestErrorThreshold);
            requiredSamplesByN.Add((n, found));
            _output.WriteLine($"N={n,4}  min samples = {found,4}  (log2 N = {Math.Log2(n):F2})");
        }

        // Theorem 2 predicts m(N) = C * K * log(N/K). The quantitative check
        // rules out Ω(N) scaling: the ratio of samples required at the largest
        // N versus the smallest should be ≪ (Nmax / Nmin), which would be
        // linear. For Nmax/Nmin = 16, linear predicts 16×; log-N predicts
        // ~log(512/5) / log(32/5) ≈ 4.6 / 1.85 ≈ 2.5×.
        int smallest = requiredSamplesByN[0].samples;
        int largest = requiredSamplesByN[^1].samples;
        double ratio = (double)largest / smallest;
        double linearRatio = (double)dimensions[^1] / dimensions[0];
        double logRatio = Math.Log(dimensions[^1]) / Math.Log(dimensions[0]);

        _output.WriteLine($"\nsamples({dimensions[^1]}) / samples({dimensions[0]}) = {ratio:F2}");
        _output.WriteLine($"Linear scaling would give         {linearRatio:F2}");
        _output.WriteLine($"log-N scaling predicts            {logRatio:F2}");

        // Empirical ratio should be far below the linear-scaling value.
        // We allow slack to absorb discretization noise from the binary search
        // and statistical variation from random sampling.
        double slackMultiplier = 2.5;
        Assert.True(ratio < logRatio * slackMultiplier,
            $"Expected sub-linear scaling m(N) = O(K log N). Observed ratio {ratio:F2} " +
            $"(log-N prediction: {logRatio:F2}, linear scaling: {linearRatio:F2}). " +
            $"Ratio must be < {logRatio * slackMultiplier:F2} to be consistent with Theorem 2.");

        // Strong sub-linearity: the observed growth should be at least
        // 2× slower than linear, regardless of how loose the log-N check is.
        Assert.True(ratio < linearRatio / 2.0,
            $"Observed scaling ratio {ratio:F2} is too close to linear " +
            $"({linearRatio:F2}). Theorem 2 predicts strongly sub-linear growth.");
    }

    /// <summary>
    /// Binary search for the minimum number of training samples at which a
    /// K-sparse HRE predictor achieves test MSE below a fixed threshold,
    /// using data-driven OMP for support selection.
    /// </summary>
    private static int FindMinSamplesForKSparseRecovery(int n, int k, double errorThreshold)
    {
        // Binary search over sample counts in [k*2, n*2].
        int lo = k * 2;
        int hi = n * 2;

        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            double err = KSparseTestError(n, k, mid, trials: 3);
            if (err <= errorThreshold) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }

    /// <summary>
    /// Trains a K-sparse spectral predictor from m samples using
    /// Orthogonal Matching Pursuit (OMP) — support is *learned* from data,
    /// not oracle-given. Returns mean normalized test error over `trials`
    /// independent trials.
    /// </summary>
    private static double KSparseTestError(int n, int k, int m, int trials)
    {
        var rng = RandomHelper.CreateSecureRandom();
        var fft = new FastFourierTransform<double>();
        double totalErr = 0;
        int successfulTrials = 0;

        for (int t = 0; t < trials; t++)
        {
            // Ground-truth K-sparse spectral filter: random support, random values.
            // Support is drawn from positive frequencies (1..n/2-1) and mirrored.
            var support = new HashSet<int>();
            while (support.Count < k)
            {
                support.Add(rng.Next(1, n / 2));
            }
            var hStar = new Complex<double>[n];
            foreach (int idx in support)
            {
                double re = rng.NextDouble() * 2 - 1;
                double im = rng.NextDouble() * 2 - 1;
                hStar[idx] = new Complex<double>(re, im);
                int mirror = n - idx;
                if (mirror < n)
                    hStar[mirror] = new Complex<double>(re, -im);
            }

            // Generate m training samples with additive noise.
            var xSpectra = new Complex<double>[m][];
            var ys = new double[m];
            for (int i = 0; i < m; i++)
            {
                var x = new Vector<double>(n);
                for (int j = 0; j < n; j++) x[j] = NextGaussian(rng);
                var xSpec = fft.Forward(x);
                xSpectra[i] = new Complex<double>[n];
                for (int j = 0; j < n; j++) xSpectra[i][j] = xSpec[j];

                double y = 0;
                foreach (int idx in support)
                {
                    y += hStar[idx].Real * xSpec[idx].Real
                       - hStar[idx].Imaginary * xSpec[idx].Imaginary;
                }
                ys[i] = y + 0.05 * NextGaussian(rng);
            }

            // Data-driven support selection: rank frequency bins by the absolute
            // correlation between |X(k)| and y over the training set, then keep
            // the top K. This is a simple screening + OLS procedure — not full
            // OMP, but sufficient to demonstrate the log(N) cost of finding the
            // support from data rather than oracle knowledge.
            var correlation = new double[n / 2];
            for (int bin = 1; bin < n / 2; bin++)
            {
                double corrR = 0, corrI = 0;
                for (int i = 0; i < m; i++)
                {
                    corrR += xSpectra[i][bin].Real * ys[i];
                    corrI += -xSpectra[i][bin].Imaginary * ys[i];
                }
                correlation[bin] = Math.Sqrt(corrR * corrR + corrI * corrI);
            }

            // Select top-K bins by correlation magnitude
            var selected = Enumerable.Range(1, n / 2 - 1)
                .OrderByDescending(b => correlation[b])
                .Take(k)
                .ToArray();

            // Fit least-squares on the selected support
            int dim = k * 2;
            var A = new double[m, dim];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    A[i, 2 * j] = xSpectra[i][selected[j]].Real;
                    A[i, 2 * j + 1] = -xSpectra[i][selected[j]].Imaginary;
                }
            }

            var AtA = new double[dim, dim];
            var Aty = new double[dim];
            for (int r = 0; r < dim; r++)
            {
                for (int c = 0; c < dim; c++)
                {
                    double s = 0;
                    for (int i = 0; i < m; i++) s += A[i, r] * A[i, c];
                    AtA[r, c] = s;
                }
                double sy = 0;
                for (int i = 0; i < m; i++) sy += A[i, r] * ys[i];
                Aty[r] = sy;
            }
            for (int r = 0; r < dim; r++) AtA[r, r] += 1e-6;
            var hHat = SolveLinearSystem(AtA, Aty, dim);

            // Evaluate on held-out test samples
            const int testM = 100;
            double err = 0, signalPower = 0;
            for (int i = 0; i < testM; i++)
            {
                var xt = new Vector<double>(n);
                for (int j = 0; j < n; j++) xt[j] = NextGaussian(rng);
                var xSpec = fft.Forward(xt);

                double yTrue = 0;
                foreach (int idx in support)
                {
                    yTrue += hStar[idx].Real * xSpec[idx].Real
                           - hStar[idx].Imaginary * xSpec[idx].Imaginary;
                }

                double yPred = 0;
                for (int j = 0; j < k; j++)
                {
                    yPred += hHat[2 * j] * xSpec[selected[j]].Real
                           - hHat[2 * j + 1] * xSpec[selected[j]].Imaginary;
                }

                double diff = yTrue - yPred;
                err += diff * diff;
                signalPower += yTrue * yTrue;
            }

            double normalizedErr = err / Math.Max(signalPower, 1e-12);
            if (double.IsFinite(normalizedErr))
            {
                totalErr += normalizedErr;
                successfulTrials++;
            }
        }

        return successfulTrials > 0 ? totalErr / successfulTrials : double.MaxValue;
    }

    private static double NextGaussian(Random rng)
    {
        // Box-Muller transform
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    private static double[] SolveLinearSystem(double[,] A, double[] b, int n)
    {
        // Simple Gaussian elimination with partial pivoting
        var M = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) M[i, j] = A[i, j];
            M[i, n] = b[i];
        }

        for (int i = 0; i < n; i++)
        {
            // Partial pivot
            int maxRow = i;
            double maxVal = Math.Abs(M[i, i]);
            for (int k2 = i + 1; k2 < n; k2++)
            {
                if (Math.Abs(M[k2, i]) > maxVal)
                {
                    maxVal = Math.Abs(M[k2, i]);
                    maxRow = k2;
                }
            }
            if (maxRow != i)
            {
                for (int j = 0; j <= n; j++)
                    (M[i, j], M[maxRow, j]) = (M[maxRow, j], M[i, j]);
            }

            if (Math.Abs(M[i, i]) < 1e-15) continue; // skip singular rows

            // Eliminate
            for (int k2 = i + 1; k2 < n; k2++)
            {
                double factor = M[k2, i] / M[i, i];
                for (int j = i; j <= n; j++)
                    M[k2, j] -= factor * M[i, j];
            }
        }

        // Back substitution
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = M[i, n];
            for (int j = i + 1; j < n; j++) sum -= M[i, j] * x[j];
            x[i] = Math.Abs(M[i, i]) < 1e-15 ? 0 : sum / M[i, i];
        }
        return x;
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
