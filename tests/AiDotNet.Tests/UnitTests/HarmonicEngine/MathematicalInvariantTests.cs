using AiDotNet.HarmonicEngine.Core;
using AiDotNet.HarmonicEngine.Learning;
using AiDotNet.HarmonicEngine.Transforms;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Mathematically invariant tests that prove the core HRE properties hold.
/// These are deterministic tests based on mathematical identities — they MUST pass
/// regardless of implementation details because they test mathematical truth.
/// </summary>
public class MathematicalInvariantTests
{
    private readonly ITestOutputHelper _output;
    private const double Tolerance = 1e-6;

    public MathematicalInvariantTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // ================================================================
    // THEOREM 1 INVARIANTS: IMD-Attention Equivalence
    // ================================================================

    [Fact]
    public void IMD_ProductToSum_Identity_HoldsForAllCarrierPairs()
    {
        // Mathematical identity: cos(a)*cos(b) = 0.5*[cos(a-b) + cos(a+b)]
        // This is the foundation of IMD-as-attention.
        // Verify numerically for all carrier pairs.

        int fftSize = 256;
        int numSamples = fftSize;
        var fft = new FastFourierTransform<double>();

        double f1 = 7, f2 = 19; // Two carrier frequencies
        double a1 = 2.0, a2 = 3.0; // Two amplitudes

        // Build composite signal
        var signal = new Vector<double>(numSamples);
        for (int t = 0; t < numSamples; t++)
        {
            signal[t] = a1 * Math.Cos(2 * Math.PI * f1 * t / numSamples)
                       + a2 * Math.Cos(2 * Math.PI * f2 * t / numSamples);
        }

        // Square the signal (quadratic nonlinearity)
        var squared = new Vector<double>(numSamples);
        for (int t = 0; t < numSamples; t++)
        {
            squared[t] = signal[t] * signal[t];
        }

        // FFT of squared signal
        var spectrum = fft.Forward(squared);

        // Product-to-sum identity predicts:
        // - Energy at f1+f2 with amplitude a1*a2
        // - Energy at |f1-f2| with amplitude a1*a2
        // - Energy at 2*f1 with amplitude a1^2/2
        // - Energy at 2*f2 with amplitude a2^2/2
        // - DC component with (a1^2 + a2^2)/2

        int sumBin = (int)(f1 + f2);    // 26
        int diffBin = (int)Math.Abs(f1 - f2); // 12

        double sumMag = spectrum[sumBin].Magnitude;
        double diffMag = spectrum[diffBin].Magnitude;

        // Both should have non-zero energy (IMD products must exist)
        Assert.True(sumMag > 1e-6, $"Sum-frequency IMD at bin {sumBin} should have non-zero energy, got {sumMag}");
        Assert.True(diffMag > 1e-6, $"Diff-frequency IMD at bin {diffBin} should have non-zero energy, got {diffMag}");

        // Both should be proportional to a1*a2 = 6.0
        // The exact amplitude depends on FFT normalization, but they should be EQUAL to each other
        Assert.Equal(sumMag, diffMag, 4);

        _output.WriteLine($"Sum bin ({sumBin}) magnitude: {sumMag:F6}");
        _output.WriteLine($"Diff bin ({diffBin}) magnitude: {diffMag:F6}");
        _output.WriteLine($"Ratio (should be 1.0): {sumMag / diffMag:F6}");
    }

    [Fact]
    public void IMD_InteractionMatrix_IsSymmetric()
    {
        // Mathematical invariant: a_i * a_j = a_j * a_i
        // The interaction matrix MUST be symmetric.

        int n = 8;
        int fftSize = 1024;
        var allocator = new CarrierAllocator();
        var carriers = allocator.AllocateCarriers(n, fftSize);
        var bus = new SpectralBus<double>(carriers, fftSize);
        var extractor = new IMDExtractor<double>(carriers, fftSize);

        var amplitudes = new Vector<double>(n);
        for (int i = 0; i < n; i++) amplitudes[i] = (i + 1) * 0.7;

        var encoded = bus.Encode(amplitudes);
        var squared = new Vector<double>(encoded.Length);
        for (int i = 0; i < encoded.Length; i++) squared[i] = encoded[i] * encoded[i];

        var M = extractor.ExtractPairwise(squared);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Assert.Equal(M[i, j], M[j, i], 10);
            }
        }
    }

    [Fact]
    public void IMD_AttentionWeights_AreValidProbabilityDistribution()
    {
        // Mathematical invariant: softmax rows sum to 1, all entries in [0, 1]

        int n = 8;
        int fftSize = 1024;
        var allocator = new CarrierAllocator();
        var carriers = allocator.AllocateCarriers(n, fftSize);
        var bus = new SpectralBus<double>(carriers, fftSize);
        var extractor = new IMDExtractor<double>(carriers, fftSize);

        var amplitudes = new Vector<double>(n);
        for (int i = 0; i < n; i++) amplitudes[i] = 1.0 + i;

        var encoded = bus.Encode(amplitudes);
        var squared = new Vector<double>(encoded.Length);
        for (int i = 0; i < encoded.Length; i++) squared[i] = encoded[i] * encoded[i];

        var W = extractor.ExtractAttentionWeights(squared);

        for (int i = 0; i < n; i++)
        {
            double rowSum = 0;
            for (int j = 0; j < n; j++)
            {
                Assert.True(W[i, j] >= 0, $"W[{i},{j}] = {W[i, j]} must be >= 0");
                Assert.True(W[i, j] <= 1 + 1e-10, $"W[{i},{j}] = {W[i, j]} must be <= 1");
                rowSum += W[i, j];
            }
            Assert.Equal(1.0, rowSum, 6);
        }
    }

    // ================================================================
    // THEOREM 2 INVARIANTS: Spectral Sparsity
    // ================================================================

    [Fact]
    public void Sparsity_EnergyRatio_MonotonicallyIncreasing()
    {
        // Mathematical invariant: adding more components can never decrease total energy
        var mask = new SpectralSparsityMask<double>();
        var fft = new FastFourierTransform<double>();
        int n = 128;

        var signal = new Vector<double>(n);
        for (int t = 0; t < n; t++)
        {
            signal[t] = Math.Sin(2 * Math.PI * 3 * t / n)
                       + 0.7 * Math.Cos(2 * Math.PI * 11 * t / n)
                       + 0.3 * Math.Sin(2 * Math.PI * 23 * t / n);
        }

        var spectrum = fft.Forward(signal);
        double prevRatio = 0;

        for (int k = 1; k <= n; k++)
        {
            double ratio = mask.EnergyRatio(spectrum, k);
            Assert.True(ratio >= prevRatio - 1e-12,
                $"Energy ratio must be non-decreasing: K={k}, ratio={ratio:F8} < prev={prevRatio:F8}");
            prevRatio = ratio;
        }

        // At K=N, must capture exactly 100%
        Assert.Equal(1.0, mask.EnergyRatio(spectrum, n), 8);
    }

    [Fact]
    public void Sparsity_TopK_PreservesTopKComponents()
    {
        // Mathematical invariant: applying top-K mask and re-extracting top-K gives same result
        var mask = new SpectralSparsityMask<double>();
        int n = 64;
        int k = 5;

        // Build spectrum with known sparsity
        var spectrum = new Vector<Complex<double>>(n);
        var zero = new Complex<double>(0, 0);
        for (int i = 0; i < n; i++) spectrum[i] = zero;
        spectrum[3] = new Complex<double>(10, 2);
        spectrum[7] = new Complex<double>(8, -1);
        spectrum[15] = new Complex<double>(6, 3);
        spectrum[22] = new Complex<double>(4, 0);
        spectrum[31] = new Complex<double>(2, -2);

        var sparse = mask.Apply(spectrum, k);

        // Verify the top-K bins were kept: bins 3, 7, 15 have the largest magnitudes
        // |spectrum[3]| = sqrt(104) ≈ 10.2, |spectrum[7]| = sqrt(65) ≈ 8.06, |spectrum[15]| = sqrt(45) ≈ 6.7
        var topIndices = mask.GetTopKIndices(spectrum, k);
        Assert.Contains(3, topIndices);
        Assert.Contains(7, topIndices);
        Assert.Contains(15, topIndices);

        // Verify that non-top-K bins are zeroed out
        int nonZeroCount = 0;
        for (int i = 0; i < n; i++)
        {
            if (sparse[i].Magnitude > 1e-10)
                nonZeroCount++;
        }
        Assert.True(nonZeroCount <= k,
            $"After top-{k} sparsity, should have at most {k} non-zero bins, got {nonZeroCount}");

        // Idempotency: applying top-K twice gives the same result
        var doubleSparse = mask.Apply(sparse, k);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(sparse[i].Real, doubleSparse[i].Real, 10);
            Assert.Equal(sparse[i].Imaginary, doubleSparse[i].Imaginary, 10);
        }
    }

    // ================================================================
    // THEOREM 3 INVARIANTS: Hebbian Learning / Wiener Filter
    // ================================================================

    [Fact]
    public void Wiener_OptimalFilter_MinimizesMSE()
    {
        // Mathematical invariant: the Wiener filter achieves the global minimum MSE
        // for linear filtering. Any perturbation should increase MSE.

        var wiener = new WienerFilterRule<double>();
        int n = 64;

        var input = new Vector<double>(n);
        var target = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * 5 * i / n) + 0.3 * Math.Cos(2 * Math.PI * 11 * i / n);
            target[i] = 0.8 * Math.Sin(2 * Math.PI * 5 * i / n) + 1.5 * Math.Cos(2 * Math.PI * 11 * i / n);
        }

        var optimalFilter = wiener.ComputeOptimal(input, target);
        double optimalMSE = wiener.ComputeMSE(input, target, optimalFilter);

        // Perturb the filter slightly and verify MSE increases
        var rng = new Random(42);
        for (int trial = 0; trial < 10; trial++)
        {
            var perturbedFilter = new Vector<Complex<double>>(n);
            for (int k = 0; k < n; k++)
            {
                perturbedFilter[k] = new Complex<double>(
                    optimalFilter[k].Real + (rng.NextDouble() - 0.5) * 0.1,
                    optimalFilter[k].Imaginary + (rng.NextDouble() - 0.5) * 0.1);
            }

            double perturbedMSE = wiener.ComputeMSE(input, target, perturbedFilter);
            Assert.True(perturbedMSE >= optimalMSE - 1e-10,
                $"Perturbed MSE ({perturbedMSE:F8}) should be >= optimal ({optimalMSE:F8})");
        }

        _output.WriteLine($"Optimal MSE: {optimalMSE:E6}");
    }

    [Fact]
    public void CrossSpectral_ParsevalTheorem_EnergyConserved()
    {
        // Parseval's theorem: sum of |x(t)|^2 = (1/N) * sum of |X(k)|^2
        // Energy in time domain = energy in frequency domain (up to normalization)

        var fft = new FastFourierTransform<double>();
        int n = 64;

        var signal = new Vector<double>(n);
        for (int t = 0; t < n; t++)
        {
            signal[t] = Math.Sin(2 * Math.PI * 3 * t / n) + 0.5 * Math.Cos(2 * Math.PI * 7 * t / n);
        }

        // Time-domain energy
        double timeEnergy = 0;
        for (int t = 0; t < n; t++)
        {
            timeEnergy += signal[t] * signal[t];
        }

        // Frequency-domain energy
        var spectrum = fft.Forward(signal);
        double freqEnergy = 0;
        for (int k = 0; k < n; k++)
        {
            double re = spectrum[k].Real;
            double im = spectrum[k].Imaginary;
            freqEnergy += re * re + im * im;
        }
        freqEnergy /= n; // Normalization factor

        Assert.Equal(timeEnergy, freqEnergy, 4);
        _output.WriteLine($"Time energy: {timeEnergy:F6}, Freq energy: {freqEnergy:F6}");
    }

    // ================================================================
    // OFDM INVARIANTS: Encode-Decode Round Trip
    // ================================================================

    [Fact]
    public void SpectralBus_EncodeDecode_RoundTrip_PreservesAmplitudes()
    {
        // Mathematical invariant: IFFT(FFT(x)) = x
        // Encoding and decoding should recover the original amplitudes

        int n = 8;
        int fftSize = 1024;
        var allocator = new CarrierAllocator();
        var carriers = allocator.AllocateCarriers(n, fftSize);
        var bus = new SpectralBus<double>(carriers, fftSize);

        var original = new Vector<double>(n);
        for (int i = 0; i < n; i++) original[i] = (i + 1) * 1.5;

        var encoded = bus.Encode(original);
        var recovered = bus.Decode(encoded);

        // Encode-Decode should recover amplitudes proportionally
        // (exact values depend on FFT normalization, but ratios should match)
        for (int i = 0; i < n; i++)
        {
            Assert.True(recovered[i] > 0,
                $"Recovered amplitude[{i}] = {recovered[i]} should be positive");
        }

        // Verify ratios match: recovered[i]/recovered[0] should equal original[i]/original[0]
        for (int i = 1; i < n; i++)
        {
            double expectedRatio = original[i] / original[0];
            double actualRatio = recovered[i] / recovered[0];
            Assert.Equal(expectedRatio, actualRatio, 3);
        }
    }

    [Fact]
    public void Sidon_AllPairwiseSums_AreDistinct()
    {
        // Mathematical invariant of Sidon sets: all pairwise sums a_i + a_j are distinct

        var allocator = new CarrierAllocator();
        int n = 12;
        int fftSize = 4096;
        var carriers = allocator.AllocateCarriers(n, fftSize);

        var sums = new HashSet<int>();
        bool allDistinct = true;

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                int sum = carriers[i] + carriers[j];
                if (!sums.Add(sum))
                {
                    allDistinct = false;
                    _output.WriteLine($"Collision: carriers[{i}]({carriers[i]}) + carriers[{j}]({carriers[j]}) = {sum}");
                }
            }
        }

        Assert.True(allDistinct, "All pairwise sums in a Sidon set must be distinct");
        _output.WriteLine($"Verified {sums.Count} distinct sums for {n} carriers");
    }

    // ================================================================
    // MELLIN-FOURIER INVARIANTS
    // ================================================================

    [Fact]
    public void MellinFourier_ScaleInvariance_MagnitudeSpectrumUnchanged()
    {
        // Mathematical invariant: |M{f(ax)}(s)| = |a|^(-s) * |M{f}(s)|
        // For the magnitude spectrum, scaling the input only changes overall normalization,
        // not the shape. Normalized fingerprints should match.

        var mellin = new MellinTransform<double>();
        int n = 64;

        var signal = new Vector<double>(n);
        for (int t = 0; t < n; t++)
        {
            signal[t] = Math.Sin(2 * Math.PI * 5 * t / n) + 0.7 * Math.Cos(2 * Math.PI * 11 * t / n);
        }

        // Scale by 3x
        var scaled = new Vector<double>(n);
        for (int t = 0; t < n; t++) scaled[t] = 3.0 * signal[t];

        var fp1 = mellin.ScaleInvariantFingerprint(signal);
        var fp2 = mellin.ScaleInvariantFingerprint(scaled);

        // Compute cosine similarity (should be very high for pure scaling)
        double dot = 0, norm1 = 0, norm2 = 0;
        for (int i = 0; i < n; i++)
        {
            dot += fp1[i] * fp2[i];
            norm1 += fp1[i] * fp1[i];
            norm2 += fp2[i] * fp2[i];
        }
        double cosSim = dot / (Math.Sqrt(norm1) * Math.Sqrt(norm2));

        // The Mellin magnitude spectrum is mathematically scale-invariant,
        // so the cosine similarity should be essentially 1.0 for a pure
        // amplitude scaling. 0.9 was too loose.
        Assert.True(cosSim > 0.999,
            $"Scale-invariant fingerprints should be essentially identical, cosine={cosSim:F6}");
        _output.WriteLine($"Scale invariance cosine similarity: {cosSim:F6}");
    }

    [Fact]
    public void AnalyticSignal_CosineInput_ImaginaryIsSine()
    {
        // Mathematical identity: Hilbert[cos(wt)] = sin(wt)
        // The analytic signal of cos is cos + i*sin = e^(iwt)

        var analytic = new AnalyticSignal<double>();
        int n = 128;
        double freq = 8;

        var signal = new Vector<double>(n);
        for (int t = 0; t < n; t++)
        {
            signal[t] = Math.Cos(2 * Math.PI * freq * t / n);
        }

        var result = analytic.Compute(signal);

        // Check middle of signal (away from edge effects)
        int start = n / 4;
        int end = 3 * n / 4;
        double maxError = 0;

        for (int t = start; t < end; t++)
        {
            double expectedImag = Math.Sin(2 * Math.PI * freq * t / n);
            double actualImag = result[t].Imaginary;
            double error = Math.Abs(expectedImag - actualImag);
            maxError = Math.Max(maxError, error);
        }

        Assert.True(maxError < 0.5,
            $"Hilbert[cos] should approximate sin, max error = {maxError:F6}");
        _output.WriteLine($"Hilbert transform max error in middle region: {maxError:F6}");
    }
}
