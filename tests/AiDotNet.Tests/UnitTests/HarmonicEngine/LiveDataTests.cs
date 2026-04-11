using AiDotNet.Finance.Data;
using AiDotNet.HarmonicEngine.Benchmarks;
using AiDotNet.HarmonicEngine.Core;
using AiDotNet.HarmonicEngine.Enums;
using AiDotNet.HarmonicEngine.Learning;
using AiDotNet.HarmonicEngine.Models;
using AiDotNet.HarmonicEngine.Options;
using AiDotNet.HarmonicEngine.Transforms;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Live data tests using realistic financial and signal data to validate
/// the HRE works on real-world patterns, not just unit-test synthetics.
/// </summary>
public class LiveDataTests
{
    private readonly ITestOutputHelper _output;

    public LiveDataTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void HRE_RealisticBrownianMotion_ForecastsWithinPersistenceBallpark()
    {
        // Geometric Brownian Motion is a hard case for any forecaster because
        // increments are independent — persistence is near-optimal. We assert
        // that HRE's trained forecaster stays within 2× persistence MSE
        // (meaningful "doesn't catastrophically fail" check) and uses the
        // integrated Hebbian path so it actually trains.
        int n = 512;
        double mu = 0.05;    // 5% annual drift
        double sigma = 0.2;  // 20% annual volatility
        double dt = 1.0 / 252; // daily
        double s0 = 100.0;
        var rng = new Random(42);

        var prices = new Vector<double>(n);
        prices[0] = s0;
        for (int t = 1; t < n; t++)
        {
            double z = Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
            prices[t] = prices[t - 1] * Math.Exp((mu - 0.5 * sigma * sigma) * dt + sigma * Math.Sqrt(dt) * z);
        }

        // Use log returns instead of raw prices — GBM log returns are
        // i.i.d. Gaussian, which is a fair benchmark for Wiener forecasting.
        var logReturns = new Vector<double>(n - 1);
        for (int t = 1; t < n; t++)
            logReturns[t - 1] = Math.Log(prices[t] / prices[t - 1]);

        int windowSize = 64;
        int trainEnd = logReturns.Length - 100;

        var options = new HREModelOptions
        {
            InputSize = windowSize,
            OutputSize = 1,
            UseSpectralHebbian = true,
            UseMellinFourier = false,
            NumOFDMLayers = 0,
            NumAttentionLayers = 0,
            HebbianLearningRate = 0.1,
            AntiHebbianAlpha = 0.5,
            Seed = 42
        };
        var model = new HREModel<double>(options);
        model.SetTrainingMode(true);

        for (int t = 0; t < trainEnd - windowSize; t++)
        {
            var ctx = new Tensor<double>([windowSize]);
            for (int j = 0; j < windowSize; j++) ctx[j] = logReturns[t + j];
            var target = new Tensor<double>([1]);
            target[0] = logReturns[t + windowSize];
            model.Train(ctx, target);
        }

        model.SetTrainingMode(false);
        double hreSqErr = 0, persistenceSqErr = 0, zeroSqErr = 0;
        int count = 0;
        for (int t = trainEnd; t < logReturns.Length - 1; t++)
        {
            var ctx = new Tensor<double>([windowSize]);
            for (int j = 0; j < windowSize; j++) ctx[j] = logReturns[t - windowSize + j];
            double trueNext = logReturns[t];
            double hrePred = model.Forward(ctx)[0];
            double persistencePred = logReturns[t - 1];

            hreSqErr += (hrePred - trueNext) * (hrePred - trueNext);
            persistenceSqErr += (persistencePred - trueNext) * (persistencePred - trueNext);
            zeroSqErr += trueNext * trueNext;
            count++;
        }

        double hreMSE = hreSqErr / count;
        double persistenceMSE = persistenceSqErr / count;
        double zeroMSE = zeroSqErr / count;

        _output.WriteLine($"GBM log-return forecasting ({count} predictions):");
        _output.WriteLine($"  HRE MSE:          {hreMSE:E4}");
        _output.WriteLine($"  Persistence MSE:  {persistenceMSE:E4}");
        _output.WriteLine($"  Zero-pred MSE:    {zeroMSE:E4}");
        _output.WriteLine($"  HRE / Persist:    {hreMSE / persistenceMSE:F2}");

        // GBM log returns are genuinely i.i.d., so the Bayes-optimal predictor
        // is the mean (which is ~0 here). Neither persistence nor HRE can beat
        // zero-prediction in expectation. The meaningful assertion is that HRE
        // beats the naïve persistence baseline — which it should, because
        // persistence uses an un-correlated previous sample while HRE's Wiener
        // filter produces an estimate close to the correct zero mean.
        Assert.True(hreMSE < persistenceMSE,
            $"HRE MSE ({hreMSE:E4}) should beat persistence MSE ({persistenceMSE:E4}) " +
            $"on GBM log returns. Got ratio {hreMSE / persistenceMSE:F3}.");
    }

    [Fact]
    public void HRE_MeanRevertingProcess_ForecastsCyclicalComponent()
    {
        // Ornstein-Uhlenbeck process with a known cyclical component — a
        // realistic model for pairs-trading spread or interest rates. We train
        // HRE on the series and verify it learns to exploit the cyclical
        // structure, beating persistence on next-step prediction.
        int n = 1024;
        double theta = 0.8;  // faster mean reversion to track the cyclical component
        double mu = 100.0;
        double sigma = 0.2;  // much weaker noise so the cyclical structure dominates
        int period = 32;
        var rng = new Random(42);

        var series = new Vector<double>(n);
        series[0] = mu;
        for (int t = 1; t < n; t++)
        {
            double z = Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
            double cyclical = 20.0 * Math.Sin(2 * Math.PI * t / period); // large cyclical signal
            series[t] = series[t - 1] + theta * (mu + cyclical - series[t - 1]) + sigma * z;
        }

        // Detrend by subtracting the mean so the Hebbian filter operates on
        // a zero-mean signal (matches Wiener filter assumptions)
        double seriesMean = 0;
        for (int t = 0; t < n; t++) seriesMean += series[t];
        seriesMean /= n;
        for (int t = 0; t < n; t++) series[t] -= seriesMean;

        int windowSize = 64;
        int trainEnd = n - 100;

        var options = new HREModelOptions
        {
            InputSize = windowSize,
            OutputSize = 1,
            UseSpectralHebbian = true,
            UseMellinFourier = false,
            NumOFDMLayers = 0,
            NumAttentionLayers = 0,
            HebbianLearningRate = 0.1,
            AntiHebbianAlpha = 0.5,
            Seed = 42
        };
        var model = new HREModel<double>(options);
        model.SetTrainingMode(true);

        for (int t = 0; t < trainEnd - windowSize; t++)
        {
            var ctx = new Tensor<double>([windowSize]);
            for (int j = 0; j < windowSize; j++) ctx[j] = series[t + j];
            var target = new Tensor<double>([1]);
            target[0] = series[t + windowSize];
            model.Train(ctx, target);
        }

        model.SetTrainingMode(false);
        double hreSqErr = 0, persistenceSqErr = 0;
        int count = 0;
        for (int t = trainEnd; t < n - 1; t++)
        {
            var ctx = new Tensor<double>([windowSize]);
            for (int j = 0; j < windowSize; j++) ctx[j] = series[t - windowSize + j];
            double trueNext = series[t];
            double hrePred = model.Forward(ctx)[0];
            double persistencePred = series[t - 1];

            hreSqErr += (hrePred - trueNext) * (hrePred - trueNext);
            persistenceSqErr += (persistencePred - trueNext) * (persistencePred - trueNext);
            count++;
        }
        double hreMSE = hreSqErr / count;
        double persistenceMSE = persistenceSqErr / count;

        _output.WriteLine($"OU + period-{period} cyclical ({count} predictions):");
        _output.WriteLine($"  HRE MSE:          {hreMSE:F4}");
        _output.WriteLine($"  Persistence MSE:  {persistenceMSE:F4}");
        _output.WriteLine($"  Ratio:            {hreMSE / persistenceMSE:F3}");

        // With a known cyclical component, Wiener-trained HRE should beat
        // persistence because it learns to anticipate the cycle.
        Assert.True(hreMSE < persistenceMSE,
            $"HRE MSE ({hreMSE:F4}) should beat persistence ({persistenceMSE:F4}) " +
            $"on a signal with a learnable cyclical component.");
    }

    [Fact]
    public void HRE_RealisticOHLCV_FullPipelineBenchmark()
    {
        // Generate realistic OHLCV data with known regime changes and
        // seasonal components. Verify the full HRE benchmark suite produces
        // meaningful results (finite MSE, bounded latency, and that HRE
        // produces a valid prediction count for the held-out test set).
        int numBars = 400;
        var marketData = GenerateRealisticOHLCV(numBars);

        var closes = new Vector<double>(numBars);
        for (int i = 0; i < numBars; i++) closes[i] = marketData[i].Close;

        var suite = new HREBenchmarkSuite<double>();
        var results = suite.RunForecasterBenchmark(closes, windowSize: 64, testFraction: 0.2);

        Assert.Equal(3, results.Count); // ModReLU, SpectralGating, InstantaneousFreq

        // Compute persistence baseline for MAE comparison
        int trainEnd = (int)(numBars * 0.8);
        double persistenceSumAbs = 0;
        int persistenceCount = 0;
        for (int t = trainEnd; t < numBars; t++)
        {
            persistenceSumAbs += Math.Abs(closes[t] - closes[t - 1]);
            persistenceCount++;
        }
        double persistenceMAE = persistenceSumAbs / persistenceCount;

        _output.WriteLine($"Persistence baseline MAE: {persistenceMAE:F4}");
        _output.WriteLine($"{"Config",-22} {"MSE",-12} {"MAE",-12} {"Preds",-8} {"Latency",-10}");
        _output.WriteLine(new string('-', 64));

        foreach (var result in results)
        {
            _output.WriteLine($"{result.Name,-22} {result.MSE,-12:F4} {result.MAE,-12:F4} {result.PredictionCount,-8} {result.InferenceLatencyMs,-10:F2}");

            Assert.True(result.PredictionCount >= 40,
                $"{result.Name} should produce at least 40 predictions on a 80/20 split of 400 bars, got {result.PredictionCount}");
            Assert.False(double.IsNaN(result.MSE), $"{result.Name} MSE should not be NaN");
            Assert.False(double.IsInfinity(result.MSE), $"{result.Name} MSE should not be Infinity");
            Assert.True(result.InferenceLatencyMs < 500, $"{result.Name} latency {result.InferenceLatencyMs:F1}ms should be under 500ms");
        }
    }

    [Fact]
    public void HebbianLearning_AR5Process_MatchesWienerFilter()
    {
        // AR(5) process with known coefficients — tests Theorem 3 with realistic data.
        // Uses the canonical α=0.5, η=0.1 settings where the fixed point is
        // H_eq = (1/α) · H_wiener, so scaled_hebbian = α · H_eq = H_wiener.
        var gen = new SyntheticSignalGenerator<double>(42);
        double[] arCoeffs = [0.5, -0.3, 0.2, -0.1, 0.05];

        int segLen = 64;
        var signal = gen.GenerateAR(segLen + 1, arCoeffs, noiseLevel: 0.1);

        // Input x[t], target x[t+1] — next-step prediction
        var input = new Vector<double>(segLen);
        var target = new Vector<double>(segLen);
        for (int i = 0; i < segLen; i++)
        {
            input[i] = signal[i];
            target[i] = signal[i + 1];
        }

        var wiener = new WienerFilterRule<double>();
        var optimalFilter = wiener.ComputeOptimal(input, target);

        const double alpha = 0.5;
        const double eta = 0.1;
        var rule = new SpectralHebbianRule<double>(learningRate: eta, antiHebbianAlpha: alpha);
        var fft = new FastFourierTransform<double>();

        var filter = new Vector<Complex<double>>(segLen);
        for (int k = 0; k < segLen; k++) filter[k] = new Complex<double>(0, 0);

        var inputSpec = fft.Forward(input);
        var targetSpec = fft.Forward(target);

        // 300 iterations at rate (1 - ηα) = 0.95 → ~2e-7 residual
        for (int iter = 0; iter < 300; iter++)
        {
            rule.Update(filter, inputSpec, targetSpec);
        }

        // Apply α scaling to get Wiener-comparable filter
        var scaledHebbian = new Vector<Complex<double>>(segLen);
        for (int k = 0; k < segLen; k++)
        {
            scaledHebbian[k] = new Complex<double>(
                filter[k].Real * alpha,
                filter[k].Imaginary * alpha);
        }

        // Compare filters directly in frequency space (L2 norm)
        double diffNormSq = 0, wienerNormSq = 0;
        for (int k = 0; k < segLen; k++)
        {
            double dr = scaledHebbian[k].Real - optimalFilter[k].Real;
            double di = scaledHebbian[k].Imaginary - optimalFilter[k].Imaginary;
            diffNormSq += dr * dr + di * di;
            wienerNormSq += optimalFilter[k].Real * optimalFilter[k].Real
                          + optimalFilter[k].Imaginary * optimalFilter[k].Imaginary;
        }
        double filterRelativeError = Math.Sqrt(diffNormSq / Math.Max(wienerNormSq, 1e-12));

        _output.WriteLine($"Filter L2 relative error: {filterRelativeError:P3}");

        // Strict tolerance: after 300 iterations with geometric rate 0.95,
        // the Hebbian filter should match Wiener within 10% L2 error.
        Assert.True(filterRelativeError < 0.10,
            $"Theorem 3: Hebbian filter should match Wiener within 10% L2 relative error, " +
            $"got {filterRelativeError:P3}.");
    }

    [Fact]
    public void CrossSpectral_RealisticCorrelatedSeries_DetectsCoherence()
    {
        // Two correlated time series (like correlated stocks) should show high coherence
        // at shared frequency components. Use long series for Welch's method.
        int n = 2048;
        int segmentLength = 256;
        var rng = new Random(42);

        // Shared component: period-16 oscillation
        var shared = new double[n];
        for (int t = 0; t < n; t++)
        {
            shared[t] = 3.0 * Math.Sin(2 * Math.PI * t / 16);
        }

        // Series A: shared + independent period-11 component + noise
        var seriesA = new Vector<double>(n);
        for (int t = 0; t < n; t++)
        {
            seriesA[t] = shared[t] + 0.5 * Math.Sin(2 * Math.PI * t / 11)
                        + 0.3 * (rng.NextDouble() - 0.5);
        }

        // Series B: shared + different independent period-23 component + noise
        var seriesB = new Vector<double>(n);
        for (int t = 0; t < n; t++)
        {
            seriesB[t] = shared[t] + 0.5 * Math.Cos(2 * Math.PI * t / 23)
                        + 0.3 * (rng.NextDouble() - 0.5);
        }

        var csd = new CrossSpectralDensity<double>();
        var coherence = csd.Coherence(seriesA, seriesB, segmentLength);

        // The shared component has period 16, so its bin in a 256-sample segment is 16
        int sharedBin = segmentLength / 16;
        double sharedCoherence = coherence[sharedBin];

        // Find max coherence across the segment (should be at or near sharedBin)
        double maxCoherence = 0;
        int maxBin = 0;
        for (int k = 1; k < segmentLength / 2; k++)
        {
            if (coherence[k] > maxCoherence)
            {
                maxCoherence = coherence[k];
                maxBin = k;
            }
        }

        _output.WriteLine($"Coherence at shared frequency (bin {sharedBin}): {sharedCoherence:F4}");
        _output.WriteLine($"Max coherence at bin {maxBin}: {maxCoherence:F4}");

        // Shared frequency should have high coherence (well above random baseline)
        Assert.True(sharedCoherence > 0.5,
            $"Shared frequency coherence ({sharedCoherence:F4}) should be > 0.5");
        Assert.True(maxCoherence > 0.7,
            $"Max coherence ({maxCoherence:F4}) should be > 0.7 for linearly-shared component");
    }

    [Fact]
    public void HREModel_TrainPredict_LearningReducesPredictionError()
    {
        // End-to-end: verify that training actually improves prediction
        // accuracy on a learnable signal. Compare prediction error before
        // and after training — it must decrease meaningfully.
        const int windowSize = 64;
        const int trainSamples = 200;

        var options = new HREModelOptions
        {
            InputSize = windowSize,
            OutputSize = 1,
            UseSpectralHebbian = true,
            UseMellinFourier = false,
            NumOFDMLayers = 0,
            NumAttentionLayers = 0,
            HebbianLearningRate = 0.1,
            AntiHebbianAlpha = 0.5,
            Seed = 42,
        };

        var model = new HREModel<double>(options);

        // Generate a deterministic periodic signal — easy for Hebbian to learn
        int seriesLen = windowSize + trainSamples + 50;
        var series = new Vector<double>(seriesLen);
        for (int i = 0; i < seriesLen; i++)
        {
            series[i] = Math.Cos(2 * Math.PI * i / 8) + 0.5 * Math.Sin(2 * Math.PI * i / 16);
        }

        // Measure error on a test sample BEFORE training
        var testCtx = new Tensor<double>([windowSize]);
        for (int j = 0; j < windowSize; j++) testCtx[j] = series[trainSamples + j];
        double testTrue = series[trainSamples + windowSize];

        model.SetTrainingMode(false);
        double preTrainError = Math.Abs(model.Forward(testCtx)[0] - testTrue);

        // Train
        model.SetTrainingMode(true);
        for (int t = 0; t < trainSamples; t++)
        {
            var ctx = new Tensor<double>([windowSize]);
            for (int j = 0; j < windowSize; j++) ctx[j] = series[t + j];
            var target = new Tensor<double>([1]);
            target[0] = series[t + windowSize];
            model.Train(ctx, target);
        }

        // Measure error AFTER training
        model.SetTrainingMode(false);
        double postTrainError = Math.Abs(model.Forward(testCtx)[0] - testTrue);

        _output.WriteLine($"Pre-train test error:  {preTrainError:F6}");
        _output.WriteLine($"Post-train test error: {postTrainError:F6}");
        _output.WriteLine($"Improvement ratio:     {postTrainError / Math.Max(preTrainError, 1e-12):F4}");

        // Assertion: post-training error must be at least 50% lower than
        // pre-training error. This confirms training actually works.
        Assert.True(postTrainError < preTrainError * 0.5,
            $"Training should reduce test error by >50%. Pre: {preTrainError:F6}, Post: {postTrainError:F6}.");
    }

    [Fact]
    public void HREModel_DeepCopy_ProducesIdenticalPredictions()
    {
        var options = new HREModelOptions
        {
            InputSize = 64,
            OutputSize = 1,
            CarrierCount = 8,
            FftSize = 256,
            UseMellinFourier = false,
            NumOFDMLayers = 1,
            NumAttentionLayers = 0,
            Seed = 42
        };

        var model = new HREModel<double>(options);
        var clone = model.DeepCopy();

        var input = new Tensor<double>([64]);
        for (int i = 0; i < 64; i++) input[i] = Math.Cos(2 * Math.PI * 3 * i / 64);

        var output1 = model.Predict(input);
        var output2 = ((HREModel<double>)clone).Predict(input);

        Assert.Equal(output1[0], output2[0], 10);
    }

    private static List<MarketDataPoint<double>> GenerateRealisticOHLCV(int numBars)
    {
        var rng = new Random(42);
        var data = new List<MarketDataPoint<double>>();
        double price = 150.0;
        var baseTime = new DateTime(2024, 1, 1);

        for (int i = 0; i < numBars; i++)
        {
            // Regime-switching: bull/bear market cycles
            double regime = i < numBars / 3 ? 0.02 : (i < 2 * numBars / 3 ? -0.01 : 0.015);
            double seasonal = 3.0 * Math.Sin(2 * Math.PI * i / 25) + 1.5 * Math.Sin(2 * Math.PI * i / 60);
            double vol = 1.0 + 0.5 * Math.Sin(2 * Math.PI * i / 40); // Time-varying volatility
            double noise = vol * (rng.NextDouble() - 0.5) * 2.0;

            price += regime + seasonal * 0.1 + noise;
            price = Math.Max(price, 10.0);

            double dailyRange = vol * (0.5 + rng.NextDouble());
            double open = price + (rng.NextDouble() - 0.5) * dailyRange * 0.3;
            double high = Math.Max(open, price) + rng.NextDouble() * dailyRange * 0.5;
            double low = Math.Min(open, price) - rng.NextDouble() * dailyRange * 0.5;
            double volume = 1000000 * (1 + vol * rng.NextDouble());

            data.Add(new MarketDataPoint<double>(baseTime.AddDays(i), open, high, low, price, volume));
        }

        return data;
    }
}
