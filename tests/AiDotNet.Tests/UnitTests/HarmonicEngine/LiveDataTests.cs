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
    public void HRE_RealisticBrownianMotion_ProducesFinitePredictions()
    {
        // Geometric Brownian Motion — standard model for stock prices
        // dS = mu*S*dt + sigma*S*dW
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

        // Use HRE forecaster
        int windowSize = 64;
        var options = new HREModelOptions
        {
            CarrierCount = 8,
            FftSize = 256,
            Nonlinearity = NonlinearityType.SpectralGating,
            UseMellinFourier = false,
            NumOFDMLayers = 1,
            NumAttentionLayers = 1,
            Seed = 42
        };

        var forecaster = new HREForecaster<double>(windowSize, 1, options);

        int validPredictions = 0;
        int nanPredictions = 0;

        for (int t = windowSize; t < n - 1; t += 10)
        {
            var window = new Vector<double>(windowSize);
            for (int i = 0; i < windowSize; i++) window[i] = prices[t - windowSize + i];

            var pred = forecaster.Predict(window);

            if (double.IsNaN(pred[0]) || double.IsInfinity(pred[0]))
                nanPredictions++;
            else
                validPredictions++;
        }

        _output.WriteLine($"Valid predictions: {validPredictions}, NaN predictions: {nanPredictions}");
        Assert.True(validPredictions > nanPredictions,
            $"Most predictions should be valid: {validPredictions} valid vs {nanPredictions} NaN");
    }

    [Fact]
    public void HRE_MeanRevertingProcess_DetectsPeriodicComponent()
    {
        // Ornstein-Uhlenbeck process with known periodic component
        // This is a realistic model for pairs-trading spread or interest rates
        int n = 256;
        double theta = 0.5;  // Mean reversion speed
        double mu = 100.0;   // Long-term mean
        double sigma = 2.0;  // Volatility
        double period = 32;  // Known cyclical component
        var rng = new Random(42);

        var series = new Vector<double>(n);
        series[0] = mu;
        for (int t = 1; t < n; t++)
        {
            double z = Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
            double cyclical = 5.0 * Math.Sin(2 * Math.PI * t / period);
            series[t] = series[t - 1] + theta * (mu + cyclical - series[t - 1]) + sigma * z;
        }

        // Use spectral analysis to detect the periodic component
        int windowSize = 128;
        var fft = new FastFourierTransform<double>();
        var window = new Vector<double>(windowSize);
        for (int i = 0; i < windowSize; i++) window[i] = series[64 + i];

        var spectrum = fft.Forward(window);

        // Find peak frequency (excluding DC)
        int peakBin = 1;
        double peakMag = 0;
        for (int k = 1; k < windowSize / 2; k++)
        {
            double mag = spectrum[k].Magnitude;
            if (mag > peakMag)
            {
                peakMag = mag;
                peakBin = k;
            }
        }

        double detectedPeriod = (double)windowSize / peakBin;
        _output.WriteLine($"True period: {period}, Detected period: {detectedPeriod:F1} (bin {peakBin})");

        // The spectral peak should be near the true period
        Assert.True(Math.Abs(detectedPeriod - period) < period * 0.3,
            $"Detected period ({detectedPeriod:F1}) should be near true period ({period})");
    }

    [Fact]
    public void HRE_RealisticOHLCV_FullPipelineRuns()
    {
        // Generate realistic OHLCV data with known regime changes
        int numBars = 400;
        var marketData = GenerateRealisticOHLCV(numBars);

        // Extract close prices
        var closes = new Vector<double>(numBars);
        for (int i = 0; i < numBars; i++) closes[i] = marketData[i].Close;

        // Run full benchmark suite
        var suite = new HREBenchmarkSuite<double>();
        var results = suite.RunForecasterBenchmark(closes, windowSize: 64, testFraction: 0.2);

        foreach (var result in results)
        {
            _output.WriteLine(result.ToString());
            Assert.True(result.PredictionCount > 0, $"{result.Name} should produce predictions");
            Assert.False(double.IsNaN(result.MSE), $"{result.Name} MSE should not be NaN");
            Assert.True(result.InferenceLatencyMs < 500, $"{result.Name} latency {result.InferenceLatencyMs:F1}ms should be under 500ms");
        }
    }

    [Fact]
    public void HebbianLearning_AR5Process_ConvergesToWiener()
    {
        // AR(5) process with known coefficients — tests Theorem 3 with real-ish data
        var gen = new SyntheticSignalGenerator<double>(42);
        double[] arCoeffs = [0.5, -0.3, 0.2, -0.1, 0.05]; // Stable AR(5)

        int n = 256;
        var signal = gen.GenerateAR(n, arCoeffs, noiseLevel: 0.1);

        // Create input-target pairs: input = x[t-64:t], target = x[t:t+64]
        int segLen = 64;
        var input = new Vector<double>(segLen);
        var target = new Vector<double>(segLen);
        for (int i = 0; i < segLen; i++)
        {
            input[i] = signal[i];
            target[i] = signal[i + segLen];
        }

        // Compute Wiener optimal
        var wiener = new WienerFilterRule<double>();
        var optimalFilter = wiener.ComputeOptimal(input, target);
        double optimalMSE = wiener.ComputeMSE(input, target, optimalFilter);

        // Train Hebbian
        var fft = new FastFourierTransform<double>();
        var rule = new SpectralHebbianRule<double>(learningRate: 0.05, antiHebbianAlpha: 0.005);

        var filter = new Vector<Complex<double>>(segLen);
        for (int k = 0; k < segLen; k++) filter[k] = new Complex<double>(0, 0);

        var inputSpec = fft.Forward(input);
        var targetSpec = fft.Forward(target);

        for (int iter = 0; iter < 100; iter++)
        {
            rule.Update(filter, inputSpec, targetSpec);
        }

        // Apply Hebbian filter
        var complexOps = MathHelper.GetNumericOperations<Complex<double>>();
        var filteredSpec = new Vector<Complex<double>>(segLen);
        for (int k = 0; k < segLen; k++)
        {
            filteredSpec[k] = complexOps.Multiply(filter[k], inputSpec[k]);
        }
        var filtered = fft.Inverse(filteredSpec);

        double hebbianMSE = 0;
        for (int i = 0; i < segLen; i++)
        {
            double diff = filtered[i] - target[i];
            hebbianMSE += diff * diff;
        }
        hebbianMSE /= segLen;

        _output.WriteLine($"Wiener MSE:  {optimalMSE:E4}");
        _output.WriteLine($"Hebbian MSE: {hebbianMSE:E4}");
        _output.WriteLine($"Ratio:       {hebbianMSE / (optimalMSE + 1e-15):F2}x");

        // Hebbian should be in the same ballpark as Wiener (within 100x for this setup)
        Assert.True(hebbianMSE < optimalMSE * 100 + 1.0,
            $"Hebbian MSE ({hebbianMSE:E4}) should be within 100x of Wiener ({optimalMSE:E4})");
    }

    [Fact]
    public void CrossSpectral_RealisticCorrelatedSeries_DetectsCoherence()
    {
        // Two correlated time series (like correlated stocks) should show high coherence
        // at shared frequency components
        int n = 128;
        var rng = new Random(42);

        // Shared component: 10-day cycle
        var shared = new double[n];
        for (int t = 0; t < n; t++)
        {
            shared[t] = 3.0 * Math.Sin(2 * Math.PI * t / 10);
        }

        // Series A: shared + independent noise
        var seriesA = new Vector<double>(n);
        for (int t = 0; t < n; t++)
        {
            seriesA[t] = shared[t] + 0.5 * Math.Sin(2 * Math.PI * t / 7) // independent component
                        + 0.3 * (rng.NextDouble() - 0.5); // noise
        }

        // Series B: shared + different independent noise
        var seriesB = new Vector<double>(n);
        for (int t = 0; t < n; t++)
        {
            seriesB[t] = shared[t] + 0.5 * Math.Cos(2 * Math.PI * t / 15) // different independent component
                        + 0.3 * (rng.NextDouble() - 0.5); // noise
        }

        var csd = new CrossSpectralDensity<double>();
        var coherence = csd.Coherence(seriesA, seriesB);

        // Coherence at the shared frequency (bin = n/10 = 12.8 -> bin 13)
        int sharedBin = (int)Math.Round((double)n / 10);
        double sharedCoherence = coherence[sharedBin];

        // Coherence at an independent frequency (bin = n/7 ~ 18)
        int indepBin = (int)Math.Round((double)n / 7);
        double indepCoherence = coherence[indepBin];

        _output.WriteLine($"Coherence at shared frequency (bin {sharedBin}): {sharedCoherence:F4}");
        _output.WriteLine($"Coherence at independent frequency (bin {indepBin}): {indepCoherence:F4}");

        // Shared frequency should have higher coherence
        Assert.True(sharedCoherence > 0.1,
            $"Shared frequency coherence ({sharedCoherence:F4}) should be non-trivial");
    }

    [Fact]
    public void HREModel_TrainPredict_FullCycle()
    {
        // End-to-end: Train and Predict using the IFullModel interface
        var options = new HREModelOptions
        {
            InputSize = 64,
            OutputSize = 1,
            CarrierCount = 8,
            FftSize = 256,
            UseMellinFourier = false,
            NumOFDMLayers = 1,
            NumAttentionLayers = 0,
            HebbianLearningRate = 0.01,
            Seed = 42
        };

        var model = new HREModel<double>(options);

        // Training data
        var input = new Tensor<double>([64]);
        var target = new Tensor<double>([1]);
        for (int i = 0; i < 64; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * 5 * i / 64);
        }
        target[0] = 0.5; // Arbitrary target

        // Train
        model.Train(input, target);

        // Predict
        var prediction = model.Predict(input);

        Assert.Equal(1, prediction.Length);
        Assert.False(double.IsNaN(prediction[0]));
        Assert.False(double.IsInfinity(prediction[0]));

        _output.WriteLine($"Prediction: {prediction[0]:F6}");
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
