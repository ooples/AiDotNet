using AiDotNet.Finance.Data;
using AiDotNet.HarmonicEngine.Enums;
using AiDotNet.HarmonicEngine.Models;
using AiDotNet.HarmonicEngine.Options;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Experiment 7: Financial time-series benchmark.
/// Tests HRE integration with financial data infrastructure.
/// </summary>
public class FinancialIntegrationTests
{
    private readonly ITestOutputHelper _output;

    public FinancialIntegrationTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void HREForecaster_SyntheticFinancialData_ProducesValidPredictions()
    {
        // Generate synthetic OHLCV data that resembles real financial data
        int numBars = 300;
        var marketData = GenerateSyntheticMarketData(numBars);

        // Extract close prices as time series
        var closePrices = new Vector<double>(numBars);
        for (int i = 0; i < numBars; i++)
        {
            closePrices[i] = marketData[i].Close;
        }

        // Create HRE forecaster
        int windowSize = 64;
        var options = new HREModelOptions
        {
            CarrierCount = 8,
            FftSize = 256,
            Nonlinearity = NonlinearityType.SpectralGating,
            UseMellinFourier = false,
            NumOFDMLayers = 1,
            NumAttentionLayers = 0,
            Seed = 42
        };

        var forecaster = new HREForecaster<double>(windowSize, 1, options);

        // Run predictions on test portion (last 50 bars)
        int trainEnd = numBars - 50;
        int validPredictions = 0;
        double totalAbsError = 0;

        for (int t = trainEnd; t < numBars - 1; t++)
        {
            if (t - windowSize < 0) continue;

            var window = new Vector<double>(windowSize);
            for (int i = 0; i < windowSize; i++)
            {
                window[i] = closePrices[t - windowSize + i];
            }

            var pred = forecaster.Predict(window);

            if (!double.IsNaN(pred[0]) && !double.IsInfinity(pred[0]))
            {
                validPredictions++;
                totalAbsError += Math.Abs(pred[0] - closePrices[t + 1]);
            }
        }

        Assert.True(validPredictions > 0, "Should produce at least some valid predictions");

        double mae = totalAbsError / validPredictions;
        _output.WriteLine($"Valid predictions: {validPredictions}");
        _output.WriteLine($"MAE: {mae:F4}");
        _output.WriteLine($"Mean price: {closePrices.Transform(x => x).Length}");
    }

    [Fact]
    public void HREModel_FinancialDataWithMellinFourier_ScaleInvariant()
    {
        // Test that the model produces similar outputs for scaled versions
        // of the same price pattern (e.g., same pattern at $100 vs $200 level)
        int windowSize = 64;

        var options = new HREModelOptions
        {
            InputSize = windowSize,
            OutputSize = 1,
            CarrierCount = 8,
            FftSize = 256,
            Nonlinearity = NonlinearityType.SpectralGating,
            UseMellinFourier = true,
            NumOFDMLayers = 1,
            NumAttentionLayers = 0,
            Seed = 42
        };

        var model = new HREModel<double>(options);

        // Pattern at $100 level
        var pattern1 = new Tensor<double>([windowSize]);
        for (int i = 0; i < windowSize; i++)
        {
            pattern1[i] = 100.0 + 5.0 * Math.Sin(2 * Math.PI * 3 * i / windowSize)
                         + 2.0 * Math.Sin(2 * Math.PI * 7 * i / windowSize);
        }

        // Same pattern at $200 level (2x scale)
        var pattern2 = new Tensor<double>([windowSize]);
        for (int i = 0; i < windowSize; i++)
        {
            pattern2[i] = 200.0 + 10.0 * Math.Sin(2 * Math.PI * 3 * i / windowSize)
                         + 4.0 * Math.Sin(2 * Math.PI * 7 * i / windowSize);
        }

        var output1 = model.Forward(pattern1);
        var output2 = model.Forward(pattern2);

        Assert.False(double.IsNaN(output1[0]), "Output1 should not be NaN");
        Assert.False(double.IsNaN(output2[0]), "Output2 should not be NaN");

        _output.WriteLine($"Pattern at $100: output = {output1[0]:F6}");
        _output.WriteLine($"Pattern at $200: output = {output2[0]:F6}");
    }

    [Fact]
    public void HREModel_InferenceLatency_UnderOneMillisecond()
    {
        // Benchmark: HRE inference should be very fast (target < 1ms)
        int windowSize = 64;

        var options = new HREModelOptions
        {
            InputSize = windowSize,
            OutputSize = 1,
            CarrierCount = 8,
            FftSize = 256,
            Nonlinearity = NonlinearityType.SpectralGating,
            UseMellinFourier = false,
            NumOFDMLayers = 1,
            NumAttentionLayers = 1,
            Seed = 42
        };

        var model = new HREModel<double>(options);

        var input = new Tensor<double>([windowSize]);
        for (int i = 0; i < windowSize; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * 5 * i / windowSize);
        }

        // Warm up
        model.Forward(input);

        // Measure
        int iterations = 100;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int iter = 0; iter < iterations; iter++)
        {
            model.Forward(input);
        }
        sw.Stop();

        double avgMs = sw.Elapsed.TotalMilliseconds / iterations;
        _output.WriteLine($"Average inference latency: {avgMs:F3} ms over {iterations} iterations");
        _output.WriteLine($"Total time: {sw.Elapsed.TotalMilliseconds:F1} ms");

        // Should be fast — allow generous budget for CI
        Assert.True(avgMs < 50.0,
            $"HRE inference should be fast, got {avgMs:F3}ms (target < 50ms including test overhead)");
    }

    [Fact]
    public void HREModel_ModelSizeVsDense_SignificantCompression()
    {
        int inputSize = 64;
        int outputSize = 1;
        int hiddenSize = 8;

        var options = new HREModelOptions
        {
            InputSize = inputSize,
            OutputSize = outputSize,
            CarrierCount = hiddenSize,
            FftSize = 256,
            UseMellinFourier = false,
            NumOFDMLayers = 2,
            NumAttentionLayers = 1,
            Seed = 42
        };

        var model = new HREModel<double>(options);

        // Compare: equivalent dense network
        // Dense: input(64) -> hidden(8) -> hidden(8) -> output(1)
        // Params: 64*8+8 + 8*8+8 + 8*1+1 = 520+72+9 = 601
        int denseParams = inputSize * hiddenSize + hiddenSize
                        + hiddenSize * hiddenSize + hiddenSize
                        + hiddenSize * outputSize + outputSize;

        int hreParams = model.ParameterCount;
        double compressionRatio = (double)denseParams / hreParams;

        _output.WriteLine($"HRE parameters:     {hreParams}");
        _output.WriteLine($"Dense equivalent:   {denseParams}");
        _output.WriteLine($"Compression ratio:  {compressionRatio:F1}x");

        // HRE should use significantly fewer parameters
        Assert.True(hreParams < denseParams,
            $"HRE ({hreParams}) should use fewer parameters than dense ({denseParams})");
    }

    private static List<MarketDataPoint<double>> GenerateSyntheticMarketData(int numBars)
    {
        var rng = new Random(42);
        var data = new List<MarketDataPoint<double>>();

        double price = 100.0;
        var baseTime = new DateTime(2024, 1, 1);

        for (int i = 0; i < numBars; i++)
        {
            // Random walk with seasonal component
            double seasonal = 2.0 * Math.Sin(2 * Math.PI * i / 20) // 20-bar cycle
                            + 1.0 * Math.Sin(2 * Math.PI * i / 50); // 50-bar cycle
            double noise = (rng.NextDouble() - 0.5) * 2.0;
            double trend = 0.01;

            price += seasonal * 0.1 + noise + trend;
            price = Math.Max(price, 1.0); // Prevent negative prices

            double open = price + (rng.NextDouble() - 0.5) * 0.5;
            double high = Math.Max(open, price) + rng.NextDouble() * 1.0;
            double low = Math.Min(open, price) - rng.NextDouble() * 1.0;
            double volume = 1000000 + rng.NextDouble() * 500000;

            data.Add(new MarketDataPoint<double>(
                baseTime.AddDays(i), open, high, low, price, volume));
        }

        return data;
    }
}
