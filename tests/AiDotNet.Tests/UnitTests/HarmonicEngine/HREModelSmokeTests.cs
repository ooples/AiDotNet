using AiDotNet.HarmonicEngine.Core;
using AiDotNet.HarmonicEngine.Enums;
using AiDotNet.HarmonicEngine.Models;
using AiDotNet.HarmonicEngine.Options;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Integration smoke tests for the full HRE pipeline.
/// Verifies that all components compose correctly and produce valid output.
/// </summary>
public class HREModelSmokeTests
{
    [Theory]
    [InlineData(NonlinearityType.ModReLU)]
    [InlineData(NonlinearityType.SpectralGating)]
    [InlineData(NonlinearityType.InstantaneousFreq)]
    public void HREModel_ForwardPass_ProducesValidOutput(NonlinearityType nonlinearity)
    {
        // Arrange
        var options = new HREModelOptions
        {
            InputSize = 64,
            OutputSize = 1,
            CarrierCount = 8,
            FftSize = 256,
            Nonlinearity = nonlinearity,
            UseMellinFourier = false, // Skip for speed
            NumOFDMLayers = 1,
            NumAttentionLayers = 1,
            Seed = 42
        };

        var model = new HREModel<double>(options);
        var input = new Tensor<double>([64]);
        for (int i = 0; i < 64; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * 3 * i / 64);
        }

        // Act
        var output = model.Forward(input);

        // Assert
        Assert.Equal(1, output.Length);
        Assert.False(double.IsNaN(output[0]), "Output should not be NaN");
        Assert.False(double.IsInfinity(output[0]), "Output should not be Infinity");
    }

    [Fact]
    public void HREModel_WithMellinFourier_ProducesValidOutput()
    {
        var options = new HREModelOptions
        {
            InputSize = 64,
            OutputSize = 2,
            CarrierCount = 8,
            FftSize = 256,
            UseMellinFourier = true,
            NumOFDMLayers = 1,
            NumAttentionLayers = 0,
            Seed = 42
        };

        var model = new HREModel<double>(options);
        var input = new Tensor<double>([64]);
        for (int i = 0; i < 64; i++)
        {
            input[i] = Math.Cos(2 * Math.PI * 5 * i / 64);
        }

        var output = model.Forward(input);

        Assert.Equal(2, output.Length);
        for (int i = 0; i < 2; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Output[{i}] should not be NaN");
            Assert.False(double.IsInfinity(output[i]), $"Output[{i}] should not be Infinity");
        }
    }

    [Fact]
    public void HREForecaster_PredictSyntheticSine_ProducesValidOutput()
    {
        var options = new HREModelOptions
        {
            CarrierCount = 8,
            FftSize = 256,
            UseMellinFourier = false,
            NumOFDMLayers = 1,
            NumAttentionLayers = 0,
            Seed = 42
        };

        var forecaster = new HREForecaster<double>(
            windowSize: 64, predictionHorizon: 1, options: options);

        var window = new Vector<double>(64);
        for (int i = 0; i < 64; i++)
        {
            window[i] = Math.Sin(2 * Math.PI * i / 32);
        }

        var prediction = forecaster.Predict(window);

        Assert.Equal(1, prediction.Length);
        Assert.False(double.IsNaN(prediction[0]), "Prediction should not be NaN");
    }

    [Fact]
    public void HREModel_GetSummary_ReturnsNonEmptyString()
    {
        var options = new HREModelOptions
        {
            InputSize = 64,
            OutputSize = 1,
            CarrierCount = 8,
            FftSize = 256,
            Seed = 42
        };

        var model = new HREModel<double>(options);
        var summary = model.GetSummary();

        Assert.False(string.IsNullOrWhiteSpace(summary));
        Assert.Contains("Harmonic Resonance Engine", summary);
        Assert.Contains("Carriers: 8", summary);
    }

    [Fact]
    public void HREModel_ParameterCount_IsReasonable()
    {
        var options = new HREModelOptions
        {
            InputSize = 64,
            OutputSize = 1,
            CarrierCount = 8,
            FftSize = 1024,
            UseMellinFourier = false,
            NumOFDMLayers = 2,
            NumAttentionLayers = 1,
            Seed = 42
        };

        var model = new HREModel<double>(options);

        // HRE should have very few parameters compared to equivalent dense network
        // Dense network: 64 * 16 + 16 * 16 + 16 * 1 = 1296
        // HRE: mostly from output projection = 16 * 1 + 1 = 17
        Assert.True(model.ParameterCount < 100,
            $"HRE should have very few parameters, got {model.ParameterCount}");
    }

    [Fact]
    public void SpectralSparsityMask_SelectK_ReturnsReasonableValue()
    {
        var mask = new SpectralSparsityMask<double>();
        int n = 64;

        // Create a 3-sparse spectrum
        var spectrum = new Vector<Complex<double>>(n);
        var zero = new Complex<double>(0, 0);
        for (int i = 0; i < n; i++) spectrum[i] = zero;

        spectrum[5] = new Complex<double>(10.0, 0);
        spectrum[12] = new Complex<double>(8.0, 0);
        spectrum[23] = new Complex<double>(6.0, 0);
        // Add some noise
        for (int i = 0; i < n; i++)
        {
            if (i != 5 && i != 12 && i != 23)
            {
                spectrum[i] = new Complex<double>(0.01 * (i % 3), 0);
            }
        }

        int optimalK = mask.SelectK(spectrum);

        // Should select a small K close to the true sparsity (3)
        Assert.True(optimalK >= 1, "K should be at least 1");
        Assert.True(optimalK <= 10, $"K should be small for a 3-sparse signal, got {optimalK}");
    }

    [Fact]
    public void SpectralSparsityMask_EnergyRatio_HighForDominantComponents()
    {
        var mask = new SpectralSparsityMask<double>();
        int n = 64;

        var spectrum = new Vector<Complex<double>>(n);
        var zero = new Complex<double>(0, 0);
        for (int i = 0; i < n; i++) spectrum[i] = zero;

        // 3 dominant components
        spectrum[5] = new Complex<double>(10.0, 0);
        spectrum[12] = new Complex<double>(8.0, 0);
        spectrum[23] = new Complex<double>(6.0, 0);

        double ratio = mask.EnergyRatio(spectrum, 3);

        // Should capture ~100% of energy since the signal is exactly 3-sparse
        Assert.True(ratio > 0.99, $"Energy ratio for exact sparsity should be ~1.0, got {ratio}");
    }

    [Fact]
    public void SpectralHebbianLayer_SerializationRoundTrip_PreservesFilter()
    {
        int signalLength = 64;
        var layer = new AiDotNet.HarmonicEngine.Layers.SpectralHebbianLayer<double>(
            signalLength, learningRate: 0.01, antiHebbianAlpha: 0.1);

        // Modify the filter by running a Hebbian update
        var input = new Tensor<double>([signalLength]);
        for (int i = 0; i < signalLength; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * 5 * i / signalLength);
        }
        layer.SetTrainingMode(true);
        layer.Forward(input);

        var inputSignal = new Vector<double>(signalLength);
        var target = new Vector<double>(signalLength);
        for (int i = 0; i < signalLength; i++)
        {
            inputSignal[i] = input[i];
            target[i] = 2.0 * Math.Sin(2 * Math.PI * 5 * i / signalLength);
        }
        layer.HebbianUpdate(inputSignal, target);

        // Get output before serialization
        layer.SetTrainingMode(false);
        var outputBefore = layer.Forward(input);

        // Serialize
        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            layer.Serialize(writer);
        }

        // Deserialize into a new layer
        ms.Position = 0;
        var layer2 = new AiDotNet.HarmonicEngine.Layers.SpectralHebbianLayer<double>(
            signalLength, learningRate: 0.01, antiHebbianAlpha: 0.1);
        using (var reader = new BinaryReader(ms))
        {
            layer2.Deserialize(reader);
        }

        // Get output after deserialization
        layer2.SetTrainingMode(false);
        var outputAfter = layer2.Forward(input);

        // Outputs should match
        for (int i = 0; i < signalLength; i++)
        {
            Assert.Equal(outputBefore[i], outputAfter[i], 6);
        }
    }

    [Fact]
    public void HREBenchmarkSuite_RunForecasterBenchmark_ProducesResults()
    {
        var suite = new AiDotNet.HarmonicEngine.Benchmarks.HREBenchmarkSuite<double>();
        var gen = new AiDotNet.HarmonicEngine.Benchmarks.SyntheticSignalGenerator<double>(42);

        var timeSeries = gen.GenerateComposite(256, [3, 7, 13], [1.0, 0.5, 0.3], trendSlope: 0.1);

        var results = suite.RunForecasterBenchmark(timeSeries, windowSize: 64, testFraction: 0.2);

        Assert.Equal(3, results.Count); // One per nonlinearity type
        foreach (var result in results)
        {
            Assert.False(string.IsNullOrEmpty(result.Name));
            Assert.True(result.ParameterCount > 0);
            Assert.True(result.PredictionCount > 0, $"{result.Name} should make predictions");
            Assert.False(double.IsNaN(result.MSE), $"{result.Name} MSE should not be NaN");
        }
    }
}
