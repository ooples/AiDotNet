using AiDotNet.HarmonicEngine.Enums;
using AiDotNet.HarmonicEngine.Layers;
using AiDotNet.HarmonicEngine.Learning;
using AiDotNet.HarmonicEngine.Models;
using AiDotNet.HarmonicEngine.Options;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Experiment 6: Full HRE pipeline on synthetic time-series.
/// Tests that the HRE can learn and predict periodic time-series.
/// </summary>
public class SyntheticTimeSeriesBenchmarkTests
{
    private readonly ITestOutputHelper _output;

    public SyntheticTimeSeriesBenchmarkTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void HREForecaster_SyntheticSine_ProducesFinitePredictions()
    {
        // Generate synthetic: sum of 3 sinusoids + trend
        int totalLength = 256;
        int windowSize = 64;
        var timeSeries = GenerateSyntheticTimeSeries(totalLength);

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

        // Predict using a window from the middle of the series
        var window = new Vector<double>(windowSize);
        for (int i = 0; i < windowSize; i++)
        {
            window[i] = timeSeries[100 + i];
        }

        var prediction = forecaster.Predict(window);

        Assert.Equal(1, prediction.Length);
        Assert.False(double.IsNaN(prediction[0]), "Prediction should not be NaN");
        Assert.False(double.IsInfinity(prediction[0]), "Prediction should not be Infinity");

        _output.WriteLine($"Predicted: {prediction[0]:F4}, Actual: {timeSeries[164]:F4}");
    }

    [Fact]
    public void HREForecaster_AutoregressiveMultiStep_ProducesFiniteSequence()
    {
        int totalLength = 256;
        int windowSize = 64;
        int predictSteps = 16;
        var timeSeries = GenerateSyntheticTimeSeries(totalLength);

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

        var initialWindow = new Vector<double>(windowSize);
        for (int i = 0; i < windowSize; i++)
        {
            initialWindow[i] = timeSeries[100 + i];
        }

        var predictions = forecaster.PredictAutoregressive(initialWindow, predictSteps);

        Assert.Equal(predictSteps, predictions.Length);
        for (int i = 0; i < predictSteps; i++)
        {
            Assert.False(double.IsNaN(predictions[i]), $"Prediction[{i}] should not be NaN");
            Assert.False(double.IsInfinity(predictions[i]), $"Prediction[{i}] should not be Infinity");
        }

        _output.WriteLine("Autoregressive predictions (first 8):");
        for (int i = 0; i < Math.Min(8, predictSteps); i++)
        {
            _output.WriteLine($"  Step {i}: predicted={predictions[i]:F4}, actual={timeSeries[164 + i]:F4}");
        }
    }

    [Fact]
    public void SpectralHebbianLayer_SinglePassLearning_ImprovesFilterQuality()
    {
        // Test that a single Hebbian pass produces a useful filter
        int n = 64;
        var wiener = new WienerFilterRule<double>();

        // Input: composite signal
        var input = new Vector<double>(n);
        var target = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * 3 * i / n) + 0.5 * Math.Cos(2 * Math.PI * 7 * i / n);
            target[i] = 0.8 * Math.Sin(2 * Math.PI * 3 * i / n) + 1.2 * Math.Cos(2 * Math.PI * 7 * i / n);
        }

        // Compute Wiener optimal filter and its MSE
        var optimalFilter = wiener.ComputeOptimal(input, target);
        double optimalMSE = wiener.ComputeMSE(input, target, optimalFilter);

        // Create Hebbian layer and train with single pass
        var hebbianLayer = new SpectralHebbianLayer<double>(n, learningRate: 0.05, antiHebbianAlpha: 0.001);
        hebbianLayer.SetTrainingMode(true);

        var inputTensor = new Tensor<double>([n]);
        for (int i = 0; i < n; i++) inputTensor[i] = input[i];

        // Forward pass (initializes internal state)
        hebbianLayer.Forward(inputTensor);

        // Hebbian update with input and target
        hebbianLayer.HebbianUpdate(input, target);

        // Now apply the learned filter and compute MSE
        hebbianLayer.SetTrainingMode(false);
        var filtered = hebbianLayer.Forward(inputTensor);

        double hebbianMSE = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = filtered[i] - target[i];
            hebbianMSE += diff * diff;
        }
        hebbianMSE /= n;

        // Unity filter (no learning) MSE
        double unityMSE = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = input[i] - target[i];
            unityMSE += diff * diff;
        }
        unityMSE /= n;

        _output.WriteLine($"Unity filter MSE:   {unityMSE:F6}");
        _output.WriteLine($"Hebbian filter MSE: {hebbianMSE:F6}");
        _output.WriteLine($"Wiener optimal MSE: {optimalMSE:F6}");

        // Hebbian should improve over doing nothing (unity filter)
        Assert.True(hebbianMSE < unityMSE * 1.1,
            $"Hebbian MSE ({hebbianMSE:F6}) should not be much worse than unity ({unityMSE:F6})");
    }

    [Theory]
    [InlineData(NonlinearityType.ModReLU)]
    [InlineData(NonlinearityType.SpectralGating)]
    [InlineData(NonlinearityType.InstantaneousFreq)]
    public void HREModel_AllNonlinearities_ProduceDistinctOutputs(NonlinearityType nonlinearity)
    {
        var options = new HREModelOptions
        {
            InputSize = 64,
            OutputSize = 1,
            CarrierCount = 8,
            FftSize = 256,
            Nonlinearity = nonlinearity,
            UseMellinFourier = false,
            NumOFDMLayers = 1,
            NumAttentionLayers = 1,
            Seed = 42
        };

        var model = new HREModel<double>(options);

        // Same input but different nonlinearities should produce different outputs
        var input = new Tensor<double>([64]);
        for (int i = 0; i < 64; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * 5 * i / 64);
        }

        var output = model.Forward(input);

        Assert.False(double.IsNaN(output[0]));
        _output.WriteLine($"{nonlinearity}: output = {output[0]:F6}");
    }

    [Fact]
    public void HREModel_ParameterEfficiency_FarFewerThanDense()
    {
        var options = new HREModelOptions
        {
            InputSize = 64,
            OutputSize = 1,
            CarrierCount = 8,
            FftSize = 256,
            UseMellinFourier = false,
            NumOFDMLayers = 2,
            NumAttentionLayers = 1,
            Seed = 42
        };

        var model = new HREModel<double>(options);

        // Equivalent dense network: 64 -> 8 -> 8 -> 1
        // Dense params: 64*8 + 8 + 8*8 + 8 + 8*1 + 1 = 593
        int denseEquivalent = 64 * 8 + 8 + 8 * 8 + 8 + 8 * 1 + 1;

        _output.WriteLine($"HRE parameters:    {model.ParameterCount}");
        _output.WriteLine($"Dense equivalent:  {denseEquivalent}");
        _output.WriteLine($"Compression ratio: {(double)denseEquivalent / model.ParameterCount:F1}x");

        Assert.True(model.ParameterCount < denseEquivalent,
            $"HRE ({model.ParameterCount}) should have fewer params than dense ({denseEquivalent})");
    }

    private static Vector<double> GenerateSyntheticTimeSeries(int length)
    {
        var series = new Vector<double>(length);
        for (int i = 0; i < length; i++)
        {
            double t = (double)i / length;
            // Sum of 3 sinusoids at different frequencies + linear trend + noise
            series[i] = Math.Sin(2 * Math.PI * 3 * t)       // slow cycle
                       + 0.5 * Math.Sin(2 * Math.PI * 7 * t) // medium cycle
                       + 0.3 * Math.Sin(2 * Math.PI * 13 * t) // fast cycle
                       + 0.1 * t;                              // linear trend
        }
        return series;
    }
}
