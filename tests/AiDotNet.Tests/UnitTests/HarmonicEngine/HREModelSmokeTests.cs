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
    public void HREModel_ForwardPass_ProducesNonTrivialOutput(NonlinearityType nonlinearity)
    {
        // Arrange
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
        var input = new Tensor<double>([64]);
        for (int i = 0; i < 64; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * 3 * i / 64);
        }

        // Act
        var output = model.Forward(input);

        // Assert: valid + non-trivial magnitude + different input produces different output
        Assert.Equal(1, output.Length);
        Assert.False(double.IsNaN(output[0]), "Output should not be NaN");
        Assert.False(double.IsInfinity(output[0]), "Output should not be Infinity");
        Assert.True(Math.Abs(output[0]) > 1e-8,
            $"Output magnitude {Math.Abs(output[0]):E4} should be non-trivial — a model that " +
            $"always outputs 0 would pass a pure !IsNaN check.");

        // A different input shape should produce a different output —
        // confirms the model is actually using its input.
        var input2 = new Tensor<double>([64]);
        for (int i = 0; i < 64; i++)
        {
            input2[i] = Math.Cos(2 * Math.PI * 7 * i / 64);
        }
        var output2 = model.Forward(input2);
        Assert.True(Math.Abs(output[0] - output2[0]) > 1e-8,
            $"Different inputs should produce different outputs, got {output[0]:F6} vs {output2[0]:F6}.");
    }

    [Fact]
    public void HREModel_WithMellinFourier_FingerprintIsScaleInvariant()
    {
        // With UseMellinFourier=true, the first stage of the model is a
        // scale-invariant fingerprint. We validate this by running two
        // signals that differ only by amplitude scaling through the
        // MellinFourier layer directly and comparing the fingerprints.
        // (The full HREModel's random output projection breaks the
        // invariance at the output level, so we test the layer directly.)
        int windowSize = 64;
        var mellin = new AiDotNet.HarmonicEngine.Transforms.MellinTransform<double>();

        var signal = new Vector<double>(windowSize);
        for (int i = 0; i < windowSize; i++)
        {
            signal[i] = Math.Cos(2 * Math.PI * 5 * i / windowSize)
                      + 0.3 * Math.Sin(2 * Math.PI * 11 * i / windowSize);
        }

        // Scale by 3×
        var scaled = new Vector<double>(windowSize);
        for (int i = 0; i < windowSize; i++) scaled[i] = 3.0 * signal[i];

        var fp1 = mellin.ScaleInvariantFingerprint(signal);
        var fp2 = mellin.ScaleInvariantFingerprint(scaled);

        // Cosine similarity should be ~1.0 for pure amplitude scaling
        double dot = 0, norm1 = 0, norm2 = 0;
        for (int i = 0; i < windowSize; i++)
        {
            dot += fp1[i] * fp2[i];
            norm1 += fp1[i] * fp1[i];
            norm2 += fp2[i] * fp2[i];
        }
        double cosSim = dot / (Math.Sqrt(norm1) * Math.Sqrt(norm2) + 1e-15);

        Assert.True(cosSim > 0.999,
            $"Mellin fingerprint should be scale-invariant under 3× amplitude scaling. " +
            $"Got cosine similarity {cosSim:F6}, expected > 0.999.");

        // Also verify the full HREModel with MellinFourier runs cleanly and
        // produces the expected output shape (this is the "smoke test" part)
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
        for (int i = 0; i < 64; i++) input[i] = signal[i];
        var output = model.Forward(input);

        Assert.Equal(2, output.Length);
        for (int i = 0; i < 2; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Output[{i}] should not be NaN");
            Assert.False(double.IsInfinity(output[i]), $"Output[{i}] should not be Infinity");
        }
    }

    [Fact]
    public void HREForecaster_PredictSyntheticSine_ProducesBoundedPrediction()
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

        // Sine wave values are in [-1, 1]. An untrained model shouldn't
        // produce NaN, Infinity, OR wildly out-of-range values — if the
        // prediction is 10⁶ for a unit-amplitude sine input, something's wrong.
        var window = new Vector<double>(64);
        for (int i = 0; i < 64; i++)
        {
            window[i] = Math.Sin(2 * Math.PI * i / 32);
        }

        var prediction = forecaster.Predict(window);

        Assert.Equal(1, prediction.Length);
        Assert.False(double.IsNaN(prediction[0]), "Prediction should not be NaN");
        Assert.False(double.IsInfinity(prediction[0]), "Prediction should not be Infinity");
        Assert.True(Math.Abs(prediction[0]) < 100.0,
            $"Prediction magnitude {Math.Abs(prediction[0]):F4} should be bounded — " +
            $"a unit-amplitude sine input should not produce predictions >100.");
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
    public void HREModel_ParameterCount_AtLeast20xCompressionVsDense()
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

        // Equivalent dense MLP: 64 inputs → 16 hidden → 16 hidden → 1 output
        // with biases: 64×16+16 + 16×16+16 + 16×1+1 = 1040+272+17 = 1329
        int denseParams = 64 * 16 + 16 + 16 * 16 + 16 + 16 * 1 + 1;
        int hreParams = model.ParameterCount;
        double compressionRatio = (double)denseParams / hreParams;

        // Assert at least 20× compression — this is the real claim of HRE's
        // parameter efficiency, not an arbitrary `< 100` threshold.
        Assert.True(compressionRatio >= 20.0,
            $"HRE compression ratio should be >= 20×, got {compressionRatio:F1}× " +
            $"(HRE params: {hreParams}, dense equivalent: {denseParams}).");
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
    public void SpectralHebbianLayer_SerializationRoundTrip_PreservesTrainedFilter()
    {
        int signalLength = 64;
        var layer = new AiDotNet.HarmonicEngine.Layers.SpectralHebbianLayer<double>(
            signalLength, learningRate: 0.1, antiHebbianAlpha: 0.5);

        // Capture the untrained initial output (identity filter baseline)
        var input = new Tensor<double>([signalLength]);
        for (int i = 0; i < signalLength; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * 5 * i / signalLength);
        }
        var untrainedOutput = layer.Forward(input);

        // Train the filter so it actually gets modified
        layer.SetTrainingMode(true);
        var inputSignal = new Vector<double>(signalLength);
        var target = new Vector<double>(signalLength);
        for (int i = 0; i < signalLength; i++)
        {
            inputSignal[i] = input[i];
            target[i] = 2.0 * Math.Sin(2 * Math.PI * 5 * i / signalLength);
        }
        for (int iter = 0; iter < 100; iter++)
        {
            layer.HebbianUpdate(inputSignal, target);
        }

        // Get output after training — must differ from untrained
        layer.SetTrainingMode(false);
        var trainedOutput = layer.Forward(input);

        double trainingDelta = 0;
        for (int i = 0; i < signalLength; i++)
            trainingDelta += Math.Abs(trainedOutput[i] - untrainedOutput[i]);
        Assert.True(trainingDelta > 1e-6,
            $"Training should modify the filter (delta={trainingDelta:E4}) — if the filter " +
            $"is unchanged, the round-trip test is meaningless.");

        // Serialize
        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            layer.Serialize(writer);
        }

        // Deserialize into a new layer
        ms.Position = 0;
        var layer2 = new AiDotNet.HarmonicEngine.Layers.SpectralHebbianLayer<double>(
            signalLength, learningRate: 0.1, antiHebbianAlpha: 0.5);
        using (var reader = new BinaryReader(ms))
        {
            layer2.Deserialize(reader);
        }

        // Restored output must match trained output exactly (round-trip preserves filter)
        layer2.SetTrainingMode(false);
        var restoredOutput = layer2.Forward(input);

        for (int i = 0; i < signalLength; i++)
        {
            Assert.Equal(trainedOutput[i], restoredOutput[i], 6);
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
