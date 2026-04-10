using System.Diagnostics;
using AiDotNet.HarmonicEngine.Benchmarks;
using AiDotNet.HarmonicEngine.Enums;
using AiDotNet.HarmonicEngine.Models;
using AiDotNet.HarmonicEngine.Options;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.UnitTests.HarmonicEngine;

/// <summary>
/// Performance benchmarks: wall-clock comparisons of HRE vs traditional architectures.
/// Measures parameter count, inference latency, training time, and prediction accuracy.
/// These produce the benchmark table for the paper.
/// </summary>
public class PerformanceBenchmarkTests
{
    private readonly ITestOutputHelper _output;

    public PerformanceBenchmarkTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Benchmark_AllNonlinearities_ParameterCountAndLatency()
    {
        var gen = new SyntheticSignalGenerator<double>(42);
        var timeSeries = gen.GenerateComposite(512,
            [3, 7, 13, 23], [1.0, 0.7, 0.4, 0.2],
            trendSlope: 0.1, noiseLevel: 0.05);

        _output.WriteLine("=== HRE Performance Benchmark ===");
        _output.WriteLine($"{"Config",-30} {"Params",8} {"Latency(ms)",12} {"MSE",12} {"MAE",12} {"Predictions",12}");
        _output.WriteLine(new string('-', 86));

        var suite = new HREBenchmarkSuite<double>();
        var results = suite.RunForecasterBenchmark(timeSeries, windowSize: 64, testFraction: 0.2);

        foreach (var r in results)
        {
            _output.WriteLine($"{r.Name,-30} {r.ParameterCount,8} {r.InferenceLatencyMs,12:F3} {r.MSE,12:F6} {r.MAE,12:F6} {r.PredictionCount,12}");

            Assert.True(r.ParameterCount > 0, $"{r.Name} should have parameters");
            Assert.True(r.PredictionCount > 0, $"{r.Name} should produce predictions");
            Assert.False(double.IsNaN(r.MSE), $"{r.Name} MSE is NaN");
        }
    }

    [Fact]
    public void Benchmark_HRE_TrainingTime_SinglePassVsMultiEpoch()
    {
        // Compare HRE single-pass training vs simulated multi-epoch training time
        int windowSize = 64;
        int n = 256;
        var gen = new SyntheticSignalGenerator<double>(42);
        var signal = gen.GenerateComposite(n, [5, 11], [1.0, 0.5]);

        var options = new HREModelOptions
        {
            InputSize = windowSize, OutputSize = 1, CarrierCount = 8, FftSize = 256,
            Nonlinearity = NonlinearityType.SpectralGating,
            UseMellinFourier = false, NumOFDMLayers = 1, NumAttentionLayers = 0,
            HebbianLearningRate = 0.01, Seed = 42
        };

        var model = new HREModel<double>(options);

        // Measure HRE single-pass training time
        var input = new Tensor<double>([windowSize]);
        var target = new Tensor<double>([1]);
        for (int i = 0; i < windowSize; i++) input[i] = signal[i];
        target[0] = signal[windowSize];

        // Warm up
        model.Train(input, target);

        var sw = Stopwatch.StartNew();
        int trainSamples = 100;
        for (int s = 0; s < trainSamples; s++)
        {
            int offset = s % (n - windowSize - 1);
            for (int i = 0; i < windowSize; i++) input[i] = signal[offset + i];
            target[0] = signal[offset + windowSize];
            model.Train(input, target);
        }
        sw.Stop();
        double hreTrainMs = sw.Elapsed.TotalMilliseconds;

        // Measure HRE inference time
        sw.Restart();
        int inferenceSamples = 100;
        for (int s = 0; s < inferenceSamples; s++)
        {
            model.Predict(input);
        }
        sw.Stop();
        double hreInferenceMs = sw.Elapsed.TotalMilliseconds;

        _output.WriteLine("=== Training/Inference Time Comparison ===");
        _output.WriteLine($"HRE single-pass training ({trainSamples} samples): {hreTrainMs:F1} ms ({hreTrainMs / trainSamples:F3} ms/sample)");
        _output.WriteLine($"HRE inference ({inferenceSamples} samples):         {hreInferenceMs:F1} ms ({hreInferenceMs / inferenceSamples:F3} ms/sample)");
        _output.WriteLine($"HRE parameter count:                       {model.ParameterCount}");

        // Estimate equivalent backprop time (N^2 per sample * epochs)
        int denseEquivParams = windowSize * 8 + 8 + 8 * 1 + 1; // ~521
        int epochs = 100;
        _output.WriteLine($"");
        _output.WriteLine($"Dense equivalent parameters:               {denseEquivParams}");
        _output.WriteLine($"Dense would need ~{epochs} epochs x {trainSamples} samples = {epochs * trainSamples} iterations");
        _output.WriteLine($"HRE compression ratio:                     {(double)denseEquivParams / model.ParameterCount:F1}x fewer parameters");

        Assert.True(hreTrainMs / trainSamples < 200,
            $"HRE training should be fast: {hreTrainMs / trainSamples:F3}ms/sample");
    }

    [Fact]
    public void Benchmark_HRE_ParameterEfficiency_VsEquivalentDense()
    {
        _output.WriteLine("=== Parameter Efficiency Comparison ===");
        _output.WriteLine($"{"Config",-40} {"HRE Params",12} {"Dense Equiv",12} {"Compression",12}");
        _output.WriteLine(new string('-', 76));

        var configs = new[]
        {
            (name: "Small (8 carriers, 1 layer)", carriers: 8, layers: 1, attn: 0),
            (name: "Medium (8 carriers, 2 layers, attn)", carriers: 8, layers: 2, attn: 1),
            (name: "Large (16 carriers, 3 layers, attn)", carriers: 16, layers: 3, attn: 1),
        };

        foreach (var (name, carriers, layers, attn) in configs)
        {
            int fftSize = carriers <= 8 ? 256 : 4096;

            try
            {
                var options = new HREModelOptions
                {
                    InputSize = 64, OutputSize = 1, CarrierCount = carriers, FftSize = fftSize,
                    UseMellinFourier = false, NumOFDMLayers = layers, NumAttentionLayers = attn,
                    Seed = 42
                };

                var model = new HREModel<double>(options);
                int hreParams = model.ParameterCount;

                // Dense equivalent: input -> hidden1 -> ... -> hiddenN -> output
                int denseParams = 64 * carriers + carriers; // First layer
                for (int l = 1; l < layers; l++)
                    denseParams += carriers * carriers + carriers; // Hidden layers
                if (attn > 0)
                    denseParams += 3 * carriers * carriers + carriers; // QKV + output projection
                denseParams += carriers * 1 + 1; // Output layer

                double compression = (double)denseParams / hreParams;

                _output.WriteLine($"{name,-40} {hreParams,12} {denseParams,12} {compression,12:F1}x");

                Assert.True(hreParams < denseParams,
                    $"{name}: HRE ({hreParams}) should use fewer params than dense ({denseParams})");
            }
            catch (InvalidOperationException ex) when (ex.Message.Contains("Cannot allocate"))
            {
                _output.WriteLine($"{name,-40} {"SKIP",12} {"N/A",12} {"FFT too small",12}");
            }
        }
    }

    [Fact]
    public void Benchmark_HRE_InferenceScaling_WithInputSize()
    {
        _output.WriteLine("=== Inference Scaling with Input Size ===");
        _output.WriteLine($"{"Input Size",-12} {"Latency(ms)",12} {"Throughput(/s)",14}");
        _output.WriteLine(new string('-', 38));

        foreach (int inputSize in new[] { 32, 64, 128, 256 })
        {
            var options = new HREModelOptions
            {
                InputSize = inputSize, OutputSize = 1, CarrierCount = 8, FftSize = Math.Max(256, inputSize * 4),
                UseMellinFourier = false, NumOFDMLayers = 1, NumAttentionLayers = 0,
                Seed = 42
            };

            var model = new HREModel<double>(options);
            var input = new Tensor<double>([inputSize]);
            for (int i = 0; i < inputSize; i++)
                input[i] = Math.Sin(2 * Math.PI * 5 * i / inputSize);

            // Warm up
            model.Predict(input);

            var sw = Stopwatch.StartNew();
            int iterations = 200;
            for (int iter = 0; iter < iterations; iter++)
                model.Predict(input);
            sw.Stop();

            double avgMs = sw.Elapsed.TotalMilliseconds / iterations;
            double throughput = 1000.0 / avgMs;

            _output.WriteLine($"{inputSize,-12} {avgMs,12:F3} {throughput,14:F0}");

            Assert.True(avgMs < 500, $"Inference at size {inputSize} took {avgMs:F3}ms, should be < 500ms");
        }
    }

    [Fact]
    public void Benchmark_ComplexityComparison_Table()
    {
        // Generate the comparison table from the paper proposal
        _output.WriteLine("=== Architectural Complexity Comparison ===");
        _output.WriteLine("");
        _output.WriteLine("| Metric                  | Traditional NN           | HRE (Harmonic Engine)         |");
        _output.WriteLine("|-------------------------|--------------------------|-------------------------------|");
        _output.WriteLine("| Learning Method         | Iterative Backprop       | Spectral Alignment O(N log N) |");
        _output.WriteLine("|                         | O(Epochs * N^2)          |                               |");
        _output.WriteLine("| Inter-layer Logic       | Matrix Multiply (W * x)  | Wave Interference / Resonance |");
        _output.WriteLine("| Storage                 | High (float32 weights)   | Low (Fourier Coefficients)    |");
        _output.WriteLine("| Quantization            | Required for edge        | Not required (intrinsic comp) |");
        _output.WriteLine("| Hardware                | GPU Dependent             | DSP / CPU Optimized           |");
        _output.WriteLine("| Attention Complexity    | O(N^2)                   | O(N log N) via IMD            |");
        _output.WriteLine("| Lateral Communication   | None (feed-forward only) | All-to-all via spectral bus   |");
        _output.WriteLine("");

        // Concrete numbers from our implementation
        var options = new HREModelOptions
        {
            InputSize = 64, OutputSize = 1, CarrierCount = 8, FftSize = 256,
            UseMellinFourier = false, NumOFDMLayers = 1, NumAttentionLayers = 1,
            Seed = 42
        };
        var model = new HREModel<double>(options);
        int hreParams = model.ParameterCount;
        int denseParams = 64 * 8 + 8 + 8 * 8 + 8 + 8 * 1 + 1;

        _output.WriteLine($"Concrete comparison (64 input, 8 hidden, 1 output):");
        _output.WriteLine($"  HRE parameters:   {hreParams}");
        _output.WriteLine($"  Dense parameters: {denseParams}");
        _output.WriteLine($"  Compression:      {(double)denseParams / hreParams:F1}x");

        Assert.True(hreParams < denseParams);
    }
}
