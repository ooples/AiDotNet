using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Allocation + GC-churn profile of CNN (and MLP) inference Predict — exact bytes
/// allocated per call (GC.GetAllocatedBytesForCurrentThread) and Gen0/1/2 collection
/// counts over a run. Env-gated (AIDOTNET_RUN_JIT_PERF=1). This is the
/// allocation/GC half of the profiling; the CPU call tree comes from dotnet-trace.
/// </summary>
public class CnnInferenceAllocProfile
{
    private readonly ITestOutputHelper _output;
    public CnnInferenceAllocProfile(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Profile_CnnAndMlp_AllocAndGc()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;

        ProfileCnn();
        ProfileMlp();
    }

    private void ProfileCnn()
    {
        var layers = new System.Collections.Generic.List<ILayer<float>>
        {
            new ConvolutionalLayer<float>(16, 3, 1, 1, activationFunction: new ReLUActivation<float>()),
            new MaxPoolingLayer<float>(2, 2),
            new ConvolutionalLayer<float>(32, 3, 1, 1, activationFunction: new ReLUActivation<float>()),
            new MaxPoolingLayer<float>(2, 2),
            new FlattenLayer<float>(),
            new DenseLayer<float>(10, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification, inputHeight: 28, inputWidth: 28, inputDepth: 1,
            outputSize: 10, layers: layers);
        var model = new ConvolutionalNeuralNetwork<float>(arch);
        var x = Rand(new[] { 1, 1, 28, 28 }, 1);
        Run("CNN", () => model.Predict(x));
    }

    private void ProfileMlp()
    {
        var layers = new System.Collections.Generic.List<ILayer<float>>
        {
            new DenseLayer<float>(512, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(128, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(10, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification, inputSize: 784, outputSize: 10, layers: layers);
        var model = new FeedForwardNeuralNetwork<float>(arch);
        var x = Rand(new[] { 1, 784 }, 1);
        Run("MLP", () => model.Predict(x));
    }

    private void Run(string name, Func<Tensor<float>> predict)
    {
        for (int i = 0; i < 50; i++) { var _ = predict(); }   // warm (materialize weights, JIT, prepack)

        const int reps = 2000;
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2);
        // GetAllocatedBytesForCurrentThread is net5+; on net471 fall back to
        // GetTotalMemory (less precise but always available) so this profiler
        // compiles on every target framework.
#if NET5_0_OR_GREATER
        long allocBefore = GC.GetAllocatedBytesForCurrentThread();
#else
        long allocBefore = GC.GetTotalMemory(forceFullCollection: false);
#endif
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < reps; i++) { var _ = predict(); }
        sw.Stop();
#if NET5_0_OR_GREATER
        long allocAfter = GC.GetAllocatedBytesForCurrentThread();
#else
        long allocAfter = GC.GetTotalMemory(forceFullCollection: false);
#endif
        int d0 = GC.CollectionCount(0) - g0, d1 = GC.CollectionCount(1) - g1, d2 = GC.CollectionCount(2) - g2;

        double usPer = sw.Elapsed.TotalMilliseconds * 1000.0 / reps;
        long bytesPer = (allocAfter - allocBefore) / reps;
        _output.WriteLine($"[{name}] {usPer:F1} us/Predict   {bytesPer:N0} bytes/Predict   " +
            $"GC over {reps} calls: gen0={d0} gen1={d1} gen2={d2}   total alloc {(allocAfter - allocBefore) / 1024.0 / 1024.0:F1} MB");
    }

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed); var t = new Tensor<float>(shape); var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }
}
