#if NET5_0_OR_GREATER
using System;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Predict runs its eager forward inside a <see cref="TensorScope"/>, so the forward's GPU intermediates go back to the
/// pool when it returns instead of waiting for the GC (#2383: DenseNet-121 b16 on CUDA held a 9.3 GB device peak of dead
/// intermediates; PyTorch peaks at 120 MB). The scope frees everything created in the forward except the output, so
/// these guard the cases that would break: a model carrying a tensor from one Predict into the next (recurrent
/// state), a host read of the returned output, and repeated calls agreeing with each other and with the CPU engine.
/// </summary>
[Collection(GpuEngineSwapCollection.Name)]
[Trait("Category", "GPU")]
public class PredictReleaseScopeGpuTests
{
    private readonly ITestOutputHelper _output;

    public PredictReleaseScopeGpuTests(ITestOutputHelper output) => _output = output;

    public static TheoryData<string> Models => new() { "rnn", "lstm", "gru", "densenet" };

    [Theory]
    [MemberData(nameof(Models))]
    public void GpuPredict_RepeatedCallsMatchEachOtherAndTheCpuEngine(string model)
    {
        var previous = AiDotNetEngine.Current;
        using var gpu = new DirectGpuTensorEngine();
        if (!gpu.IsGpuAvailable)
        {
            _output.WriteLine("No GPU - nothing to test.");
            return;
        }

        var (net, x, other) = Build(model);
        float[] cpu, first, afterOther, again;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            cpu = net.Predict(x).ToArray();

            AiDotNetEngine.Current = gpu;
            first = net.Predict(x).ToArray();       // host read of a result returned from the scope
            var kept = net.Predict(other);           // held across the next call
            again = net.Predict(x).ToArray();
            afterOther = kept.ToArray();             // still readable after another Predict ended its scope
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }

        // Nothing the model still holds may have been freed by a Predict's scope (LSTMLayer's stacked gate weights were).
        var released = GpuPredictScopeSweepTests.ReleasedReachableTensors(net);
        Assert.True(released.Count == 0, $"{model}: {released.Count} tensors the model holds were released, e.g. {string.Join("; ", released.GetRange(0, Math.Min(3, released.Count)))}");

        Assert.Equal(cpu.Length, first.Length);
        Assert.NotEmpty(afterOther);
        for (int i = 0; i < first.Length; i++)
        {
            Assert.True(first[i] == again[i], $"{model} element {i}: second GPU call {again[i]} vs first {first[i]}");
            float tolerance = 1e-3f * Math.Max(1f, Math.Abs(cpu[i]));
            Assert.True(Math.Abs(first[i] - cpu[i]) <= tolerance, $"{model} element {i}: GPU {first[i]} vs CPU {cpu[i]}");
        }
    }

    // A caller's own scope around Predict: a result kept in Predict's scope would be handed to it and freed when it ends,
    // so what the model holds is detached from every open scope instead.
    [Theory]
    [MemberData(nameof(Models))]
    public void GpuPredict_InsideACallersScope_KeepsWhatTheModelHolds(string model)
    {
        var previous = AiDotNetEngine.Current;
        using var gpu = new DirectGpuTensorEngine();
        if (!gpu.IsGpuAvailable)
        {
            _output.WriteLine("No GPU - nothing to test.");
            return;
        }

        var (net, x, _) = Build(model);
        float[] first, second;
        try
        {
            AiDotNetEngine.Current = gpu;
            using (var callers = TensorScope.Begin())
                first = callers.Keep(net.Predict(x)).ToArray();
            Assert.Empty(GpuPredictScopeSweepTests.ReleasedReachableTensors(net));
            using (TensorScope.Begin())
                second = net.Predict(x).ToArray();
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }

        Assert.Equal(first, second);
    }

    private static (NeuralNetworkBase<float> Net, Tensor<float> X, Tensor<float> Other) Build(string model)
    {
        NeuralNetworkBase<float> net;
        Tensor<float> x, other;
        if (model == "densenet")
        {
            var config = new DenseNetConfiguration(DenseNetVariant.DenseNet121, numClasses: 10, inputHeight: 32, inputWidth: 32, inputChannels: 3);
            var arch = new NeuralNetworkArchitecture<float>(InputType.ThreeDimensional,
                NeuralNetworkTaskType.MultiClassClassification, NetworkComplexity.Simple,
                inputSize: 3 * 32 * 32, inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 10);
            net = new DenseNetNetwork<float>(arch, config);
            x = Random(new[] { 2, 3, 32, 32 }, 1);
            other = Random(new[] { 2, 3, 32, 32 }, 2);
        }
        else
        {
            var arch = new NeuralNetworkArchitecture<float>(InputType.TwoDimensional,
                NeuralNetworkTaskType.MultiClassClassification, NetworkComplexity.Simple,
                inputHeight: 10, inputWidth: 16, outputSize: 5);
            net = model switch
            {
                "rnn" => new RecurrentNeuralNetwork<float>(arch),
                "lstm" => new LSTMNeuralNetwork<float>(arch, lossFunction: null, outputActivation: null),
                _ => new GRUNeuralNetwork<float>(arch),
            };
            x = Random(new[] { 10, 16 }, 1);
            other = Random(new[] { 10, 16 }, 2);
        }
        net.SetTrainingMode(false);
        return (net, x, other);
    }

    private static Tensor<float> Random(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}
#endif
