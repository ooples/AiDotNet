#if NET5_0_OR_GREATER
using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// #1824 follow-up: a GPU-resident forward rents no host scratch from the inference arena, so per-layer recycling
/// has nothing to recycle there. It used to detach every layer boundary anyway, which downloaded it to a fresh host
/// array for the next layer to upload again (a 4-layer 512-wide Transformer on DirectGpuTensorEngine: 38-39 ms and
/// 50.9 MB per Predict against 30-33 ms and 23.6 MB with recycling off). With the arena empty the walk now skips
/// the detach, so recycling on and off allocate the same and give the same bits.
/// </summary>
[Collection("InferenceArena")]
[Trait("Category", "GPU")]
public class InferenceArenaGpuRecycleTests
{
    private readonly ITestOutputHelper _output;

    public InferenceArenaGpuRecycleTests(ITestOutputHelper output) => _output = output;

    private static Transformer<float> Build()
    {
        var arch = new TransformerArchitecture<float>(inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification, numEncoderLayers: 4, numDecoderLayers: 0,
            numHeads: 8, modelDimension: 512, feedForwardDimension: 2048, inputSize: 512, outputSize: 10,
            maxSequenceLength: 256, vocabularySize: 1000);
        var net = new Transformer<float>(arch);
        net.SetTrainingMode(false);
        return net;
    }

    // One model for both settings, so only the recycling differs.
    private static (float[] Output, long BytesPerCall) Run(Transformer<float> net, bool recycle)
    {
        bool prevEnabled = InferenceArenaSettings.Enabled, prevRecycle = InferenceArenaSettings.RecycleLayerScratch;
        InferenceArenaSettings.Enabled = true;
        InferenceArenaSettings.RecycleLayerScratch = recycle;
        try
        {
            var rng = new Random(1);
            var x = new Tensor<float>(new[] { 4, 256 });
            for (int i = 0; i < x.Length; i++) x[i] = rng.Next(1000);
            for (int i = 0; i < 3; i++) net.Predict(x);
            const int calls = 5;
            long before = GC.GetTotalAllocatedBytes(true);
            float[] output = Array.Empty<float>();
            for (int i = 0; i < calls; i++) output = net.Predict(x).ToArray();
            return (output, (GC.GetTotalAllocatedBytes(true) - before) / calls);
        }
        finally
        {
            InferenceArenaSettings.Enabled = prevEnabled;
            InferenceArenaSettings.RecycleLayerScratch = prevRecycle;
        }
    }

    [Fact]
    public void GpuResidentForward_RecyclingAllocatesNoMoreThanWithout_AndGivesTheSameBits()
    {
        var previous = AiDotNetEngine.Current;
        using var gpu = new DirectGpuTensorEngine();
        if (!gpu.IsGpuAvailable)
        {
            _output.WriteLine("No GPU - nothing to test.");
            return;
        }

        (float[] Output, long BytesPerCall) off, on;
        AiDotNetEngine.Current = gpu;
        try
        {
            var net = Build();
            off = Run(net, recycle: false);
            on = Run(net, recycle: true);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
        _output.WriteLine($"bytes per Predict: recycle off {off.BytesPerCall:N0}, on {on.BytesPerCall:N0}");

        Assert.Equal(off.Output.Length, on.Output.Length);
        for (int i = 0; i < off.Output.Length; i++)
            Assert.True(off.Output[i] == on.Output[i], $"element {i}: {on.Output[i]} with recycling vs {off.Output[i]} without");
        Assert.True(on.BytesPerCall <= off.BytesPerCall * 1.2 + 1_000_000,
            $"recycling allocated {on.BytesPerCall:N0} bytes per Predict against {off.BytesPerCall:N0} without");
    }
}
#endif
