using System;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.Training;

/// <summary>
/// A custom-objective fused step runs inside the arena its caller opened around the attempt, so the session's own
/// per-step arena is nested. A nested arena that does not pool drops every buffer it rented on Dispose, including the
/// ones it took from the cross-arena pool, so every step allocated its large activations again (a DCGAN CPU step: 16 MB
/// of the step's own and 69 MB the generator's next detached forward lost to it). The step's transients are dead at its
/// return whether or not an outer arena exists, so the step arena pools either way.
/// </summary>
[Collection("FusedTrainingSerial")]
public class NestedFusedStepArenaTests
{
    private readonly ITestOutputHelper _output;

    public NestedFusedStepArenaTests(ITestOutputHelper output) => _output = output;

    private const int Batch = 1024;
    private const int Width = 512;   // [1024, 512] activations: 2 MB each, far above the arena's persist threshold

    private sealed class WideNetwork : FeedForwardNeuralNetwork<float>
    {
        internal WideNetwork() : base(
            new NeuralNetworkArchitecture<float>(inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                complexity: NetworkComplexity.Simple, inputSize: Width, outputSize: Width)
            {
                RandomSeed = 42
            },
            new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-3 }),
            new MeanSquaredErrorLoss<float>())
        {
        }

        internal float Step(Tensor<float> x, Tensor<float> y)
            => TrainWithCustomLoss(x, prediction => new MeanSquaredErrorLoss<float>().ComputeTapeLoss(prediction, y));
    }

    [Fact]
    public void ASteadyFusedCustomLossStep_ReusesItsLargeActivations()
    {
        var previousEngine = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            var random = new Random(3);
            var x = new Tensor<float>(new[] { Batch, Width });
            var y = new Tensor<float>(new[] { Batch, Width });
            var xs = x.AsWritableSpan();
            var ys = y.AsWritableSpan();
            for (int i = 0; i < xs.Length; i++) { xs[i] = (float)(random.NextDouble() - 0.5); ys[i] = (float)(random.NextDouble() - 0.5); }

            var network = new WideNetwork();
            for (int i = 0; i < 4; i++) network.Step(x, y);   // trace, commit, and the pool's first returns
            Assert.True(network.CustomFusedStepCount > 0, "the custom loss did not reach the fused session");

#if NET5_0_OR_GREATER
            const int measured = 4;
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < measured; i++) network.Step(x, y);
            long perStep = (GC.GetAllocatedBytesForCurrentThread() - before) / measured;
            long activationBytes = (long)Batch * Width * sizeof(float);
            _output.WriteLine($"steady step: {perStep} bytes allocated on the stepping thread");
            // Without pooling a step allocates at least one activation afresh; pooled, only small bookkeeping.
            Assert.True(perStep < activationBytes / 2,
                $"a steady step allocated {perStep} bytes on its thread, over half a [{Batch}, {Width}] activation "
                + $"({activationBytes} bytes): the step arena is not handing its buffers back.");
#endif
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
        }
    }
}
