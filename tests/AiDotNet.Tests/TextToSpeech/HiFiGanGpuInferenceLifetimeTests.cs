using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class HiFiGanGpuInferenceLifetimeTests
{
    [SkippableFact]
    public void PredictionReleasesScratchAndKeepsWeightsAndReturnedWaveformsAlive()
    {
        var previousEngine = AiDotNetEngine.Current;
        DirectGpuTensorEngine? gpu = null;
        try
        {
            try { gpu = new DirectGpuTensorEngine(); }
            catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
            {
                Skip.If(true, "CUDA unavailable: " + ex.GetType().Name);
                return;
            }
            Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "CUDA backend required.");
            AiDotNetEngine.Current = gpu;
            using var candidate = HiFiGanFusedExecutionTests.Create();
            Assert.True(candidate.ReleasesPredictResults);
            var parameters = candidate.GetParameters();

            AiDotNetEngine.Current = new CpuEngine();
            using var reference = HiFiGanFusedExecutionTests.Create();
            reference.SetParameters(parameters);
            var input = new Tensor<float>(Enumerable.Range(0, 128).Select(i => .2f * (float)Math.Sin(i * .17)).ToArray(),
                new[] { 1, 8, 16 });
            var expected = reference.Predict(input).ToArray();
            AiDotNetEngine.Current = gpu;
            using var retained = candidate.Predict(input);
            AssertClose(expected, retained.ToArray());
            gpu.GetBackend().Synchronize();
            long afterWarmup = GpuMemoryStats.Snapshot().AllocatedBytes;
            for (int repeat = 0; repeat < 6; repeat++)
            {
                using var output = candidate.Predict(input);
                AssertClose(expected, output.ToArray());
            }
            gpu.GetBackend().Synchronize();
            long afterRepeats = GpuMemoryStats.Snapshot().AllocatedBytes;
            Assert.True(afterRepeats <= afterWarmup + 65536,
                $"Prediction retained GPU scratch: warm={afterWarmup}, repeated={afterRepeats}.");
            AssertClose(expected, retained.ToArray());
            Assert.Equal(parameters.ToArray(), candidate.GetParameters().ToArray());

            // A later training step still consumes live weights and differentiates through fresh kernels.
            var audio = new Tensor<float>(Enumerable.Range(0, 256).Select(i => .1f * (float)Math.Cos(i * .11)).ToArray(),
                new[] { 1, 256 });
            candidate.Train(input, audio);
            var updated = candidate.GetParameters().ToArray();
            Assert.False(parameters.ToArray().SequenceEqual(updated));
            Assert.All(updated, value => Assert.True(!float.IsNaN(value) && !float.IsInfinity(value)));
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
            gpu?.Dispose();
        }
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < actual.Length; i++)
            Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 3e-4f * (1 + Math.Abs(expected[i])));
    }
}
