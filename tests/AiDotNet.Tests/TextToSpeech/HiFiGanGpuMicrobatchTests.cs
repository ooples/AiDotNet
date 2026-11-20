using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class HiFiGanGpuMicrobatchTests
{
    [SkippableTheory]
    [InlineData(2, 5)]
    [InlineData(4, 9)]
    public void ChunkedInferencePreservesEverySampleTailAndEarlierUnreadOutput(int limit, int batch)
    {
        string? previousFlag = Environment.GetEnvironmentVariable("AIDOTNET_HIFI_GPU_INFERENCE_BATCH_LIMIT");
        var previousEngine = AiDotNetEngine.Current;
        DirectGpuTensorEngine? gpu = null;
        HiFiGanGenerator<float>? reference = null, candidate = null;
        try
        {
            try { gpu = new DirectGpuTensorEngine(); }
            catch (Exception e) when (e is InvalidOperationException or NotSupportedException or DllNotFoundException)
            { Skip.If(true, "CUDA unavailable: " + e.GetType().Name); return; }
            Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "Actual CUDA required.");
            AiDotNetEngine.Current = gpu;
            HiFiGanGenerator<float> Create(int chunkLimit)
            {
                Environment.SetEnvironmentVariable("AIDOTNET_HIFI_GPU_INFERENCE_BATCH_LIMIT", chunkLimit.ToString());
                var generator = new HiFiGanGenerator<float>(gpu, 4, 32, new[] { 2 }, new[] { 4 }, new[] { 3, 5, 7 },
                    Enumerable.Range(0, 3).Select(_ => new[] { 1, 3 }).ToArray(), true,
                    initialization: new Random(514), defaultInitialization: new Random(514));
                foreach (var layer in generator.Layers) layer.SetTrainingMode(false);
                Assert.Equal(chunkLimit, PrivateState.Value(typeof(HiFiGanGenerator<float>), "_gpuInferenceBatchLimit", generator));
                return generator;
            }
            reference = Create(0); candidate = Create(limit);
            for (int i = 0; i < candidate.Layers.Count; i++) reference.Layers[i].SetParameters(candidate.Layers[i].GetParameters());
            var parameters = candidate.Layers.Select(layer => layer.GetParameters().ToArray()).ToArray();
            using var owner = TensorScope.Begin();
            using var first = Input(batch, 11);
            var original = first.ToArray();
            var expectedFirst = reference.Forward(first).ToArray();
            var unread = candidate.Forward(first);
            foreach (int currentBatch in new[] { batch + 1, 1, batch })
            {
                using var input = Input(currentBatch, 7);
                var values = input.ToArray();
                using var step = TensorScope.Begin();
                var expected = reference.Forward(input).ToArray();
                var actual = candidate.Forward(input);
                Assert.Equal(new[] { currentBatch, 1, 14 }, actual.Shape.ToArray());
                AssertClose(expected, actual.ToArray());
                Assert.Equal(values, input.ToArray());
            }
            AssertClose(expectedFirst, unread.ToArray());
            Assert.Equal(original, first.ToArray());
            for (int i = 0; i < candidate.Layers.Count; i++) Assert.Equal(parameters[i], candidate.Layers[i].GetParameters().ToArray());
        }
        finally
        {
            if (reference is not null) foreach (var layer in reference.Layers) layer.Dispose();
            if (candidate is not null) foreach (var layer in candidate.Layers) layer.Dispose();
            AiDotNetEngine.Current = previousEngine;
            gpu?.Dispose();
            Environment.SetEnvironmentVariable("AIDOTNET_HIFI_GPU_INFERENCE_BATCH_LIMIT", previousFlag);
        }
    }

    private static Tensor<float> Input(int batch, int frames)
        => new(Enumerable.Range(0, batch * 4 * frames).Select(i => .2f * (float)Math.Sin(i * .13 + frames)
            + .01f * (i / (4 * frames))).ToArray(), new[] { batch, 4, frames });

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 5e-4f * (1 + Math.Abs(expected[i])), $"[{i}]: whole={expected[i]}, chunked={actual[i]}");
    }
}
