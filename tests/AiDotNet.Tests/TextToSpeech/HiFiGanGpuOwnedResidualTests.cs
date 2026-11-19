using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class HiFiGanGpuOwnedResidualTests
{
    [SkippableTheory]
    [InlineData(true, 2)]
    [InlineData(false, 2)]
    [InlineData(true, 0)]
    public void PrivateAccumulationMatchesOrdinaryExecutionAndProtectsCallerBoundaries(bool type1, int depth)
    {
        string? previousFlag = Environment.GetEnvironmentVariable("AIDOTNET_HIFI_GPU_OWNED_RESIDUALS");
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
            HiFiGanGenerator<float> Create(bool enabled)
            {
                Environment.SetEnvironmentVariable("AIDOTNET_HIFI_GPU_OWNED_RESIDUALS", enabled ? "1" : "0");
                var generator = new HiFiGanGenerator<float>(gpu, 4, 32, new[] { 2 }, new[] { 4 }, new[] { 3, 5, 7 },
                    Enumerable.Range(0, 3).Select(_ => Enumerable.Repeat(1, depth).ToArray()).ToArray(), type1,
                    conditionChannels: 3, tanhOutput: type1, conditionInEachUpsamplingLayer: true,
                    initialization: new Random(312), defaultInitialization: new Random(312));
                foreach (var layer in generator.Layers) layer.SetTrainingMode(false);
                Assert.Equal(enabled, typeof(HiFiGanGenerator<float>).GetField("_gpuOwnedResidualsEnabled", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(generator));
                return generator;
            }
            reference = Create(false); candidate = Create(true);
            for (int i = 0; i < candidate.Layers.Count; i++)
                reference.Layers[i].SetParameters(candidate.Layers[i].GetParameters());
            var parameters = candidate.Layers.Select(layer => layer.GetParameters().ToArray()).ToArray();
            using var owner = TensorScope.Begin();
            using var firstInput = Input(7, 0);
            using var condition = new Tensor<float>(new[] { .01f, -.02f, .03f, -.04f, .05f, -.06f }, new[] { 2, 3, 1 });
            var callerInput = firstInput.ToArray(); var callerCondition = condition.ToArray();
            var expectedFirst = reference.Forward(firstInput, condition).ToArray();
            var unreadFirst = candidate.Forward(firstInput, condition);
            foreach (int frames in new[] { 7, 11, 7 })
            {
                using var input = Input(frames, frames);
                var inputValues = input.ToArray();
                using var temporary = TensorScope.Begin();
                var expected = reference.Forward(input, condition).ToArray();
                var actual = candidate.Forward(input, condition).ToArray();
                Assert.Equal(expected, actual);
                Assert.Equal(inputValues, input.ToArray());
                Assert.Equal(callerCondition, condition.ToArray());
            }
            Assert.Equal(expectedFirst, unreadFirst.ToArray());
            Assert.Equal(callerInput, firstInput.ToArray());
            for (int i = 0; i < candidate.Layers.Count; i++) Assert.Equal(parameters[i], candidate.Layers[i].GetParameters().ToArray());
        }
        finally
        {
            if (reference is not null) foreach (var layer in reference.Layers) layer.Dispose();
            if (candidate is not null) foreach (var layer in candidate.Layers) layer.Dispose();
            AiDotNetEngine.Current = previousEngine;
            gpu?.Dispose();
            Environment.SetEnvironmentVariable("AIDOTNET_HIFI_GPU_OWNED_RESIDUALS", previousFlag);
        }
    }

    private static Tensor<float> Input(int frames, int seed)
        => new(Enumerable.Range(0, 8 * frames).Select(i => .2f * (float)Math.Sin(i * .13 + seed)).ToArray(), new[] { 2, 4, frames });
}
