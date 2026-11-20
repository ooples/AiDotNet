using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class HiFiGanGpuDetachedLifetimeTests
{
    [SkippableFact]
    public void DetachedPairScopesPreserveTrainingModeInputsWeightsAndLaterDifferentiation()
    {
        string? previousFlag = Environment.GetEnvironmentVariable("AIDOTNET_HIFI_GPU_DETACHED_LIFETIMES");
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
                Environment.SetEnvironmentVariable("AIDOTNET_HIFI_GPU_DETACHED_LIFETIMES", enabled ? "1" : "0");
                var generator = new HiFiGanGenerator<float>(gpu, 4, 32, new[] { 2 }, new[] { 4 }, new[] { 3, 5, 7 },
                    Enumerable.Range(0, 3).Select(_ => new[] { 1, 3 }).ToArray(), true,
                    initialization: new Random(413), defaultInitialization: new Random(413));
                foreach (var layer in generator.Layers) layer.SetTrainingMode(true);
                return generator;
            }
            reference = Create(false); candidate = Create(true);
            for (int i = 0; i < candidate.Layers.Count; i++) reference.Layers[i].SetParameters(candidate.Layers[i].GetParameters());
            var parameters = candidate.Layers.Select(layer => layer.GetParameters().ToArray()).ToArray();
            Tensor<float>? retained = null;
            float[]? firstExpected = null;
            using var owner = TensorScope.Begin();
            foreach (int frames in new[] { 11, 7, 11 })
            {
                using var input = new Tensor<float>(Enumerable.Range(0, 8 * frames).Select(i => .2f * (float)Math.Sin(i * .13 + frames)).ToArray(), new[] { 2, 4, frames });
                var original = input.ToArray();
                using (new NoGradScope<float>())
                {
                    var expected = reference.ForwardDetachedForDiscriminator(input).ToArray();
                    var actual = candidate.ForwardDetachedForDiscriminator(input);
                    Assert.Equal(expected, actual.ToArray());
                    if (retained is null) { retained = actual; firstExpected = expected; }
                }
                Assert.Equal(original, input.ToArray());
                Assert.All(candidate.Layers, layer => Assert.True(layer.InTrainingMode));
            }
            Assert.Equal(firstExpected, retained!.ToArray());
            for (int i = 0; i < candidate.Layers.Count; i++) Assert.Equal(parameters[i], candidate.Layers[i].GetParameters().ToArray());
            using var differentiableInput = new Tensor<float>(Enumerable.Range(0, 88).Select(i => .01f * (i - 30)).ToArray(), new[] { 2, 4, 11 });
            using var tape = new GradientTape<float>();
            var waveform = candidate.Forward(differentiableInput);
            var gradients = tape.ComputeGradients(gpu.ReduceSum(waveform, null), sources: new[] { differentiableInput });
            Assert.True(gradients.TryGetValue(differentiableInput, out var gradient));
            Assert.NotNull(gradient);
            var values = gradient!.ToArray();
            Assert.All(values, value => Assert.True(!float.IsNaN(value) && !float.IsInfinity(value)));
            Assert.Contains(values, value => value != 0f);
        }
        finally
        {
            if (reference is not null) foreach (var layer in reference.Layers) layer.Dispose();
            if (candidate is not null) foreach (var layer in candidate.Layers) layer.Dispose();
            AiDotNetEngine.Current = previousEngine;
            gpu?.Dispose();
            Environment.SetEnvironmentVariable("AIDOTNET_HIFI_GPU_DETACHED_LIFETIMES", previousFlag);
        }
    }
}
