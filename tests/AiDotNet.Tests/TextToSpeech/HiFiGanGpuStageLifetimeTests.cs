using AiDotNet.NeuralNetworks.Graph;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class HiFiGanGpuStageLifetimeTests
{
    private static HiFiGanGenerator<float> Create(IEngine engine, int depth, bool type1 = true, bool condition = false)
    {
        var result = new HiFiGanGenerator<float>(engine, 4, 32, new[] { 2 }, new[] { 4 }, new[] { 3, 5, 7 },
            Enumerable.Range(0, 3).Select(_ => Enumerable.Repeat(1, depth).ToArray()).ToArray(), type1,
            conditionChannels: condition ? 3 : 0, tanhOutput: type1,
            conditionInEachUpsamplingLayer: condition);
        foreach (var layer in result.Layers) layer.SetTrainingMode(false);
        return result;
    }

    private static Tensor<float> Input(int frames, int seed)
        => new(Enumerable.Range(0, 8 * frames).Select(i => .2f * (float)Math.Sin(i * .13 + seed)).ToArray(),
            new[] { 2, 4, frames });

    private static Tensor<float> Condition(int seed)
        => new(Enumerable.Range(0, 6).Select(i => .03f * (i + seed)).ToArray(), new[] { 2, 3, 1 });

    [SkippableTheory]
    [InlineData(true, 3)]
    [InlineData(false, 3)]
    [InlineData(true, 0)] // Every branch is its shared stage input: exercise identity/alias carry protection.
    public void ChangingInputsAndConditionsMatchCpuAndKeepAnUnreadEarlierWaveform(bool type1, int depth)
        => WithCuda(gpu =>
        {
            var cpu = new CpuEngine();
            var candidate = Create(gpu, depth, type1, condition: true);
            AiDotNetEngine.Current = cpu;
            var reference = Create(cpu, depth, type1, condition: true);
            try
            {
                for (int i = 0; i < candidate.Layers.Count; i++)
                    reference.Layers[i].SetParameters(candidate.Layers[i].GetParameters());
                var firstInput = Input(11, 0);
                var firstCondition = Condition(0);
                var inputBytes = firstInput.ToArray();
                var conditionBytes = firstCondition.ToArray();
                var expectedFirst = reference.Forward(firstInput, firstCondition).ToArray();
                AiDotNetEngine.Current = gpu;
                using var owner = TensorScope.Begin();
                var retained = candidate.Forward(firstInput, firstCondition); // Deliberately not materialized yet.
                for (int step = 1; step <= 3; step++)
                {
                    using var input = Input(10 + step, step);
                    using var condition = Condition(step);
                    AiDotNetEngine.Current = cpu;
                    var expected = reference.Forward(input, condition).ToArray();
                    AiDotNetEngine.Current = gpu;
                    using var scratch = TensorScope.Begin();
                    var actual = candidate.Forward(input, condition);
                    AssertClose(expected, actual.ToArray());
                }
                AssertClose(expectedFirst, retained.ToArray());
                Assert.Equal(inputBytes, firstInput.ToArray());
                Assert.Equal(conditionBytes, firstCondition.ToArray());
            }
            finally
            {
                AiDotNetEngine.Current = cpu;
                Dispose(reference);
                AiDotNetEngine.Current = gpu;
                Dispose(candidate);
            }
        });

    [SkippableFact]
    public void WarmedLiveScratchPeakIsBoundedByTheCurrentPairInsteadOfResidualDepth()
        => WithCuda(gpu =>
        {
            var shallow = Create(gpu, 1);
            var deep = Create(gpu, 8);
            using var input = Input(512, 0);
            try
            {
                long Measure(HiFiGanGenerator<float> generator)
                {
                    for (int warm = 0; warm < 2; warm++)
                    {
                        using var scope = TensorScope.Begin();
                        _ = generator.Forward(input);
                        gpu.GetBackend().Synchronize();
                    }
                    gpu.GetBackend().Synchronize();
                    long baseline = GpuMemoryStats.Snapshot(gpu.GetBackend()).AllocatedBytes;
                    GpuMemoryStats.ResetPeakStats(gpu.GetBackend());
                    long peak;
                    using (var scope = TensorScope.Begin())
                    {
                        _ = generator.Forward(input);
                        gpu.GetBackend().Synchronize();
                        peak = GpuMemoryStats.Snapshot(gpu.GetBackend()).PeakAllocatedBytes - baseline;
                    }
                    gpu.GetBackend().Synchronize();
                    return peak;
                }
                long shallowPeak = Measure(shallow);
                long deepPeak = Measure(deep);
                Assert.True(shallowPeak > 0, "The device allocator must observe the inference scratch allocations.");
                Assert.True(deepPeak <= shallowPeak * 1.25 + 256 * 1024,
                    $"Warmed live scratch grew with residual depth: shallow={shallowPeak}, deep={deepPeak}.");
            }
            finally { Dispose(shallow); Dispose(deep); }
        });

    [SkippableFact]
    public void ActiveTapeAndLayerObserversKeepTheirIntermediateValues()
        => WithCuda(gpu =>
        {
            var generator = Create(gpu, 1);
            using var input = Input(11, 0);
            using var owner = TensorScope.Begin();
            try
            {
                using (var tape = new GradientTape<float>())
                {
                    var output = generator.Forward(input);
                    var loss = gpu.ReduceSum(output, null);
                    var gradients = tape.ComputeGradients(loss, sources: new[] { input });
                    Assert.True(gradients.TryGetValue(input, out var inputGradient));
                    if (inputGradient is not null)
                    {
                        Assert.All(inputGradient.ToArray(), value => Assert.True(!float.IsNaN(value) && !float.IsInfinity(value)));
                        Assert.Contains(inputGradient.ToArray(), value => value != 0f);
                    }
                }
                using var observer = new LayerForwardObserver<float>();
                _ = generator.Forward(input);
                Assert.NotEmpty(observer.Calls);
                foreach (var call in observer.Calls)
                    Assert.All(call.Output.ToArray(), value => Assert.True(!float.IsNaN(value) && !float.IsInfinity(value)));
            }
            finally { Dispose(generator); }
        });

    private static void Dispose(HiFiGanGenerator<float> generator)
    {
        foreach (var layer in generator.Layers) layer.Dispose();
    }

    private static void WithCuda(Action<DirectGpuTensorEngine> action)
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
            action(gpu);
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
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 5e-4f * (1 + Math.Abs(expected[i])),
                $"[{i}]: cpu={expected[i]}, gpu={actual[i]}");
    }
}
