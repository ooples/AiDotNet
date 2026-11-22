using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class HiFiGanInferenceWorkspaceDemandTests
{
    [SkippableFact]
    public void PublicPredictRetiresOnlyIdleTrainingWorkspaceAndDifferentiablePredictPreservesIt()
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
            var parameters = candidate.GetParameters();
            var input = new Tensor<float>(Enumerable.Range(0, 128).Select(i => .2f * (float)Math.Sin(i * .17)).ToArray(),
                new[] { 1, 8, 16 });

            AiDotNetEngine.Current = new CpuEngine();
            using var reference = HiFiGanFusedExecutionTests.Create();
            reference.SetParameters(parameters);
            var expected = reference.Predict(input).ToArray();
            AiDotNetEngine.Current = gpu;
            using var retained = candidate.Predict(input);

            // A completed compiled step has already detached every observer from broker cells. Use one real native
            // owning cell to test the consumer boundary without relying on this small model's compiled cell sizes.
            var idle = gpu.GetBackend().AllocateBuffer(8);
            var broker = gpu.CompiledWorkspaceBroker;
            broker.Return(gpu.GetBackend(), idle);
            long before = broker.NativeReturnedForInferenceBytes;
            using (var tape = new GradientTape<float>())
            using (var differentiable = candidate.PredictWithGradients(input))
                Assert.NotEqual(IntPtr.Zero, idle.Handle);
            Assert.Equal(before, broker.NativeReturnedForInferenceBytes);

            using var output = candidate.Predict(input);
            Assert.Equal(IntPtr.Zero, idle.Handle);
            Assert.Equal(before + 8 * sizeof(float), broker.NativeReturnedForInferenceBytes);
            Assert.Equal(0, broker.RetainedBytes);
            AssertClose(expected, output.ToArray());
            AssertClose(expected, retained.ToArray());
            Assert.Equal(parameters.ToArray(), candidate.GetParameters().ToArray());
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
