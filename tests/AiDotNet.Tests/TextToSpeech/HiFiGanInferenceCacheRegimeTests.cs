using System.Reflection;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class HiFiGanInferenceCacheRegimeTests
{
    [SkippableFact]
    public void PublicProgressivePredictReusesTheNewClassAndPreservesRetainedWaveformAliasesAndTraining()
    {
        var previous = AiDotNetEngine.Current;
        DirectGpuTensorEngine? gpu = null;
        try
        {
            try { gpu = new DirectGpuTensorEngine(); }
            catch (Exception error) when (error is InvalidOperationException or NotSupportedException or DllNotFoundException)
            { Skip.If(true, "CUDA unavailable: " + error.GetType().Name); return; }
            Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "Actual CUDA required.");
            AiDotNetEngine.Current = gpu;
            using var candidate = HiFiGanFusedExecutionTests.Create();
            using var input = Input(1);
            using var audio = new Tensor<float>(Enumerable.Range(0, 256).Select(index => .1f * (float)Math.Cos(index * .11)).ToArray(), new[] { 1, 256 });
            candidate.Train(input, audio);
            var trainedParameters = candidate.GetParameters();

            var cpu = new CpuEngine();
            AiDotNetEngine.Current = cpu;
            using var reference = HiFiGanFusedExecutionTests.Create();
            reference.SetParameters(trainedParameters);
            var expectedFirst = reference.Predict(input).ToArray();
            AiDotNetEngine.Current = gpu;
            using var retained = candidate.Predict(input); // Keep the actual returned waveform unread during reclamation.
            using var alias = gpu.Reshape(retained, new[] { retained.Length });

            object pool = typeof(CudaBackend).GetField("_bufferPool", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(gpu.GetBackend())!;
            const BindingFlags internalInstance = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
            long Requests() => (long)pool.GetType().GetProperty("InferenceRegimeNativeReleaseRequestedBuffers", internalInstance)!.GetValue(pool)!;
            // Isolate the idle-cache admission fixture without touching any parameter, result, alias or driver-pool threshold.
            pool.GetType().GetMethod("DrainAll", internalInstance)!.Invoke(pool, null);
            var idle = gpu.GetBackend()!.AllocateBuffer(1);
            idle.Dispose();
            long before = Requests();
            foreach (int batch in new[] { 1, 4, 8 })
            {
                using var features = Input(batch);
                AiDotNetEngine.Current = cpu;
                var expected = reference.Predict(features).ToArray();
                AiDotNetEngine.Current = gpu;
                using var actual = candidate.Predict(features); // Actual NNBase public boundary, not a manually-entered scope.
                AssertClose(expected, actual.ToArray());
                using var repeated = candidate.Predict(features);
                AssertClose(expected, repeated.ToArray());
            }
            Assert.True(Requests() > before, "The actual public Predict must engage idle-cache regime retirement.");
            AssertClose(expectedFirst, alias.ToArray());
            AssertClose(expectedFirst, retained.ToArray());
            Assert.Equal(trainedParameters.ToArray(), candidate.GetParameters().ToArray());
            candidate.Train(input, audio);
            var updated = candidate.GetParameters().ToArray();
            Assert.False(trainedParameters.ToArray().SequenceEqual(updated));
            Assert.All(updated, value => Assert.True(float.IsFinite(value)));
        }
        finally { AiDotNetEngine.Current = previous; gpu?.Dispose(); }
    }

    private static Tensor<float> Input(int batch)
        => new(Enumerable.Range(0, batch * 8 * 16).Select(index => .2f * (float)Math.Sin(index * .17)).ToArray(), new[] { batch, 8, 16 });

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int index = 0; index < actual.Length; ++index)
            Assert.InRange(Math.Abs(actual[index] - expected[index]), 0, 3e-4f * (1 + Math.Abs(expected[index])));
    }
}
