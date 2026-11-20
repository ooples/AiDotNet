using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class GanTrainingInputLifetimeTests
{
    private static IGpuBuffer? Buffer(Tensor<float> tensor)
    {
        for (Type? type = tensor.GetType(); type is not null; type = type.BaseType)
        {
            var property = type.GetProperty("_gpuBuffer", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
            if (property is not null) return (IGpuBuffer?)property.GetValue(tensor);
        }
        throw new InvalidOperationException("Device binding metadata unavailable.");
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void OwnedInputReleasePreservesCallerViewsAndReturnsUploadedStorage(bool cuda)
    {
        var previous = AiDotNetEngine.Current;
        DirectGpuTensorEngine? gpu = null;
        try
        {
            IEngine engine = new CpuEngine();
            if (cuda)
            {
                gpu = new DirectGpuTensorEngine();
                Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "Actual CUDA required.");
                engine = gpu;
            }
            AiDotNetEngine.Current = engine;
            var values = Enumerable.Range(0, 32).Select(i => .01f * (i - 7)).ToArray();
            using var caller = new Tensor<float>(values, new[] { 1, 4, 8 });
            using var target = new Tensor<float>(values.ToArray(), new[] { 32 });
            gpu?.EnsureResidentInput(caller);
            var callerBuffer = Buffer(caller);
            var view = engine.Reshape(caller, new[] { 4, 8 });
            GanVocoderBase<float>.ReleaseOwnedTrainingInput(engine, view, caller, target);
            Assert.Equal(values, caller.ToArray());
            if (cuda) { Assert.Same(callerBuffer, Buffer(caller)); Assert.NotEqual(IntPtr.Zero, callerBuffer!.Handle); }

            for (int iteration = 0; iteration < 16; iteration++)
            {
                var temporary = new Tensor<float>(values.ToArray(), new[] { 32 });
                gpu?.EnsureResidentInput(temporary);
                var uploaded = Buffer(temporary);
                IntPtr address = uploaded?.Handle ?? IntPtr.Zero;
                GanVocoderBase<float>.ReleaseOwnedTrainingInput(engine, temporary, caller, target);
                if (cuda)
                {
                    Assert.Equal(IntPtr.Zero, uploaded!.Handle);
                    using var reused = gpu!.GetBackend().AllocateBuffer(32);
                    Assert.Equal(address, reused.Handle);
                    Assert.All(gpu.GetBackend().DownloadBuffer(reused), value => Assert.Equal(0f, value));
                }
            }
            Assert.Equal(values, caller.ToArray());
            Assert.Equal(values, target.ToArray());
        }
        finally { AiDotNetEngine.Current = previous; gpu?.Dispose(); }
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void AlternatingUpdatesAndPredictionKeepCallerRecordingsAndViewsValid(bool cuda)
    {
        var previous = AiDotNetEngine.Current;
        string? previousFlag = Environment.GetEnvironmentVariable("AIDOTNET_GAN_TRAINING_INPUT_LIFETIMES");
        DirectGpuTensorEngine? gpu = null;
        try
        {
            IEngine engine = new CpuEngine();
            if (cuda)
            {
                gpu = new DirectGpuTensorEngine();
                Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "Actual CUDA required.");
                engine = gpu;
            }
            AiDotNetEngine.Current = engine;
            Environment.SetEnvironmentVariable("AIDOTNET_GAN_TRAINING_INPUT_LIFETIMES", "1");
            using var model = new HiFiGAN<float>(new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
                NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 16) { RandomSeed = 11 },
                new HiFiGANOptions
                {
                    MelChannels = 8, UpsampleRates = new[] { 4, 4 }, UpsampleKernelSizes = new[] { 8, 8 }, UpsampleInitialChannels = 16,
                    ResblockKernelSizes = new[] { 3 }, ResblockDilationSizes = new[] { new[] { 1, 3 } }, DiscriminatorPeriods = new[] { 2, 3 },
                    HopSize = 16, FftSize = 64, WindowSize = 64, SampleRate = 4000, MelMaxFrequency = 2000,
                    SegmentSize = 128, DiscriminatorWidthDivisor = 16,
                });
            var initial = model.GetParameters().ToArray();
            foreach (int frames in new[] { 12, 16, 10, 16 })
            {
                var melValues = Enumerable.Range(0, 8 * frames).Select(i => .1f * MathF.Sin(i * .07f + frames)).ToArray();
                var audioValues = Enumerable.Range(0, 16 * frames).Select(i => .2f * MathF.Sin(i * .17f + frames)).ToArray();
                using var mel = new Tensor<float>(melValues, new[] { 1, 8, frames });
                using var audio = new Tensor<float>(audioValues, new[] { 16 * frames });
                using var retainedView = engine.Reshape(mel, new[] { 8, frames });
                model.Train(mel, audio);
                Assert.Equal(melValues, mel.ToArray());
                Assert.Equal(melValues, retainedView.ToArray());
                Assert.Equal(audioValues, audio.ToArray());
                using var predicted = model.Predict(mel);
                Assert.Equal(16 * frames, predicted.Length);
                Assert.All(predicted.ToArray(), value => Assert.True(float.IsFinite(value)));
            }
            Assert.False(initial.SequenceEqual(model.GetParameters().ToArray()));
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_GAN_TRAINING_INPUT_LIFETIMES", previousFlag);
            AiDotNetEngine.Current = previous;
            gpu?.Dispose();
        }
    }
}
