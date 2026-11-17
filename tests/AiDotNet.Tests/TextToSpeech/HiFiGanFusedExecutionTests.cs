using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public class HiFiGanFusedExecutionTests
{
    internal static HiFiGAN<float> Create() => new(
        new NeuralNetworkArchitecture<float>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 64) { RandomSeed = 11 },
        new HiFiGANOptions
        {
            MelChannels = 8, UpsampleRates = [4, 4], UpsampleKernelSizes = [8, 8], UpsampleInitialChannels = 16,
            ResblockKernelSizes = [3], ResblockDilationSizes = [[1, 3]], DiscriminatorPeriods = [2, 3], HopSize = 16,
            FftSize = 64, WindowSize = 64, SampleRate = 4000, MelMaxFrequency = 2000, SegmentSize = 256,
            DiscriminatorWidthDivisor = 16, SamplingSeed = 11,
        });

    [Fact(Timeout = 240000)]
    public async Task CompiledGanStepsEngageAndPreserveTheEagerTrajectory()
    {
        await Task.Yield();
        var settings = TensorCodecOptions.Current;
        bool previous = settings.EnableCompilation;
        try
        {
            using var eager = Create();
            using var compiled = Create();
            compiled.SetParameters(eager.GetParameters());
            for (int step = 0; step < 3; step++)
            {
                var audio = new Tensor<float>(new[] { 512 });
                for (int i = 0; i < audio.Length; i++)
                    audio[i] = (float)(.5 * Math.Sin(2 * Math.PI * (220 + step * 15) * i / 4000.0));
                var mel = eager.ComputeMel(audio);
                settings.EnableCompilation = false;
                eager.Train(mel, audio);
                settings.EnableCompilation = true;
                compiled.Train(mel, audio);
            }
            Assert.Equal(6, compiled.CustomFusedStepCount);
            Assert.Equal(0, compiled.CustomEagerStepCount);
            Assert.True(compiled.ObjectiveSessionCommitted, compiled.LastObjectiveMissReason ?? "No fused plan committed.");
            var expected = eager.GetParameters();
            var actual = compiled.GetParameters();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.InRange(Math.Abs(expected[i] - actual[i]), 0, 2e-4 * Math.Max(1, Math.Abs(expected[i])));
        }
        finally { settings.EnableCompilation = previous; }
    }
}
