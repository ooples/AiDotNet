using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("EngineCurrentGlobalState")]
public sealed class GanFeatureMatchingReplayTests
{
    private sealed class VocoderProbe : HiFiGAN<double>
    {
        internal VocoderProbe() : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 16) { RandomSeed = 11 },
            new HiFiGANOptions
            {
                MelChannels = 8, UpsampleRates = [4, 4], UpsampleKernelSizes = [8, 8], UpsampleInitialChannels = 16,
                ResblockKernelSizes = [3], ResblockDilationSizes = [[1, 3]], DiscriminatorPeriods = [2, 3], HopSize = 16,
                FftSize = 64, WindowSize = 64, SampleRate = 4000, MelMaxFrequency = 2000, SegmentSize = 128,
                DiscriminatorWidthDivisor = 16,
            }) { }

        internal Tensor<double> Match(Tensor<double> real, Tensor<double> fake) => FeatureMatching([real], [fake]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Compiled_feature_matching_rebinds_current_targets_without_training_the_target_branch(bool universalCodec)
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            using var vocoder = new VocoderProbe();
            var codec = new UniAudioCodecGan<double>(engine, 320, new UniAudioCodecTrainingOptions
            {
                DiscriminatorHops = [2], DiscriminatorChannels = [32],
                FullFftSizes = [8], FullHopSizes = [2], FullWindowSizes = [8],
                SubFftSizes = [8], SubHopSizes = [2], SubWindowSizes = [8], Subbands = 2,
                FeatureMatchingWeight = 2,
            });
            var realWeights = new Tensor<double>(new[] { 1.0, 2.0, -2.0, 4.0 }, new[] { 4 });
            var fakeWeights = new Tensor<double>(new[] { 3.0, -1.0, 0.0, 2.0 }, new[] { 4 });
            var parameters = new[] { realWeights, fakeWeights };
            var input = new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, new[] { 4 });
            var score = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
            Tensor<double> Objective()
            {
                var real = engine.TensorMultiply(realWeights, input);
                var fake = engine.TensorMultiply(fakeWeights, input);
                return universalCodec
                    ? codec.AdversarialGeneratorObjective(new UniAudioCriticOutput<double>([score], [[real]]),
                        new UniAudioCriticOutput<double>([score], [[fake]]))
                    : vocoder.Match(real, fake);
            }

            using var recording = ClosureObjectiveGuard<double>.BeginRecording();
            CompiledTrainingPlan<double> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
                plan = scope.CompileTraining(parameters, Objective());
            using (plan)
            {
                var guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
                Assert.True(guard is not null, reason);
                Assert.True(guard!.HasForeignLeaves);
                guard.AttachTo(plan);
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.125);
                var expectedFake = fakeWeights.ToArray();
                var cases = new[]
                {
                    (Real: new[] { 1.0, 2.0, -2.0, 4.0 }, Input: new[] { 1.0, 2.0, 3.0, 4.0 }),
                    (Real: new[] { -3.0, 0.0, 5.0, 1.0 }, Input: new[] { -2.0, 0.5, 1.5, 3.0 }),
                    (Real: new[] { 2.0, -4.0, 1.0, 6.0 }, Input: new[] { 0.25, -3.0, 2.0, 1.0 }),
                };
                foreach (var item in cases)
                {
                    // Mutate the reference parameters and replace the foreign input after tracing.
                    // A copied trace-time target would retain the first values and fail this oracle.
                    for (int i = 0; i < item.Real.Length; i++) realWeights[i] = item.Real[i];
                    input = new Tensor<double>(item.Input, new[] { 4 });
                    double scale = universalCodec ? 2.0 : 1.0;
                    var gradient = new double[4];
                    double expectedLoss = 0;
                    for (int i = 0; i < gradient.Length; i++)
                    {
                        double difference = (expectedFake[i] - item.Real[i]) * item.Input[i];
                        expectedLoss += scale * Math.Abs(difference) / gradient.Length;
                        gradient[i] = scale * Math.Sign(difference) * item.Input[i] / gradient.Length;
                    }
                    Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(Objective, parameters, out _));
                    Assert.Equal(expectedLoss, plan.Step().ToArray()[0], 10);
                    Assert.All(plan.Gradients[0].ToArray(), value => Assert.Equal(0.0, value));
                    Assert.Equal(gradient, plan.Gradients[1].ToArray());
                    Assert.Equal(item.Real, realWeights.ToArray());
                    for (int i = 0; i < gradient.Length; i++) expectedFake[i] -= 0.125 * gradient[i];
                    Assert.Equal(expectedFake, fakeWeights.ToArray());
                }
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
