using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

[Collection("FusedTrainingSerial")]
public class DisabledGroupedClippingFusedTests
{
    private sealed class GroupedNetwork : FeedForwardNeuralNetwork<float>
    {
        internal GroupedNetwork(double clip) : base(
            new NeuralNetworkArchitecture<float>(inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression, complexity: NetworkComplexity.Simple,
                inputSize: 3, outputSize: 2),
            new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                { InitialLearningRate = .01, UseAdaptiveLearningRate = false }),
            new MeanSquaredErrorLoss<float>())
            => MaxGradNorm = NumOps.FromDouble(clip);

        protected override IReadOnlyList<IReadOnlyList<Tensor<float>>>? GradientClippingGroups(
            IReadOnlyList<Tensor<float>> trainableParameters) => new[] { trainableParameters };

        internal void Step(Tensor<float> input, Tensor<float> target)
            => TrainWithCustomObjective(input, target, (x, y) =>
                new MeanSquaredErrorLoss<float>().ComputeTapeLoss(ForwardForTraining(x), y));
    }

    [Theory]
    [InlineData(0.0, true)]
    [InlineData(1.0, false)]
    public void GroupOverrideOnlyRefusesFusionWhenNormClippingIsEnabled(double clip, bool expectFused)
    {
        var options = TensorCodecOptions.Current;
        bool previous = options.EnableCompilation;
        try
        {
            using var eager = new GroupedNetwork(clip);
            using var candidate = new GroupedNetwork(clip);
            var x = new Tensor<float>(new[] { 2, 3 }, new Vector<float>(new[] { .2f, .4f, .1f, -.3f, .5f, .7f }));
            eager.Predict(x);
            candidate.Predict(x);
            candidate.SetParameters(eager.GetParameters());
            for (int step = 0; step < 3; step++)
            {
                var target = new Tensor<float>(new[] { 2, 2 }, new Vector<float>(
                    new[] { .1f + step * .03f, -.2f, .3f, .05f - step * .01f }));
                options.EnableCompilation = false;
                eager.Step(x, target);
                options.EnableCompilation = true;
                candidate.Step(x, target);
            }
            Assert.Equal(expectFused ? 3 : 0, candidate.CustomFusedStepCount);
            // A refusal before the compiled session routes directly to eager, outside the session's miss counter.
            Assert.Equal(0, candidate.CustomEagerStepCount);
            var expected = eager.GetParameters();
            var actual = candidate.GetParameters();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.InRange(Math.Abs(expected[i] - actual[i]), 0, 1e-4);
        }
        finally { options.EnableCompilation = previous; }
    }
}
