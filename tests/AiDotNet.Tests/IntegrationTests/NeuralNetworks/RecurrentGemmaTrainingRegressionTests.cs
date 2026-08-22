using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// End-to-end regression coverage for RecurrentGemma training and compiled-step fallback.
/// </summary>
public sealed class RecurrentGemmaTrainingRegressionTests
{
    [Fact(Timeout = 30000)]
    public async Task SmallLanguageModel_TrainingChangesFiniteParameters()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.TextGeneration,
            inputSize: 2,
            outputSize: 8);
        using var model = new RecurrentGemmaLanguageModel<float>(
            architecture,
            vocabSize: 8,
            modelDimension: 4,
            numLayers: 1,
            maxSeqLength: 2);

        var input = new Tensor<float>([1, 2]);
        input[0] = 1;
        input[1] = 3;

        var prediction = model.Predict(input);
        var target = new Tensor<float>(prediction.Shape.ToArray());
        int classes = prediction.Shape[^1];
        for (int row = 0; row < target.Length / classes; row++)
            target[row * classes + ((row + 2) % classes)] = 1.0f;

        var before = model.GetParameters().ToArray();
        Assert.NotEmpty(before);

        for (int step = 0; step < 3; step++)
            model.Train(input, target);

        var after = model.GetParameters().ToArray();
        Assert.Equal(before.Length, after.Length);
        Assert.All(after, value => Assert.True(float.IsFinite(value),
            $"Training produced a non-finite parameter: {value}."));
        Assert.Contains(Enumerable.Range(0, before.Length),
            i => MathF.Abs(after[i] - before[i]) > 1e-8f);
    }

    [Fact(Timeout = 120000)]
    public async Task GeneratedFixture_FirstCompiledStepFallsBackAndChangesFiniteParameters()
    {
        var harness = new GeneratedFixtureHarness();
        await harness.InitializeAsync();
        try
        {
            using var arena = TensorArena.Create();
            var (model, input, target) = harness.CreateTrainingCase();
            using (model)
            {
                var prediction = model.Predict(input);
                Assert.Equal(prediction.Shape.ToArray(), target.Shape.ToArray());
                Assert.Equal(prediction.Length / prediction.Shape[^1],
                    target.ToArray().Count(value => value == 1.0f));

                var before = model.GetParameters().ToArray();
                AiDotNet.Training.CompiledTapeTrainingStep<float>.ResetFusedStepCount();
                model.Train(input, target);
                var after = model.GetParameters().ToArray();

                Assert.True(
                    AiDotNet.Training.CompiledTapeTrainingStep<float>.GetFusedStepCount() > 0,
                    "The regression case must exercise the compiled optimizer before fallback.");
                Assert.Equal(before.Length, after.Length);
                Assert.All(after, value => Assert.True(float.IsFinite(value),
                    $"Training produced a non-finite parameter: {value}."));
                Assert.Contains(Enumerable.Range(0, before.Length), i => after[i] != before[i]);
            }
        }
        finally
        {
            await harness.DisposeAsync();
        }
    }

    private sealed class GeneratedFixtureHarness : NeuralNetworkModelTestBase<float>
    {
        protected override int[] InputShape => [128];
        protected override int[] OutputShape => [4];

        protected override INeuralNetworkModel<float> CreateNetwork() =>
            new RecurrentGemmaLanguageModel<float>(
                new NeuralNetworkArchitecture<float>(
                    inputType: InputType.OneDimensional,
                    taskType: NeuralNetworkTaskType.Regression,
                    inputSize: 128,
                    outputSize: 4),
                vocabSize: 4096);

        public (RecurrentGemmaLanguageModel<float> Model, Tensor<float> Input, Tensor<float> Target)
            CreateTrainingCase()
        {
            var rng = ModelTestHelpers.CreateSeededRandom();
            var model = (RecurrentGemmaLanguageModel<float>)CreateNetwork();
            var input = CreateRandomTensor(EffectiveInputShape, rng);
            var target = MakeTargetWellPosedForLoss(
                model,
                CreateRandomTargetTensor(ShapeCheckedOutputShape, rng),
                rng);
            return (model, input, target);
        }
    }
}
