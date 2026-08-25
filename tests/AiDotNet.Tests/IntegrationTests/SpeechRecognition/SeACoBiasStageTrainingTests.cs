using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.SpeechRecognition.AlibabaASR;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SpeechRecognition;

// The model's frozen-backbone path enters the process-wide inference-mode guard. Keep this
// regression isolated so it cannot suppress tape recording in an unrelated generated model test.
[CollectionDefinition("SeACo staged training", DisableParallelization = true)]
public sealed class SeACoStagedTrainingCollection { }

/// <summary>Regression coverage for SeACo's paper-specified frozen-backbone bias stage.</summary>
[Collection("SeACo staged training")]
public sealed class SeACoBiasStageTrainingTests
{
    [Fact]
    public void BiasStage_WithoutValidatedBiasBranch_IsRejectedBeforeObjectiveWork()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 4,
            outputSize: 5,
            layers: [new DenseLayer<double>(5)]);
        architecture.RandomSeed = 1234;
        var options = CreateSmallOptions(SeACoTrainingStage.Bias);

        using var model = new SeACo<double>(architecture, options);
        var input = new Tensor<double>([1, 4]);
        var target = new Tensor<double>([1, 5]);
        target[0, 1] = 1.0;

        var evaluationError = Assert.Throws<InvalidOperationException>(
            () => ((ITrainingObjectiveProvider<double>)model).EvaluateTrainingObjective(input, target));
        Assert.Contains("requires the native SeACo bias branch", evaluationError.Message);

        var trainingError = Assert.Throws<InvalidOperationException>(() => model.Train(input, target));
        Assert.Contains("requires the native SeACo bias branch", trainingError.Message);
    }

    [Fact]
    public void BiasTarget_UsesConfiguredNoBiasClass()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 4,
            outputSize: 5);
        architecture.RandomSeed = 1234;
        var options = CreateSmallOptions(SeACoTrainingStage.Bias);
        options.HotwordMaskTokenId = 2;

        using var model = new SeACo<double>(architecture, options);
        var labels = new Tensor<double>([2, 5]);
        labels[0, 3] = 1.0;
        labels[1, 4] = 1.0;

        var biasTarget = model.CreateBiasTrainingTarget(labels, row => row == 0);

        Assert.Equal(new[] { 2, 6 }, biasTarget.Shape);
        Assert.Equal(1.0, biasTarget[0, 3]);
        Assert.Equal(1.0, biasTarget[1, 2]);
        Assert.Equal(0.0, biasTarget[1, 5]);
    }

    [Fact(Timeout = 30000)]
    public async Task BiasStage_UpdatesBiasParameters_WhileBackboneRemainsBitStable()
    {
        await Task.Yield();
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 4,
            outputSize: 5);
        architecture.RandomSeed = 1234;
        var options = new SeACoOptions
        {
            EncoderDim = 8,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            NumAttentionHeads = 2,
            FeedForwardDim = 16,
            NumMels = 4,
            VocabSize = 5,
            MaxTextLength = 8,
            DropoutRate = 0,
            TrainingStage = SeACoTrainingStage.Bias,
            HotwordBatchRatio = 1,
            HotwordUtteranceRatio = 1,
            HotwordMinLength = 1,
            HotwordMaxLength = 2,
            Seed = 1234,
        };

        using var model = new SeACo<double>(architecture, options);
        var input = new Tensor<double>([1, 8, 4]);
        for (int i = 0; i < input.Length; i++)
            input.Data.Span[i] = ((i % 7) - 3) * 0.1;

        var prediction = model.Predict(input);
        var target = new Tensor<double>(prediction.Shape.ToArray());
        int classes = prediction.Shape[^1];
        for (int row = 0; row < target.Length / classes; row++)
            target.Data.Span[(row * classes) + ((row + 1) % classes)] = 1.0;

        // Materialize the lazily shaped bias layers without updating parameters, so the
        // before/after vectors describe the same complete parameter surface.
        _ = ((ITrainingObjectiveProvider<double>)model).EvaluateTrainingObjective(input, target);
        var before = model.GetParameters().ToArray();
        int backboneCount = checked((int)model.BackboneParameterCount);
        Assert.InRange(backboneCount, 1, before.Length - 1);

        model.Train(input, target);

        var after = model.GetParameters().ToArray();
        Assert.Equal(before.Length, after.Length);
        for (int i = 0; i < backboneCount; i++)
            Assert.True(
                before[i] == after[i],
                $"Backbone parameter {i} changed from {before[i]:R} to {after[i]:R} during Bias-stage training.");
        Assert.Contains(
            Enumerable.Range(backboneCount, before.Length - backboneCount),
            i => before[i] != after[i]);
    }

    private static SeACoOptions CreateSmallOptions(SeACoTrainingStage stage) => new()
    {
        EncoderDim = 8,
        NumEncoderLayers = 1,
        NumDecoderLayers = 1,
        NumAttentionHeads = 2,
        FeedForwardDim = 16,
        NumMels = 4,
        VocabSize = 5,
        MaxTextLength = 8,
        DropoutRate = 0,
        TrainingStage = stage,
        HotwordBatchRatio = 1,
        HotwordUtteranceRatio = 1,
        HotwordMinLength = 1,
        HotwordMaxLength = 2,
        Seed = 1234,
    };
}
