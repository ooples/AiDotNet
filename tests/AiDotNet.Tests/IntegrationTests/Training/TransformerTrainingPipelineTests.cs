using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.Training;

/// <summary>
/// End-to-end integration tests for the Transformer training pipeline.
/// Uses the <see cref="AiModelBuilder{T, TInput, TOutput}"/> facade
/// exactly as a real user would — constructs a Transformer, feeds
/// synthetic sequence data, and trains through BuildAsync(). Exercises
/// all three bugs in the chain: #1113 (feature selection), #1114
/// (loss shape mismatch), #1115 (gradient vector flatten).
/// Also covers #1121 (TokenClassification vs SequenceClassification).
/// </summary>
public class TransformerTrainingPipelineTests
{
    private readonly ITestOutputHelper _output;

    public TransformerTrainingPipelineTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// TokenClassification: per-position labels [B, S] through the facade.
    /// Before #1121 fix, SequenceClassification was used and GlobalPoolingLayer
    /// collapsed the sequence dimension, causing shape mismatch.
    /// </summary>
    [Fact]
    public async Task TokenClassification_ThroughFacade_CompletesWithoutCrash()
    {
        // Arrange: small Transformer for token classification (per-position labels)
        int vocabSize = 20;
        int seqLen = 4;
        int numSamples = 16;
        int modelDim = 32;
        int numClasses = 5;

        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.TokenClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: modelDim,
            feedForwardDimension: 64,
            inputSize: seqLen,
            outputSize: numClasses,
            maxSequenceLength: seqLen,
            vocabularySize: vocabSize);

        var transformer = new Transformer<float>(
            architecture,
            lossFunction: new CategoricalCrossEntropyLoss<float>());

        // Verify no GlobalPoolingLayer was added (TokenClassification keeps sequence dim)
        Assert.DoesNotContain(transformer.Layers,
            l => l is GlobalPoolingLayer<float>);

        // Synthetic data: random token sequences with integer class targets per position
        var rng = RandomHelper.CreateSeededRandom(42);
        var inputData = new Tensor<float>([numSamples, seqLen]);
        var targetData = new Tensor<float>([numSamples, seqLen]);
        for (int i = 0; i < numSamples; i++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                inputData[i, s] = rng.Next(vocabSize);
                targetData[i, s] = rng.Next(numClasses);
            }
        }

        // Act: train through the facade
        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureDataLoader(
                new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(inputData, targetData))
            .ConfigureModel(transformer);

        var result = await builder.BuildAsync();

        // Assert: training completed and produced a valid result
        Assert.NotNull(result);
        Assert.NotNull(result.Model);
        _output.WriteLine("TokenClassification training through AiModelBuilder facade completed.");
    }

    /// <summary>
    /// SequenceClassification: one label per sequence [B] through the facade.
    /// Verifies that GlobalPoolingLayer is present and per-sequence labels work correctly.
    /// </summary>
    [Fact]
    public async Task SequenceClassification_ThroughFacade_CompletesWithoutCrash()
    {
        int vocabSize = 20;
        int seqLen = 4;
        int numSamples = 16;
        int modelDim = 32;
        int numClasses = 5;

        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: modelDim,
            feedForwardDimension: 64,
            inputSize: seqLen,
            outputSize: numClasses,
            maxSequenceLength: seqLen,
            vocabularySize: vocabSize);

        var transformer = new Transformer<float>(
            architecture,
            lossFunction: new CategoricalCrossEntropyLoss<float>());

        // Verify GlobalPoolingLayer IS present for SequenceClassification
        Assert.Contains(transformer.Layers,
            l => l is GlobalPoolingLayer<float>);

        // Synthetic data: per-sequence labels (1D targets)
        var rng = RandomHelper.CreateSeededRandom(42);
        var inputData = new Tensor<float>([numSamples, seqLen]);
        var targetData = new Tensor<float>([numSamples]);
        for (int i = 0; i < numSamples; i++)
        {
            targetData[i] = rng.Next(numClasses);
            for (int s = 0; s < seqLen; s++)
            {
                inputData[i, s] = rng.Next(vocabSize);
            }
        }

        // Act: train through the facade
        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureDataLoader(
                new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(inputData, targetData))
            .ConfigureModel(transformer);

        var result = await builder.BuildAsync();

        Assert.NotNull(result);
        Assert.NotNull(result.Model);
        _output.WriteLine("SequenceClassification training through AiModelBuilder facade completed.");
    }

    /// <summary>
    /// Verifies InferClassificationTaskType correctly identifies TokenClassification
    /// from [B, S] targets and SequenceClassification from [B] targets.
    /// </summary>
    [Fact]
    public void InferClassificationTaskType_DetectsCorrectTaskType()
    {
        int seqLen = 8;

        // Per-position targets [numSamples, seqLen] → TokenClassification
        var tokenResult = TransformerArchitecture<float>.InferClassificationTaskType(
            [100, seqLen], seqLen);
        Assert.Equal(NeuralNetworkTaskType.TokenClassification, tokenResult);

        // Per-sequence targets [numSamples] → SequenceClassification
        var seqResult = TransformerArchitecture<float>.InferClassificationTaskType(
            [100], seqLen);
        Assert.Equal(NeuralNetworkTaskType.SequenceClassification, seqResult);

        // Per-sequence targets with different second dim → SequenceClassification
        var mismatchResult = TransformerArchitecture<float>.InferClassificationTaskType(
            [100, 3], seqLen);
        Assert.Equal(NeuralNetworkTaskType.SequenceClassification, mismatchResult);

        _output.WriteLine("InferClassificationTaskType correctly detects task types.");
    }

    /// <summary>
    /// Verifies ValidateTaskTypeVsTargetShape throws a clear error when
    /// SequenceClassification is configured but targets have per-position labels.
    /// </summary>
    [Fact]
    public void ValidateTaskType_SequenceClassification_WithPerPositionTargets_ThrowsClearError()
    {
        int seqLen = 4;

        var ex = Assert.Throws<InvalidOperationException>(() =>
            TransformerArchitecture<float>.ValidateTaskTypeVsTargetShape(
                NeuralNetworkTaskType.SequenceClassification,
                [16, seqLen],
                seqLen));

        Assert.Contains("TokenClassification", ex.Message);
        Assert.Contains("InferClassificationTaskType", ex.Message);
        _output.WriteLine($"Clear error message: {ex.Message}");
    }

    /// <summary>
    /// Verifies ValidateTaskTypeVsTargetShape throws a clear error when
    /// TokenClassification is configured but targets have per-sequence labels.
    /// </summary>
    [Fact]
    public void ValidateTaskType_TokenClassification_WithPerSequenceTargets_ThrowsClearError()
    {
        int seqLen = 4;

        var ex = Assert.Throws<InvalidOperationException>(() =>
            TransformerArchitecture<float>.ValidateTaskTypeVsTargetShape(
                NeuralNetworkTaskType.TokenClassification,
                [16],
                seqLen));

        Assert.Contains("SequenceClassification", ex.Message);
        Assert.Contains("InferClassificationTaskType", ex.Message);
        _output.WriteLine($"Clear error message: {ex.Message}");
    }

    /// <summary>
    /// Verifies ValidateTaskTypeVsTargetShape passes when configuration matches target shape.
    /// </summary>
    [Fact]
    public void ValidateTaskType_CorrectConfig_DoesNotThrow()
    {
        int seqLen = 4;

        // TokenClassification + per-position targets → OK
        TransformerArchitecture<float>.ValidateTaskTypeVsTargetShape(
            NeuralNetworkTaskType.TokenClassification,
            [16, seqLen],
            seqLen);

        // SequenceClassification + per-sequence targets → OK
        TransformerArchitecture<float>.ValidateTaskTypeVsTargetShape(
            NeuralNetworkTaskType.SequenceClassification,
            [16],
            seqLen);

        _output.WriteLine("Correct configurations pass validation.");
    }

    /// <summary>
    /// Verifies that CategoricalCrossEntropyLoss, CrossEntropyLoss, and
    /// FocalLoss all handle integer targets (rank less than predictions)
    /// by auto one-hot encoding via EnsureTargetMatchesPredicted.
    /// </summary>
    [Fact]
    public void CrossEntropyLosses_HandleIntegerTargets()
    {
        // predicted: [batch=2, seq=3, vocab=5]
        var predicted = new Tensor<float>([2, 3, 5]);
        for (int i = 0; i < predicted.Length; i++)
            predicted[i] = 0.2f;

        // target: [batch=2, seq=3] — integer class indices
        var target = new Tensor<float>([2, 3]);
        target[0] = 0; target[1] = 1; target[2] = 4;
        target[3] = 2; target[4] = 3; target[5] = 0;

        var losses = new LossFunctionBase<float>[]
        {
            new CategoricalCrossEntropyLoss<float>(),
            new CrossEntropyLoss<float>(),
            new FocalLoss<float>(),
        };

        foreach (var loss in losses)
        {
            var result = loss.ComputeTapeLoss(predicted, target);
            Assert.True(result.Length > 0,
                $"{loss.GetType().Name} should produce a valid loss tensor");
            _output.WriteLine($"{loss.GetType().Name}: loss computed successfully");
        }
    }

    /// <summary>
    /// Verifies that one-hot encoded targets (matching shape) pass through
    /// EnsureTargetMatchesPredicted unchanged.
    /// </summary>
    [Fact]
    public void CategoricalCE_OneHotTargets_WorkUnchanged()
    {
        var predicted = new Tensor<float>([2, 4]);
        for (int i = 0; i < 8; i++) predicted[i] = 0.25f;

        var target = new Tensor<float>([2, 4]);
        target[0] = 1f; target[5] = 1f;

        var loss = new CategoricalCrossEntropyLoss<float>();
        var result = loss.ComputeTapeLoss(predicted, target);
        Assert.True(result.Length > 0);
    }

    /// <summary>
    /// Verifies that feature selection correctly skips Transformer models
    /// whose first layer is an EmbeddingLayer. Without #1113 fix, this
    /// throws ArgumentOutOfRangeException.
    /// </summary>
    [Fact]
    public void FeatureSelection_SkipsTransformerWithEmbedding()
    {
        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: 32,
            feedForwardDimension: 64,
            inputSize: 4,
            outputSize: 10,
            maxSequenceLength: 4,
            vocabularySize: 10);

        var transformer = new Transformer<float>(architecture);

        // Verify EmbeddingLayer is the first layer
        Assert.True(transformer.Layers.Count > 0,
            "Transformer should have layers after construction");
        Assert.IsType<EmbeddingLayer<float>>(transformer.Layers[0]);

        // Before #1113 fix: crashes with ArgumentOutOfRangeException
        transformer.SetActiveFeatureIndices(new[] { 0, 1, 2, 3 });
        _output.WriteLine("Feature selection correctly skipped for embedding model.");
    }

    /// <summary>
    /// Edge case: TokenClassification with a single class (binary per-position labeling).
    /// </summary>
    [Fact]
    public void TokenClassification_SingleClass_BuildsCorrectLayers()
    {
        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.TokenClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: 16,
            feedForwardDimension: 32,
            inputSize: 4,
            outputSize: 2,
            maxSequenceLength: 4,
            vocabularySize: 10);

        var transformer = new Transformer<float>(architecture);

        // No global pooling for token classification
        Assert.DoesNotContain(transformer.Layers,
            l => l is GlobalPoolingLayer<float>);

        // Has embedding layer
        Assert.IsType<EmbeddingLayer<float>>(transformer.Layers[0]);

        _output.WriteLine("TokenClassification with 2 classes builds correctly.");
    }

    /// <summary>
    /// Edge case: large number of classes for token classification.
    /// </summary>
    [Fact]
    public void TokenClassification_ManyClasses_BuildsCorrectLayers()
    {
        int numClasses = 50;
        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.TokenClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: 32,
            feedForwardDimension: 64,
            inputSize: 8,
            outputSize: numClasses,
            maxSequenceLength: 8,
            vocabularySize: 100);

        var transformer = new Transformer<float>(architecture);

        Assert.DoesNotContain(transformer.Layers,
            l => l is GlobalPoolingLayer<float>);
        Assert.True(transformer.Layers.Count > 3,
            "Transformer should have multiple layers");

        _output.WriteLine($"TokenClassification with {numClasses} classes builds correctly.");
    }

    /// <summary>
    /// Verifies that InferClassificationTaskType handles edge cases correctly.
    /// </summary>
    [Fact]
    public void InferClassificationTaskType_EdgeCases()
    {
        // 3D target shape [B, S, C] — already one-hot, but has seq dim → TokenClassification
        var result3d = TransformerArchitecture<float>.InferClassificationTaskType(
            [100, 8, 5], 8);
        Assert.Equal(NeuralNetworkTaskType.TokenClassification, result3d);

        // seqLen = 1 with [B, 1] → TokenClassification (single-token sequence)
        var singleToken = TransformerArchitecture<float>.InferClassificationTaskType(
            [100, 1], 1);
        Assert.Equal(NeuralNetworkTaskType.TokenClassification, singleToken);

        _output.WriteLine("InferClassificationTaskType edge cases handled correctly.");
    }
}
