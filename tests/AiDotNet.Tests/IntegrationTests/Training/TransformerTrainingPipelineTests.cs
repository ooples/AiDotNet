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
/// </summary>
public class TransformerTrainingPipelineTests
{
    private readonly ITestOutputHelper _output;

    public TransformerTrainingPipelineTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// The full happy path: build a small Transformer through the facade,
    /// train on synthetic token sequences, and verify it completes without
    /// crashing. Before fixes #1113/#1114/#1115, this would throw at
    /// different points depending on which bug was hit first.
    /// </summary>
    [Fact]
    public async Task TransformerTraining_ThroughFacade_CompletesWithoutCrash()
    {
        // Arrange: small Transformer for sequence classification
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

        // Synthetic data: random token sequences with integer class targets
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

        // Act: train through the facade — this exercises all 3 bugs
        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureDataLoader(
                new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(inputData, targetData))
            .ConfigureModel(transformer);

        var result = await builder.BuildAsync();

        // Assert: training completed and produced a valid result
        Assert.NotNull(result);
        Assert.NotNull(result.Model);
        _output.WriteLine("Transformer training through AiModelBuilder facade completed.");
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
}
