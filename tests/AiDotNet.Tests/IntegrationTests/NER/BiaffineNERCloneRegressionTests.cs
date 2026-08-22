using AiDotNet.Enums;
using AiDotNet.NER.Options;
using AiDotNet.NER.SpanBased;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NER;

public class BiaffineNERCloneRegressionTests
{
    [Fact(Timeout = 120000)]
    public async Task Clone_AfterLazyMaterialization_PreservesParametersAndDecodedLabels()
    {
        await Task.Yield();
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 32,
            outputSize: 9);
        var options = new BiaffineNEROptions
        {
            HiddenDimension = 32,
            NumAttentionHeads = 4,
            NumTransformerLayers = 2,
            IntermediateDimension = 64,
            NumLabels = 9,
            MaxSequenceLength = 12,
            MaxSpanLength = 3,
            SpanEmbeddingDimension = 32,
            DropoutRate = 0.0,
            LearningRate = 1e-3,
            BiLstmHiddenSize = 7,
            BiLstmLayers = 1,
            BiLstmDropout = 0.0,
            EmbeddingsDropout = 0.0,
        };
        using var model = new BiaffineNER<double>(architecture, options);
        model.SetTrainingMode(false);

        var input = new Tensor<double>([8, 32]);
        var random = new Random(42);
        for (int i = 0; i < input.Length; i++) input[i] = random.NextDouble();

        var expectedLabels = model.Predict(input);
        var expectedParameters = model.GetParameters();
        var expectedChunks = model.GetParameterStateChunks().ToArray();
        using var clone = Assert.IsType<BiaffineNER<double>>(model.Clone());
        clone.SetTrainingMode(false);
        var actualParameters = clone.GetParameters();
        var actualChunks = clone.GetParameterStateChunks().ToArray();

        Assert.Equal(expectedParameters.Length, actualParameters.Length);
        Assert.Equal(expectedChunks.Length, actualChunks.Length);
        for (int chunkIndex = 0; chunkIndex < expectedChunks.Length; chunkIndex++)
        {
            var expectedChunk = expectedChunks[chunkIndex];
            var actualChunk = actualChunks[chunkIndex];
            Assert.Equal(expectedChunk.StableId, actualChunk.StableId);
            Assert.Equal(expectedChunk.Tensor.Shape, actualChunk.Tensor.Shape);
            for (int i = 0; i < expectedChunk.Tensor.Length; i++)
            {
                Assert.True(expectedChunk.Tensor[i] == actualChunk.Tensor[i],
                    $"Clone changed {expectedChunk.StableId}[{i}]: " +
                    $"expected {expectedChunk.Tensor[i]:R}, actual {actualChunk.Tensor[i]:R}.");
            }
        }

        var actualLabels = clone.Predict(input);
        Assert.Equal(expectedLabels.Shape, actualLabels.Shape);
        Assert.Equal(expectedLabels.ToArray(), actualLabels.ToArray());
    }
}
