using AiDotNet.Enums;
using AiDotNet.Finance.NLP;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// FinBERT's training forward runs in training mode: before the fix it went through the inference path, which switched
/// every layer to evaluation mode and turned dropout off for the whole of training.
/// </summary>
public sealed class FinBERTTrainingModeTests
{
    [Fact]
    public void ForwardForTraining_StaysInTrainingMode()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 8, outputSize: 3);
        var model = new FinBERT<double>(architecture, new FinBERTOptions<double>
        {
            MaxSequenceLength = 8, MaxPositionEmbeddings = 8, VocabularySize = 32, HiddenDimension = 8,
            NumAttentionHeads = 2, IntermediateDimension = 16, NumLayers = 1, NumSentimentClasses = 3,
        });
        model.SetTrainingMode(true);
        var tokens = new Tensor<double>(new[] { 1, 8 });
        for (int i = 0; i < tokens.Length; i++) tokens[i] = i % 32;

        model.ForwardForTraining(tokens);

        Assert.True(model.IsTrainingMode, "ForwardForTraining switched the model to evaluation mode, which turns dropout off.");
    }
}
