using AiDotNet.Enums;
using AiDotNet.NER.Options;
using AiDotNet.NER.TransformerBased;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NER;

public sealed class TransformerNEROptionsContractTests
{
    [Theory]
    [InlineData(0.0)]
    [InlineData(0.001)]
    public void GetOptions_ReturnsTheConfigurationActuallyUsedByTheModel(double initialRate)
    {
        var options = CreateOptions(initialRate);
        using var model = CreateModel(options);

        var reported = Assert.IsType<TransformerNEROptions>(model.GetOptions());
        Assert.Same(options, reported);
        Assert.Equal(8, reported.HiddenDimension);
        Assert.Equal(4, reported.WarmupSteps);
        Assert.Equal(initialRate, reported.WarmupInitialLearningRate);
    }

    [Fact]
    public void GetOptions_CopyRetainsConfigurationWithoutSharingMutableNestedOptions()
    {
        var original = CreateOptions(0.0);
        using var model = CreateModel(original);
        var reported = Assert.IsType<TransformerNEROptions>(model.GetOptions());

        var copy = new TransformerNEROptions(reported);
        Assert.Equal(original.LearningRate, copy.LearningRate);
        Assert.Equal(original.WarmupSteps, copy.WarmupSteps);
        Assert.Equal(0.0, copy.WarmupInitialLearningRate);
        Assert.Equal(original.LabelNames, copy.LabelNames);
        Assert.NotSame(original.LabelNames, copy.LabelNames);
        Assert.NotSame(original.OnnxOptions, copy.OnnxOptions);
        copy.LabelNames[0] = "COPY-ONLY";
        Assert.NotEqual(copy.LabelNames[0], original.LabelNames[0]);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.001)]
    public void Clone_ReportsAnIndependentCopyOfTheActualConfiguration(double initialRate)
    {
        using var arena = TensorArena.Create();
        var options = CreateOptions(initialRate);
        using var model = CreateModel(options);
        var input = new Tensor<float>(new[] { 4, 8 });
        for (int i = 0; i < input.Length; i++) input[i] = (i % 7 - 3) * 0.1f;
        var target = new Tensor<float>(new[] { 4 });
        for (int i = 0; i < target.Length; i++) target[i] = i;
        model.Train(input, target);
        model.Train(input, target);

        using var clone = Assert.IsType<TinyBERTNER<float>>(model.Clone());
        var originalOptions = Assert.IsType<TransformerNEROptions>(model.GetOptions());
        var clonedOptions = Assert.IsType<TransformerNEROptions>(clone.GetOptions());
        Assert.NotSame(originalOptions, clonedOptions);
        Assert.Equal(originalOptions.HiddenDimension, clonedOptions.HiddenDimension);
        Assert.Equal(originalOptions.LearningRate, clonedOptions.LearningRate);
        Assert.Equal(originalOptions.WarmupSteps, clonedOptions.WarmupSteps);
        Assert.Equal(initialRate, clonedOptions.WarmupInitialLearningRate);
        Assert.Equal(originalOptions.LabelNames, clonedOptions.LabelNames);
        Assert.NotSame(originalOptions.LabelNames, clonedOptions.LabelNames);
        Assert.NotSame(originalOptions.OnnxOptions, clonedOptions.OnnxOptions);
        Assert.Equal(model.GetParameters().ToArray(), clone.GetParameters().ToArray());
        clonedOptions.LabelNames[0] = "CLONE-ONLY";
        Assert.NotEqual(clonedOptions.LabelNames[0], originalOptions.LabelNames[0]);
    }

    private static TransformerNEROptions CreateOptions(double initialRate) => new()
    {
        HiddenDimension = 8,
        NumAttentionHeads = 2,
        NumTransformerLayers = 1,
        IntermediateDimension = 16,
        NumLabels = 9,
        MaxSequenceLength = 4,
        DropoutRate = 0,
        LearningRate = 0.01,
        WarmupSteps = 4,
        WarmupInitialLearningRate = initialRate
    };

    private static TinyBERTNER<float> CreateModel(TransformerNEROptions options) => new(
        new NeuralNetworkArchitecture<float>(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.SequenceToSequence, inputSize: 8, outputSize: 9)
        { RandomSeed = 1337 }, options);
}
