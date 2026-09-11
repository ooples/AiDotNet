using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.NER.Options;
using AiDotNet.NER.TransformerBased;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NER;

/// <summary>Verifies the public warmup contract on a real model's optimizer and training path.</summary>
public sealed class TransformerNERWarmupTests
{
    public TransformerNERWarmupTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.001)]
    [InlineData(0.01)]
    public void ExplicitInitialLearningRate_IsUsedUnchanged(double initialRate)
    {
        using var model = CreateModel(CreateOptions(initialRate));
        var scheduler = Assert.IsType<LinearWarmupScheduler>(GetOptimizer(model).LearningRateScheduler);

        Assert.Equal(initialRate, scheduler.CurrentLearningRate);
        Assert.Equal(initialRate, scheduler.GetLearningRateAtStep(0));
        Assert.Equal(0.01, scheduler.GetLearningRateAtStep(4));
    }

    [Fact]
    public void DisabledWarmupWithoutDecay_DoesNotCreateAScheduler()
    {
        var options = CreateOptions(0.0);
        options.WarmupSteps = 0;
        using var model = CreateModel(options);

        Assert.Null(GetOptimizer(model).LearningRateScheduler);
    }

    [Fact]
    public void OptionsCopy_PreservesAnExplicitZeroStartingRate()
    {
        var options = new TransformerNEROptions(CreateOptions(0.0));
        using var model = CreateModel(options);
        var scheduler = Assert.IsType<LinearWarmupScheduler>(GetOptimizer(model).LearningRateScheduler);

        Assert.Equal(0.0, options.WarmupInitialLearningRate);
        Assert.Equal(0.0, scheduler.CurrentLearningRate);
    }

    [Fact]
    public void NegativeInitialLearningRate_RemainsInvalid()
    {
        var options = CreateOptions(0.0);
        Assert.Throws<ArgumentOutOfRangeException>(() => options.WarmupInitialLearningRate = -0.001);
    }

    [Fact]
    public void ZeroStartingRate_PreservesWeightsThenAdvancesToATrainableUpdate()
    {
        using var arena = TensorArena.Create();
        using var model = CreateModel(CreateOptions(0.0));
        var input = new Tensor<float>(new[] { 4, 8 });
        for (int i = 0; i < input.Length; i++) input[i] = (i % 7 - 3) * 0.1f;
        var target = new Tensor<float>(new[] { 4 });
        for (int i = 0; i < target.Length; i++) target[i] = i;
        model.Predict(input);
        float[] initialWeights = model.GetParameters().ToArray();
        Assert.NotEmpty(initialWeights);

        model.Train(input, target);
        Assert.Equal(initialWeights, model.GetParameters().ToArray());
        var scheduler = Assert.IsType<LinearWarmupScheduler>(GetOptimizer(model).LearningRateScheduler);
        Assert.Equal(1, scheduler.CurrentStep);
        Assert.Equal(0.0025, scheduler.CurrentLearningRate);

        model.Train(input, target);
        float[] trainedWeights = model.GetParameters().ToArray();
        Assert.Equal(initialWeights.Length, trainedWeights.Length);
        Assert.All(trainedWeights, value => Assert.False(float.IsNaN(value) || float.IsInfinity(value)));
        Assert.Contains(Enumerable.Range(0, initialWeights.Length), index => initialWeights[index] != trainedWeights[index]);
        Assert.Equal(2, scheduler.CurrentStep);
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

    private static GradientBasedOptimizerBase<float, Tensor<float>, Tensor<float>> GetOptimizer(TinyBERTNER<float> model)
    {
        // Inspect the instance created by the shared base, not a separately configured optimizer
        // that could honor the option while the model silently ignored it.
        FieldInfo? field = typeof(TransformerNERBase<float>).GetField("_optimizer",
            BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(field);
        return Assert.IsAssignableFrom<GradientBasedOptimizerBase<float, Tensor<float>, Tensor<float>>>(field.GetValue(model));
    }
}
