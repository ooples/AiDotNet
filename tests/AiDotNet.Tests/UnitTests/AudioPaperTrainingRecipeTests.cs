using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;
using AiDotNet.SpeechRecognition.ConformerFamily;
using AiDotNet.SpeechRecognition.Streaming;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

public class AudioPaperTrainingRecipeTests
{
    [Fact]
    public void ConvTransformer_DefaultOptimizerMatchesConformerRecipe()
    {
        var options = CreateConvTransformerOptions();
        using var model = new ConvTransformer<double>(CreateArchitecture(), options);

        AssertTransformerAdamRecipe(model, options.EncoderDim, options.WarmupSteps, options.LearningRateFactor, options.WeightDecay);
    }

    [Fact]
    public void EfficientConformer_DefaultOptimizerMatchesAuthorsRecipe()
    {
        var options = CreateEfficientConformerOptions();
        using var model = new EfficientConformer<double>(CreateArchitecture(), options);

        AssertTransformerAdamRecipe(model, options.EncoderDim, options.WarmupSteps, options.LearningRateFactor, options.WeightDecay);
    }

    [Fact]
    public void KyutaiMoshi_DefaultOptimizerMatchesOfficialFineTuningRecipe()
    {
        var options = new KyutaiMoshiOptions
        {
            EncoderDim = 4,
            NumEncoderLayers = 1,
            NumAttentionHeads = 1,
            NumMels = 4,
            VocabSize = 4,
            DropoutRate = 0.0,
        };
        using var model = new KyutaiMoshi<double>(CreateArchitecture(), options);

        var field = typeof(KyutaiMoshi<double>).GetField("_optimizer", BindingFlags.Instance | BindingFlags.NonPublic);
        var optimizer = Assert.IsType<AdamWOptimizer<double, Tensor<double>, Tensor<double>>>(field?.GetValue(model));
        var optimizerOptions = Assert.IsType<AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
        var scheduler = Assert.IsType<OneCycleLRScheduler>(optimizerOptions.LearningRateScheduler);

        Assert.Equal(options.LearningRate, scheduler.MaxLearningRate, 15);
        Assert.Equal(options.TotalTrainingSteps, scheduler.TotalSteps);
        Assert.Equal(options.WarmupFraction, scheduler.PctStart, 15);
        Assert.Equal(0.9, optimizerOptions.Beta1, 12);
        Assert.Equal(0.95, optimizerOptions.Beta2, 12);
        Assert.Equal(1e-8, optimizerOptions.Epsilon, 15);
        Assert.Equal(options.WeightDecay, optimizerOptions.WeightDecay, 12);
        Assert.True(optimizerOptions.EnableGradientClipping);
        Assert.Equal(options.MaxGradientNorm, optimizerOptions.MaxGradientNorm, 12);
        Assert.False(optimizerOptions.UseAdaptiveLearningRate);
        Assert.False(optimizerOptions.UseAdaptiveBetas);
        Assert.False(optimizerOptions.UseAMSGrad);
        Assert.Equal(SchedulerStepMode.StepPerBatch, optimizerOptions.SchedulerStepMode);
        Assert.Equal(scheduler.CurrentLearningRate, optimizerOptions.InitialLearningRate, 15);
    }

    private static void AssertTransformerAdamRecipe(
        object model,
        int expectedModelDimension,
        int expectedWarmupSteps,
        double expectedFactor,
        double expectedWeightDecay)
    {
        var field = model.GetType().GetField("_optimizer", BindingFlags.Instance | BindingFlags.NonPublic);
        var optimizer = Assert.IsType<AdamOptimizer<double, Tensor<double>, Tensor<double>>>(field?.GetValue(model));
        var optimizerOptions = Assert.IsType<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
        var scheduler = Assert.IsType<NoamSchedule>(optimizerOptions.LearningRateScheduler);

        Assert.Equal(0.9, optimizerOptions.Beta1, 12);
        Assert.Equal(0.98, optimizerOptions.Beta2, 12);
        Assert.Equal(1e-9, optimizerOptions.Epsilon, 15);
        Assert.False(optimizerOptions.UseAdaptiveLearningRate);
        Assert.False(optimizerOptions.UseAdaptiveBetas);
        Assert.False(optimizerOptions.UseAMSGrad);
        Assert.False(optimizerOptions.EnableGradientClipping);
        Assert.Equal(SchedulerStepMode.StepPerBatch, optimizerOptions.SchedulerStepMode);
        Assert.Equal(expectedModelDimension, scheduler.ModelDimension);
        Assert.Equal(expectedWarmupSteps, scheduler.WarmupSteps);
        Assert.Equal(expectedFactor, scheduler.Factor, 12);

        var regularization = Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(optimizerOptions.Regularization);
        Assert.Equal(expectedWeightDecay, regularization.GetOptions().Strength, 12);
        Assert.Equal(scheduler.CurrentLearningRate, optimizerOptions.InitialLearningRate, 15);
    }

    private static ConvTransformerOptions CreateConvTransformerOptions() => new()
    {
        EncoderDim = 4,
        NumEncoderLayers = 1,
        NumAttentionHeads = 1,
        FeedForwardDim = 8,
        NumMels = 4,
        VocabSize = 4,
        DropoutRate = 0.0,
    };

    private static EfficientConformerOptions CreateEfficientConformerOptions() => new()
    {
        EncoderDim = 4,
        NumEncoderLayers = 1,
        NumAttentionHeads = 1,
        FeedForwardExpansionFactor = 2,
        NumMels = 4,
        VocabSize = 4,
        DropoutRate = 0.0,
    };

    private static NeuralNetworkArchitecture<double> CreateArchitecture() => new(
        inputType: InputType.OneDimensional,
        taskType: NeuralNetworkTaskType.Regression,
        inputSize: 4,
        outputSize: 4);
}
