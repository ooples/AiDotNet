using System.Reflection;
using AiDotNet.Audio.TextToSpeech;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;
using AiDotNet.SpeechRecognition.ConformerFamily;
using AiDotNet.SpeechRecognition.Specialized;
using AiDotNet.SpeechRecognition.Streaming;
using AiDotNet.TextToSpeech.Classic;
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

        AssertTransformerAdamRecipe(model, options.EncoderDim * 2, options.WarmupSteps, options.LearningRateFactor, options.WeightDecay);
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

    [Fact]
    public void KeywordSpotting_DefaultNetworkAndOptimizerMatchDeepKwsPaper()
    {
        var architecture = CreateArchitecture();
        var options = new KeywordSpottingOptions();
        var layers = LayerHelper<double>.CreateDefaultKeywordSpottingLayers(
            architecture,
            options.NumEncoderLayers,
            options.EncoderDim,
            architecture.OutputSize).ToArray();

        Assert.Equal(3, options.NumEncoderLayers);
        Assert.Equal(128, options.EncoderDim);
        Assert.Equal(40, options.NumMels);
        Assert.Equal(4, layers.Length);
        foreach (var hidden in layers.Take(3))
        {
            var dense = Assert.IsType<DenseLayer<double>>(hidden);
            Assert.IsType<ReLUActivation<double>>(dense.ScalarActivation);
        }
        var output = Assert.IsType<DenseLayer<double>>(layers[^1]);
        Assert.IsType<SoftmaxActivation<double>>(output.ScalarActivation);

        using var model = new KeywordSpotting<double>(architecture, options);
        var field = typeof(KeywordSpotting<double>).GetField("_optimizer", BindingFlags.Instance | BindingFlags.NonPublic);
        var optimizer = Assert.IsType<StochasticGradientDescentOptimizer<double, Tensor<double>, Tensor<double>>>(field?.GetValue(model));
        var optimizerOptions = Assert.IsType<StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
        var scheduler = Assert.IsType<ExponentialLRScheduler>(optimizerOptions.LearningRateScheduler);

        Assert.Equal(options.LearningRate, optimizerOptions.InitialLearningRate, 15);
        Assert.Equal(options.LearningRate, scheduler.BaseLearningRate, 15);
        Assert.Equal(options.LearningRateDecay, scheduler.Gamma, 15);
        Assert.False(optimizerOptions.UseAdaptiveLearningRate);
        Assert.Equal(SchedulerStepMode.StepPerBatch, optimizerOptions.SchedulerStepMode);
    }

    [Fact]
    public void KeywordSpotting_PosteriorHandlingImplementsPaperEquations()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 2,
            inputWidth: 3,
            inputDepth: 1,
            outputSize: 3);
        var options = new KeywordSpottingOptions
        {
            Vocabulary = new[] { "filler", "okay", "google" },
        };
        using var model = new KeywordSpotting<double>(architecture, options);
        var posteriors = new Tensor<double>(new[] { 2, 3 });
        posteriors[0] = 0.1;
        posteriors[1] = 0.8;
        posteriors[2] = 0.1;
        posteriors[3] = 0.0;
        posteriors[4] = 0.0;
        posteriors[5] = 1.0;

        var decode = typeof(KeywordSpotting<double>).GetMethod(
            "DecodeKeywordPosteriors",
            BindingFlags.Instance | BindingFlags.NonPublic);
        var decoded = Assert.IsType<ValueTuple<List<int>, double>>(decode?.Invoke(model, new object[] { posteriors }));

        Assert.Equal(new[] { 1, 2 }, decoded.Item1);
        Assert.Equal(Math.Sqrt(0.8 * 0.55), decoded.Item2, 12);
    }

    [Fact]
    public void NoamHoldAnnealing_FollowsSqueezeformerEquation()
    {
        var scheduler = new NoamHoldAnnealingScheduler(
            peakLearningRate: 2e-3,
            warmupSteps: 20,
            holdSteps: 160,
            decayRate: 1.0);

        Assert.Equal(1e-4, scheduler.GetLearningRateAtStep(0), 15);
        Assert.Equal(2e-3, scheduler.GetLearningRateAtStep(19), 15);
        Assert.Equal(2e-3, scheduler.GetLearningRateAtStep(178), 15);
        Assert.Equal(2e-3, scheduler.GetLearningRateAtStep(179), 15);
        Assert.Equal(1e-3, scheduler.GetLearningRateAtStep(199), 15);
    }

    [Fact]
    public void Squeezeformer_DefaultOptimizerMatchesPaperSchedule()
    {
        var options = new SqueezeformerOptions
        {
            EncoderDim = 4,
            NumEncoderLayers = 1,
            NumAttentionHeads = 1,
            NumMels = 4,
            VocabSize = 4,
            DropoutRate = 0.0,
        };
        using var model = new Squeezeformer<double>(CreateArchitecture(), options);
        var field = typeof(Squeezeformer<double>).GetField("_optimizer", BindingFlags.Instance | BindingFlags.NonPublic);
        var optimizer = Assert.IsType<AdamWOptimizer<double, Tensor<double>, Tensor<double>>>(field?.GetValue(model));
        var optimizerOptions = Assert.IsType<AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
        var scheduler = Assert.IsType<NoamHoldAnnealingScheduler>(optimizerOptions.LearningRateScheduler);

        Assert.Equal(options.PeakLearningRate, scheduler.PeakLearningRate, 15);
        Assert.Equal(options.WarmupSteps, scheduler.WarmupSteps);
        Assert.Equal(options.PeakHoldSteps, scheduler.HoldSteps);
        Assert.Equal(options.LearningRateDecayPower, scheduler.DecayRate, 15);
        Assert.Equal(scheduler.CurrentLearningRate, optimizerOptions.InitialLearningRate, 15);
        Assert.Equal(0.9, optimizerOptions.Beta1, 12);
        Assert.Equal(0.999, optimizerOptions.Beta2, 12);
        Assert.Equal(1e-8, optimizerOptions.Epsilon, 15);
        Assert.Equal(options.WeightDecay, optimizerOptions.WeightDecay, 15);
        Assert.False(optimizerOptions.UseAdaptiveLearningRate);
        Assert.False(optimizerOptions.UseAdaptiveBetas);
        Assert.False(optimizerOptions.UseAMSGrad);
        Assert.Equal(SchedulerStepMode.StepPerBatch, optimizerOptions.SchedulerStepMode);
        Assert.True(optimizerOptions.EnableGradientClipping);
        Assert.Equal(1.0, optimizerOptions.MaxGradientNorm, 15);
    }

    [Fact]
    public void Tacotron2_DefaultOptimizerMatchesPublishedAdamRecipe()
    {
        var options = new Tacotron2Options
        {
            EncoderDim = 4,
            DecoderDim = 4,
            HiddenDim = 4,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            NumHeads = 1,
            DropoutRate = 0.0,
            VocabSize = 16,
            PrenetDim = 4,
            AttentionRnnDim = 4,
            DecoderRnnDim = 4,
            AttentionDimension = 2,
            AttentionLocationChannels = 2,
            PostnetDim = 4,
            PostnetLayers = 1,
            OutputsPerStep = 1,
        };
        using var model = new Tacotron2<double>(CreateArchitecture(), options);
        var field = typeof(Tacotron2Model<double>).GetField("_optimizer", BindingFlags.Instance | BindingFlags.NonPublic);
        var optimizer = Assert.IsType<AdamOptimizer<double, Tensor<double>, Tensor<double>>>(field?.GetValue(model));
        var optimizerOptions = Assert.IsType<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());

        Assert.Equal(1e-3, optimizerOptions.InitialLearningRate, 15);
        Assert.Equal(0.9, optimizerOptions.Beta1, 12);
        Assert.Equal(0.999, optimizerOptions.Beta2, 12);
        Assert.Equal(1e-6, optimizerOptions.Epsilon, 15);
        Assert.False(optimizerOptions.UseAdaptiveLearningRate);
        Assert.False(optimizerOptions.UseAdaptiveMomentum);
        Assert.False(optimizerOptions.UseAdaptiveBetas);
        Assert.False(optimizerOptions.UseAMSGrad);
        Assert.False(optimizerOptions.EnableGradientClipping);
        Assert.Null(optimizerOptions.LearningRateScheduler);
        var regularization = Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(
            optimizerOptions.Regularization);
        Assert.Equal(1e-6, regularization.GetOptions().Strength, 15);

        var copy = new Tacotron2Options(options);
        Assert.Equal(options.LearningRate, copy.LearningRate, 15);
        Assert.Equal(options.WeightDecay, copy.WeightDecay, 15);
        Assert.Equal(options.EncoderDim, copy.EncoderDim);
        Assert.Equal(options.NumEncoderLayers, copy.NumEncoderLayers);
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
