using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class UniAudioOptimizerOptionsTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EitherVocabularyNameConfiguresTheSameNativeVocabulary(bool useJointName)
    {
        using var original = UniAudioNativeFixture.Create<double>();
        var options = new UniAudioOptions((UniAudioOptions)original.GetOptions());
        if (useJointName) options.JointVocabularySize = 193;
        else options.VocabSize = 193;
        Assert.Equal(193, options.JointVocabularySize);
        Assert.Equal(193, options.VocabSize);
        using var model = new UniAudio<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 64) { RandomSeed = 2294 }, options);
        Assert.NotEqual(original.GetParameters().Length, model.GetParameters().Length);
    }

    [Theory]
    [InlineData(0.0001, 0.001)]
    [InlineData(0.003, 0.02)]
    [InlineData(0.0002, 0.0)]
    public void ConfiguredRatesReachThePaperOptimizer(double learningRate, double weightDecay)
    {
        using var template = UniAudioNativeFixture.Create<double>();
        var options = new UniAudioOptions((UniAudioOptions)template.GetOptions())
        {
            LearningRate = learningRate,
            WeightDecay = weightDecay,
        };
        using var model = new UniAudio<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 64) { RandomSeed = 2294 }, options);
        var optimizer = Assert.IsAssignableFrom<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>(
            typeof(UniAudio<double>).GetField("_optimizer", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model));
        var configured = Assert.IsType<AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
        Assert.Equal(learningRate, configured.InitialLearningRate);
        Assert.Equal(weightDecay, configured.WeightDecay);
        Assert.Equal(0.9, configured.Beta1);
        Assert.Equal(0.95, configured.Beta2);
        Assert.NotNull(configured.LearningRateScheduler);
    }
}
