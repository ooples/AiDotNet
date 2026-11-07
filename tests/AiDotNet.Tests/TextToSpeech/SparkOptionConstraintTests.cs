using System;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class SparkOptionConstraintTests
{
    [Theory]
    [InlineData(1, 0.75, 0)]
    [InlineData(0, 0.5, 0.5)]
    [InlineData(0, double.NaN, 1)]
    [InlineData(0, 0.5, 1.1)]
    public void InvalidDiscriminatorBandsAreRejectedBeforeTraining(double first, double middle, double last)
    {
        var options = SparkTTSPaperTests.SmallOptions();
        options.DiscriminatorBands = new[] { first, middle, last };
        Assert.Throws<ArgumentException>(() => new SparkTTS<double>(new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 320), options));
    }

    [Fact]
    public void ValidAlternativeBandsRunTheActualBiCodecTrainingObjective()
    {
        var options = SparkTTSPaperTests.SmallOptions(SparkTTSTrainingComponent.BiCodec);
        options.DiscriminatorBands = new[] { 0.0, 0.25, 0.5, 1.0 };
        using var model = new SparkTTS<double>(new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 320), options);
        double loss = model.Train(new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 3.0, 4.0 })),
            Audio = SparkNativeFixture.Recording<double>(),
        });
        Assert.False(double.IsNaN(loss) || double.IsInfinity(loss));
    }
}
