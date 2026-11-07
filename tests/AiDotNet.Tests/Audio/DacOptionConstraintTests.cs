using AiDotNet.Audio.Effects;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.Audio;

public class DacOptionConstraintTests
{
    [Fact]
    public void MonoModelStillRunsTheActualEncoderQuantizerAndDecoder()
    {
        using var model = new DAC<double>(new NeuralNetworkArchitecture<double>(inputFeatures: 1, outputSize: 1),
            new DACOptions
            {
                EncoderDim = 4, DecoderDim = 16, LatentDim = 8,
                EncoderRates = new[] { 2, 2 }, DecoderRates = new[] { 2, 2 },
                CodebookDim = 2, NumQuantizers = 2, CodebookSize = 8,
                MelWindows = new[] { 16 }, MelBins = new[] { 4 },
                DiscriminatorChannels = 4, DiscriminatorWidthDivisor = 32
            });
        model.SetTrainingMode(false);
        var audio = new Tensor<double>(new[] { 256 });
        for (int i = 0; i < audio.Length; i++) audio[i] = 0.4 * Math.Sin(i * 0.13);
        var reconstructed = model.Predict(audio);
        Assert.Equal(audio.Shape, reconstructed.Shape);
        Assert.All(reconstructed, value => Assert.True(double.IsFinite(value)));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    [InlineData(-1)]
    public void NativeModelRejectsUnsupportedWaveformChannelCountsBeforeBuildingLayers(int channels)
    {
        var error = Assert.Throws<ArgumentException>(() => new DAC<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: 1, outputSize: 1),
            new DACOptions { Channels = channels }));
        Assert.Equal("Channels", error.ParamName);
    }
}
