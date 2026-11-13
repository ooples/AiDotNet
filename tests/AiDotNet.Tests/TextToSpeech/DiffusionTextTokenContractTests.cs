using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class DiffusionTextTokenContractTests
{
    [Fact]
    public void PublicTextToTokens_ReturnsTheActualBertAndSentencePieceIds()
    {
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        using var ditto = new DiTToTTS<double>(architecture, DiTToTTSPaperTests.TinyOptions());
        using var e3 = new E3TTS<double>(architecture, E3TTSPaperTests.TinyOptions());
        const string text = "Hello, research papers!";
        Assert.Equal(ditto.TokenizeText(text).Select(x => (double)x), ditto.TextToTokens(text).ToArray());
        Assert.Equal(e3.TokenizeText(text).Select(x => (double)x), e3.TextToTokens(text).ToArray());
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(30522)]
    [InlineData(.5)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void EmbeddingIds_AreRejectedRatherThanRoundedOrClampedToAnotherToken(double value)
    {
        var tokens = new Tensor<double>(new[] { 1 }); tokens[0] = value;
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        using var ditto = new DiTToTTS<double>(architecture, DiTToTTSPaperTests.TinyOptions());
        using var e3 = new E3TTS<double>(architecture, E3TTSPaperTests.TinyOptions());
        Assert.ThrowsAny<ArgumentException>(() => ditto.Predict(tokens));
        Assert.ThrowsAny<ArgumentException>(() => e3.Predict(tokens));
        Assert.Throws<ArgumentException>(() => e3.Train(new TtsTrainingSample<double>
            { Tokens = tokens, Audio = new Tensor<double>(new[] { 4096 }) }));
    }

    // A continuous front end reads each character code / 128. U+FFFD (65533) is past Half's largest finite value
    // (65504), so casting the code before scaling gives +inf; the scaled code must be finite and rounded once.
    [Fact]
    public void ScaledCharacterCodes_StayFiniteInHalf()
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        var half = TtsModelBase<Half>.ScaledCharacterCodes(engine, "�A￿", 3).ToArray();
        Assert.Equal(ops.ToDouble(ops.FromDouble(65533 / 128.0)), ops.ToDouble(half[0]));
        Assert.Equal(65 / 128.0, ops.ToDouble(half[1]));
        Assert.Equal(ops.ToDouble(ops.FromDouble(65535 / 128.0)), ops.ToDouble(half[2]));
        var full = TtsModelBase<double>.ScaledCharacterCodes(engine, "�A￿", 3).ToArray();
        Assert.Equal(new[] { 65533 / 128.0, 65 / 128.0, 65535 / 128.0 }, full);
    }
}
