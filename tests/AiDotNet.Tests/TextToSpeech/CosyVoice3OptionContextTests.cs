using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class CosyVoice3OptionContextTests
{
    [Fact]
    public void DitOptionsExcludeUnusedConformerAndNativeS3Controls()
    {
        foreach (string option in new[] { "FlowDim", "FlowHeads", "FlowFeedForwardDim", "FlowEncoderLayers",
            "FlowAttentionDropoutRate", "DecoderChannels", "DecoderHeads", "DecoderHeadDim", "DecoderBlocks",
            "DecoderMidBlocks", "FlowTimesteps", "TrainingCfgRate", "InferenceCfgRate", "FlowLoss",
            "SpeechTokenizerState", "SpeechTokenizerHeads", "SpeechTokenizerLayers" })
        {
            Assert.Null(typeof(CosyVoice3Options).GetProperty(option));
            Assert.NotNull(typeof(CosyVoiceOptions).GetProperty(option));
            Assert.NotNull(typeof(CosyVoice2Options).GetProperty(option));
            Assert.NotNull(typeof(AiDotNet.TextToSpeech.VoiceCloning.ChatterboxOptions).GetProperty(option));
        }
    }

    [Theory]
    [InlineData(16, 2, 8)]
    [InlineData(32, 2, 16)]
    [InlineData(32, 4, 8)]
    public void PairedDitGeometryRunsNativeTrainingAndInference(int width, int heads, int headWidth)
    {
        var options = new CosyVoice3Options
        {
            CodebookSize = 8, MelChannels = 4, SpeakerEmbeddingDim = 4,
            DiTInputDim = 4, PreLookaheadChannels = 8, DiTDim = width, DiTDepth = 1,
            DiTHeads = heads, DiTHeadDim = headWidth, DiTFeedForwardMultiplier = 2,
            DiTTimesteps = 2, StaticChunkSize = 2, DropoutRate = 0,
        };
        var layers = new List<LayerBase<double>>();
        var flow = new CosyVoice3Flow<double>(AiDotNetEngine.Current, layers, options, new Random(3));
        var speaker = new Tensor<double>(new[] { 4 }); speaker[0] = 1;
        var mel = new Tensor<double>(new[] { 1, 4, 6 });
        for (int i = 0; i < mel.Length; i++) mel[i] = 0.2 * Math.Sin(i);
        var loss = flow.TrainingLoss(new[] { 1, 2, 3 }, mel, speaker, training: true, new Random(4));
        Assert.True(TtsPaperNumerics.IsFinite(loss[0]));
        var generated = flow.Inference(new[] { 1, 2, 3 }, Array.Empty<int>(), null, speaker,
            new Tensor<double>(new[] { 1, 4, 6 }));
        Assert.Equal(new[] { 1, 4, 6 }, generated.Shape.ToArray());
        Assert.All(generated, value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Theory]
    [InlineData("DiTDim")]
    [InlineData("DiTHeads")]
    [InlineData("DiTHeadDim")]
    [InlineData("TokenMelRatio")]
    public void UnpairedReleasedFlowGeometryIsRejected(string option)
    {
        var options = new CosyVoice3Options();
        switch (option)
        {
            case "DiTDim": options.DiTDim++; break;
            case "DiTHeads": options.DiTHeads++; break;
            case "DiTHeadDim": options.DiTHeadDim++; break;
            case "TokenMelRatio": options.TokenMelRatio = 1; break;
            default: throw new ArgumentOutOfRangeException(nameof(option));
        }
        Assert.Throws<ArgumentException>(() => new CosyVoice3<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: 1, outputSize: 1), options));
    }
}
