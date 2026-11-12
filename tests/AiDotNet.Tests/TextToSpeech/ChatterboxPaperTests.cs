using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Checks the released Chatterbox topology and its preprocessing/conditioning contracts.</summary>
public class ChatterboxPaperTests
{
    [Fact(Timeout = 60000)]
    public async Task Defaults_MatchTheReleasedEnglishModel()
    {
        await Task.Yield();
        var o = new ChatterboxOptions();

        Assert.Equal((24000, 80, 480, 1920), (o.SampleRate, o.MelChannels, o.HopSize, o.FftSize));
        Assert.Equal((704, 255, 0, 2048), (o.VocabSize, o.TextStartToken, o.TextStopToken, o.MaxTextLength));
        Assert.Equal((6561, 6561, 6562, 8194, 4096),
            (o.CodebookSize, o.SpeechStartToken, o.SpeechStopToken, o.SpeechVocabularySize, o.MaxSpeechTokens));
        Assert.Equal((1024, 4096, 30, 16, 16),
            (o.LLMDim, o.LmFeedForwardDim, o.NumLLMLayers, o.NumHeads, o.KeyValueHeads));
        Assert.Equal((500000.0, 8.0, 1.0, 4.0, 8192),
            (o.RopeTheta, o.RopeScalingFactor, o.RopeLowFrequencyFactor, o.RopeHighFrequencyFactor,
                o.RopeOriginalMaxPositions));
        Assert.Equal((32, 4, 0.2, 256, 150),
            (o.PerceiverQueries, o.PerceiverHeads, o.PerceiverDropout, o.T3SpeakerEmbeddingDim,
                o.SpeechConditionPromptTokens));
        Assert.Equal((512, 8, 2048, 10, 2, 3),
            (o.FlowDim, o.FlowHeads, o.FlowFeedForwardDim, o.FlowTimesteps, o.TokenMelRatio, o.Lookahead));
        Assert.Equal(new[] { 8, 5, 3 }, o.HiftUpsampleRates);
        Assert.Equal(new[] { 16, 11, 7 }, o.HiftUpsampleKernels);
        Assert.Equal(new[] { 7, 7, 11 }, o.HiftSourceKernels);
    }

    [Fact(Timeout = 60000)]
    public async Task NativeModel_IsCustomT3PlusTheReleasedFrozenFrontEnds()
    {
        await Task.Yield();
        using var model = ChatterboxFixture.Create<double>();
        var t3 = Assert.IsType<ChatterboxT3<double>>(model.T3Model);
        var cfg = t3.TransformerConfiguration;

        Assert.Equal(0, cfg.VocabSize); // T3 supplies independent text and speech embeddings around a bare Llama.
        Assert.Equal((16, 1, 2, 2, 32), (cfg.Hidden, cfg.Layers, cfg.Heads, cfg.KeyValueHeads, cfg.Intermediate));
        Assert.Equal(32, t3.TextEmbedding.VocabularySize);
        Assert.Equal(6563, t3.SpeechEmbedding.VocabularySize);
        Assert.Equal(2, t3.Perceiver.Query.VocabularySize);
        Assert.Single(model.VoiceEncoderLayers.OfType<LSTMCellLayer<double>>());
        Assert.NotNull(model.VoiceEncoderModel);
        Assert.Equal(TtsSupervision.ReferenceRecording, model.SynthesisVoiceRequirement);
    }

    [Fact(Timeout = 60000)]
    public async Task PunctuationAndBpe_MatchTheReleasePipeline()
    {
        await Task.Yield();
        Assert.Equal("You need to add some text for me to talk.", Chatterbox<double>.NormalizeText(string.Empty));
        // The release applies the spaced-hyphen rule before replacing the em dash, so this remains a hyphen.
        Assert.Equal("Hello there - \"friend\".", Chatterbox<double>.NormalizeText("hello   there — “friend”"));

        using var model = ChatterboxFixture.Create<double>();
        int[] ids = model.Tokenize("hello world.");
        Assert.Equal(new[] { 10, 2, 11, 6, 12, 5, 13, 14 }, ids);
        Assert.Equal("hello world.", model.Detokenize(ids));
    }

    [Fact(Timeout = 120000)]
    public async Task VoiceConditioning_KeepsTheFullVoiceButBoundsS3GenReference()
    {
        await Task.Yield();
        using var model = ChatterboxFixture.Create<double>();
        var voice = model.CreateVoice(ChatterboxFixture.Recording<double>(2), 24000);

        Assert.Equal(48000, voice.ReferenceAudio!.Length); // The voice encoder receives the complete recording.
        Assert.NotNull(voice.Reference);
        Assert.NotNull(voice.ReferenceCodecTokens);
        Assert.InRange(voice.ReferenceCodecTokens!.Shape[0], 1, 30); // S3Gen sees only DecoderReferenceSeconds (one here).
        Assert.Equal(8, voice.SpeakerEmbedding!.Length);             // CAM++ conditioning for the flow.
    }
}
