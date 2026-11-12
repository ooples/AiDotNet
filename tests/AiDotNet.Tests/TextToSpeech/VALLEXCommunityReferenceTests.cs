using System.Text.Json;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class VALLEXCommunityReferenceTests
{
    [Fact]
    public void CommunityPhoneTokenizer_AllVocabularyIdsAndBoundaryCasesMatchOriginalHuggingFace()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "vallex_phone_reference.json")));
        Assert.Equal(document.RootElement.GetProperty("vocabulary").EnumerateArray().Select(x => x.GetString()),
            VallEXCommunityPhoneTokenizer.Vocabulary);
        var options = VALLEXOptions.CommunityCheckpoint();
        options.HiddenDim = 16; options.NumHeads = 2;
        options.NumEncoderLayers = options.NumDecoderLayers = 1; options.FeedForwardDim = 32;
        options.NumCodebooks = 3; options.CodebookSize = 64;
        options.Codec = new EnCodecOptions { SampleRate = 24000, NumQuantizers = 3, CodebookSize = 64,
            Filters = 4, Ratios = [8, 5, 4, 2], Dimension = 8, TargetBandwidthKbps = 1.35 };
        using var model = new VALLEX<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1), options);
        foreach (var item in document.RootElement.GetProperty("cases").EnumerateArray())
            Assert.Equal(item.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()),
                model.EncodePhonemes(new[] { item.GetProperty("phones").GetString()! }));
        Assert.Equal(70, options.VocabSize);
        Assert.Equal(66, model.Core!.ArAudioEmbedding.GetParameters().Length / options.HiddenDim);
    }

    [Fact]
    public void CommunityCheckpointOptions_CopyPreservesAxesAndPaperDefaultsRemainDistinct()
    {
        var paper = new VALLEXOptions(); var released = VALLEXCloneOptions.CommunityCheckpoint();
        released.SynthesisLanguage = VallEXLanguage.Japanese;
        var copy = new VALLEXCloneOptions(released);
        Assert.Equal(2, paper.LanguageCount); Assert.False(paper.PrependBos);
        Assert.Equal(VallELanguagePlacement.AcousticTokens, paper.LanguagePlacement);
        Assert.False(paper.UseCommunityPhoneTokenizer);
        Assert.Equal(3, copy.LanguageCount); Assert.Equal(2048, copy.TextTokens);
        Assert.True(copy.PrependBos); Assert.True(copy.UseCommunityPhoneTokenizer);
        Assert.False(copy.StopOnMostLikelyEndToken);
        Assert.Equal(VallELanguagePlacement.TextTokens, copy.LanguagePlacement);
        Assert.Equal(VallEXLanguage.Japanese, copy.SynthesisLanguage);
    }
}
