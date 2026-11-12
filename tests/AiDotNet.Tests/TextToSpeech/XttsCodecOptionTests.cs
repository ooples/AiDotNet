using AiDotNet.TextToSpeech.VoiceCloning;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class XttsCodecOptionTests
{
    [Theory]
    [InlineData("NumEncoderLayers", 1)]
    [InlineData("NumEncoderLayers", -1)]
    [InlineData("NumCodebooks", 2)]
    [InlineData("CodebookSize", 17)]
    [InlineData("AudioVocabularySize", 19)]
    [InlineData("StartAudioToken", 17)]
    [InlineData("StopAudioToken", 16)]
    [InlineData("PerceiverHeadDim", 32)]
    [InlineData("PerceiverHeadDim", 128)]
    [InlineData("DvaeLayers", 1)]
    [InlineData("DvaeLayers", 3)]
    [InlineData("TextVocabularySize", 33)]
    [InlineData("VocabSize", 33)]
    public void NativeGraphRejectsUnpairedOrUnsupportedOptions(string option, int value)
    {
        foreach (bool clone in new[] { false, true })
        {
            void Prepare(XTTSv2Options options) => typeof(XTTSv2Options).GetProperty(option)!.SetValue(options, value);
            XTTSv2<double> Create() => clone ? XTTSv2Fixture.CreateClone<double>(Prepare) : XTTSv2Fixture.Create<double>(Prepare);
            if (option is "VocabSize" or "TextVocabularySize") Assert.Throws<InvalidDataException>(() => Create());
            else Assert.Throws<ArgumentException>(() => Create());
        }
    }

    [Theory]
    [InlineData(false, 8)]
    [InlineData(false, 16)]
    [InlineData(false, 32)]
    [InlineData(true, 8)]
    [InlineData(true, 16)]
    [InlineData(true, 32)]
    public void PairedAudioVocabularyEncodesAndReconstructsTheNativeSingleStream(bool clone, int codes)
    {
        void Prepare(XTTSv2Options options)
        {
            options.CodebookSize = codes;
            options.AudioVocabularySize = codes + 2;
            options.StartAudioToken = codes;
            options.StopAudioToken = codes + 1;
        }
        using XTTSv2<double> model = clone ? XTTSv2Fixture.CreateClone<double>(Prepare) : XTTSv2Fixture.Create<double>(Prepare);
        model.SetTrainingMode(false);
        var audio = new Tensor<double>(new[] { 4096 });
        for (int i = 0; i < audio.Length; i++) audio[i] = .3 * Math.Sin(.07 * i);
        var tokens = model.EncodeToTokens(audio);
        Assert.NotEmpty(tokens.ToArray());
        Assert.All(tokens.ToArray(), value => Assert.InRange(value, 0, codes - 1));
        Assert.Equal(1, model.CodecTokenCodebooks);
        Assert.Equal(codes, model.CodecTokenVocabulary);
        Assert.Equal(4, model.DvaeDownsampleFactor);
        var mel = model.DecodeTokensToMel(tokens);
        Assert.NotEmpty(mel.ToArray());
        Assert.All(mel.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LoadedTokenizerOwnsTextSpecialIds(bool clone)
    {
        using XTTSv2<double> original = clone ? XTTSv2Fixture.CreateClone<double>() : XTTSv2Fixture.Create<double>();
        var asset = JObject.Parse(File.ReadAllText(((XTTSv2Options)original.GetOptions()).TokenizerPath!));
        foreach (var token in asset["added_tokens"]!)
        {
            if ((string?)token["content"] == "[START]") token["id"] = 1;
            if ((string?)token["content"] == "[STOP]") token["id"] = 0;
        }
        WithTokenizer(asset, path =>
        {
            void Prepare(XTTSv2Options options) { options.TokenizerPath = path; options.StartTextToken = 7; options.StopTextToken = 8; }
            using XTTSv2<double> model = clone ? XTTSv2Fixture.CreateClone<double>(Prepare) : XTTSv2Fixture.Create<double>(Prepare);
            var options = (XTTSv2Options)model.GetOptions();
            Assert.Equal(1, options.StartTextToken);
            Assert.Equal(0, options.StopTextToken);
            Assert.NotEmpty(model.TokenizeText("hi", "en"));
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PairedTokenizerVocabularyUsesTheAdditionalEmbeddingRow(bool clone)
    {
        using XTTSv2<double> original = clone ? XTTSv2Fixture.CreateClone<double>() : XTTSv2Fixture.Create<double>();
        var asset = JObject.Parse(File.ReadAllText(((XTTSv2Options)original.GetOptions()).TokenizerPath!));
        asset["model"]!["vocab"]!["aa"] = 32;
        ((JArray)asset["model"]!["merges"]!).Add("a a");
        WithTokenizer(asset, path =>
        {
            void Prepare(XTTSv2Options options) { options.TokenizerPath = path; options.TextVocabularySize = 33; }
            using XTTSv2<double> model = clone ? XTTSv2Fixture.CreateClone<double>(Prepare) : XTTSv2Fixture.Create<double>(Prepare);
            Assert.Contains(32, model.TokenizeText("aa", "en"));
        });
    }

    private static void WithTokenizer(JObject asset, Action<string> test)
    {
        string path = Path.Combine(Path.GetTempPath(), "aidotnet-xtts-option-" + Guid.NewGuid().ToString("N") + ".json");
        try { File.WriteAllText(path, asset.ToString()); test(path); }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Theory]
    [InlineData(false, "Temperature", 1.5)]
    [InlineData(false, "TopP", .5)]
    [InlineData(false, "RepetitionPenalty", 1)]
    [InlineData(true, "Temperature", 1.5)]
    [InlineData(true, "TopP", .5)]
    [InlineData(true, "RepetitionPenalty", 1)]
    public void NativeSamplingExercisesTemperatureNucleusAndRepetition(bool clone, string option, double value)
    {
        void Prepare(XTTSv2Options options) => XttsOptionFixture.PrepareOptions(options, option);
        using XTTSv2<double> control = clone ? XTTSv2Fixture.CreateClone<double>(Prepare) : XTTSv2Fixture.Create<double>(Prepare);
        using XTTSv2<double> perturbed = clone ? XTTSv2Fixture.CreateClone<double>(Prepare) : XTTSv2Fixture.Create<double>(Prepare);
        typeof(XTTSv2Options).GetProperty(option)!.SetValue(perturbed.GetOptions(), value);
        control.SetTrainingMode(true);
        perturbed.SetTrainingMode(true);
        Assert.NotEqual(XttsOptionFixture.ObserveSampler(control), XttsOptionFixture.ObserveSampler(perturbed));
        Assert.True(control.IsTrainingMode);
        Assert.True(perturbed.IsTrainingMode);
    }

    [Theory]
    [InlineData(false, "DvaeFftSize", 255)]
    [InlineData(false, "DvaeWindowSize", 257)]
    [InlineData(false, "DvaeHopSize", 65)]
    [InlineData(true, "DvaeFftSize", 255)]
    [InlineData(true, "DvaeWindowSize", 257)]
    [InlineData(true, "DvaeHopSize", 65)]
    public void NativeWaveformEncodingEnforcesSpectralOptions(bool clone, string option, int value)
    {
        using XTTSv2<double> control = clone ? XTTSv2Fixture.CreateClone<double>() : XTTSv2Fixture.Create<double>();
        using XTTSv2<double> perturbed = clone ? XTTSv2Fixture.CreateClone<double>() : XTTSv2Fixture.Create<double>();
        typeof(XTTSv2Options).GetProperty(option)!.SetValue(perturbed.GetOptions(), value);
        control.SetTrainingMode(true);
        perturbed.SetTrainingMode(true);
        long baseline = XttsOptionFixture.ObserveCodec(control);
        if (option == "DvaeHopSize")
            Assert.NotEqual(baseline, XttsOptionFixture.ObserveCodec(perturbed));
        else
            Assert.Throws<ArgumentOutOfRangeException>(() => XttsOptionFixture.ObserveCodec(perturbed));
        Assert.True(control.IsTrainingMode);
        Assert.True(perturbed.IsTrainingMode);
    }
}
