using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class VoiceboxPhoneFrontEndTests
{
    private static Dictionary<string,int> Vocabulary() => new(StringComparer.Ordinal)
    {
        ["SIL"] = 7, ["ɑ_B"] = 2, ["b_E"] = 9, ["ŋ_S"] = 12,
        ["t_B"] = 1, ["ɪ_I"] = 3, ["ç_E"] = 11
    };
    private static IReadOnlyList<IReadOnlyList<string>> Words(string _) => new IReadOnlyList<string>[]
        { new[] { "ɑ", "b" }, new[] { "ŋ" }, new[] { "t", "ɪ", "ç" } };
    private static int[] Ids() => new[] { 7,2,9,7,12,7,1,3,11,7 };

    [Fact]
    public void MfaWordPhones_KeepIpaSymbolsAndExactIds_WithAllPaperSuffixesAndGhostSilences()
    {
        var tokenizer = new VoiceboxPhoneTokenizer(16,Vocabulary());
        Assert.Equal(new[] { "SIL","ɑ_B","b_E","SIL","ŋ_S","SIL","t_B","ɪ_I","ç_E","SIL" },tokenizer.Phonemize("three words",Words));
        Assert.Equal(Ids(),tokenizer.Encode("three words",16,Words));
        Assert.Equal(7,tokenizer.SilenceId);
        Assert.Throws<ArgumentException>(() => tokenizer.Encode("three words",9,Words));
        Assert.Throws<KeyNotFoundException>(() => tokenizer.Encode("unknown",16,_ => new IReadOnlyList<string>[] { new[] { "ʔ" } }));
    }

    [Fact]
    public void MfaInventory_RejectsMissingBackendAndDuplicateIds_InsteadOfSubstitutingArpabet()
    {
        var tokenizer = new VoiceboxPhoneTokenizer(16,Vocabulary());
        Assert.Throws<InvalidOperationException>(() => tokenizer.Phonemize("example"));
        Assert.Throws<InvalidOperationException>(() => new VoiceboxPhoneTokenizer(320).Phonemize("example",Words));
        var invalid = Vocabulary(); invalid["ɑ_B"] = 7;
        Assert.Throws<ArgumentException>(() => new VoiceboxPhoneTokenizer(16,invalid));
        Assert.False(new VoiceboxOptions().UseNativeEnglishFallback);
        Assert.Throws<InvalidOperationException>(() => new VoiceboxPhoneTokenizer(320).Phonemize("example",null,false));
        Assert.NotEmpty(new VoiceboxPhoneTokenizer(320).Phonemize("example",null,true));
    }

    private static VoiceboxOptions Options() => new()
    {
        PhoneVocabularySize=16, PhoneVocabulary=Vocabulary(), PhoneEmbeddingDim=8,
        AudioModelDim=16, AudioFeedForwardDim=32, AudioLayers=2, AudioHeads=4,
        AudioAttentionDropout=0, AudioActivationDropout=0, AudioConvPositionWidth=3, AudioConvPositionGroups=4, AudioConvPositionDepth=1,
        DurationModelDim=16, DurationFeedForwardDim=32, DurationLayers=2, DurationHeads=4,
        DurationAttentionDropout=0, DurationActivationDropout=0, DurationConvPositionWidth=3, DurationConvPositionGroups=4, DurationConvPositionDepth=1,
        OdeSteps=1, GuidanceStrength=.2, MaximumAudioFrames=64, MaximumPhoneDuration=4,
        MaxTextLength=16, VocoderInitialChannels=16, VocoderResBlockKernels=new[] {3},
        VocoderResBlockDilations=new[] {new[] {1}}, SamplingSeed=7
    };

    [Fact]
    public void PublicMfaPath_ReachesPromptEnrollmentAndWaveform_AndRestoresItsExactInventory()
    {
        AiDotNetEngine.ResetToCpu();
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,NeuralNetworkTaskType.Generative,inputSize:16,outputSize:80);
        var options = Options(); using var model = new Voicebox<double>(architecture,options);
        int calls=0; model.MfaWordPhonemizer = text => { calls++; return Words(text); };
        var audio = new Tensor<double>(new[] {1280},new Vector<double>(Enumerable.Range(0,1280).Select(i => .2*Math.Sin(.11*i))));
        var durations = new[] {0,1,1,0,1,0,1,1,1,0};
        var voice = model.CreateVoice(audio,16000,"MFA prompt",durations);
        Assert.Equal(Ids().Select(x => (double)x),voice.ReferenceTokens!.ToArray());
        Assert.Equal(durations,voice.ReferenceDurations); model.Voice=voice;
        var waveform = model.Synthesize("MFA target");
        Assert.NotEmpty(waveform.ToArray()); Assert.All(waveform.ToArray(),value => Assert.True(!double.IsNaN(value) && !double.IsInfinity(value)));
        Assert.Equal(2,calls);
        var copiedOptions = new VoiceboxOptions(options); copiedOptions.PhoneVocabulary!["ɑ_B"]=4;
        Assert.Equal(2,options.PhoneVocabulary!["ɑ_B"]);
        using var restored = new Voicebox<double>(architecture,copiedOptions);
        restored.Deserialize(model.Serialize()); restored.MfaWordPhonemizer=Words;
        var restoredVoice = restored.CreateVoice(audio,16000,"restored MFA prompt",durations);
        Assert.Equal(Ids().Select(x => (double)x),restoredVoice.ReferenceTokens!.ToArray());
        Assert.Equal(model.Phonemize("MFA"),restored.Phonemize("MFA"));
    }
}
