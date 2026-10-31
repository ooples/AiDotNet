using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FrontEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// VALL-E X (Zhang et al. 2023): VALL-E's networks with a language ID, an English and Mandarin vocabulary, the NAR
/// prompted by another sentence of the same speaker, and cross-lingual synthesis.
/// </summary>
public class VALLEXPaperTests
{
    private static VALLEXOptions Tiny(VallELanguagePlacement placement = VallELanguagePlacement.AcousticTokens) => new()
    {
        HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, NumDecoderLayers = 1, FeedForwardDim = 32, TextTokens = 192,
        NumCodebooks = 3, CodebookSize = 64, MaxCodesPerTextToken = 2, LearningRate = 3e-3, WarmupSteps = 0, DropoutRate = 0.0,
        LanguagePlacement = placement,
        Codec = new EnCodecOptions
        {
            SampleRate = 24000, NumQuantizers = 3, CodebookSize = 64, Filters = 4, Ratios = [8, 5, 4, 2], Dimension = 8,
            ResidualKernelSizes = [3, 1], TargetBandwidthKbps = 3 * 75 * 6 / 1000.0,
        },
    };

    private static VALLEX<double> Create(VALLEXOptions options) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);

    private static Tensor<double> Tone(int samples)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
            audio[i] = 0.4 * Math.Sin(2 * Math.PI * 180.0 * i / 24000) + 0.2 * Math.Sin(2 * Math.PI * 470.0 * i / 24000);
        return audio;
    }

    private static Tensor<double> Codes(int frames, int offset)
    {
        var codes = new Tensor<double>(new[] { frames, 3 });
        for (int f = 0; f < frames; f++)
            for (int q = 0; q < 3; q++) codes[f, q] = (5 * f + 11 * q + offset) % 64;
        return codes;
    }

    private static TtsTrainingSample<double> Sample(VALLEX<double> model, string text, int language, bool withPrompt = true)
    {
        var tokens = model.TextToTokens(text);
        return new TtsTrainingSample<double>
        {
            Tokens = tokens,
            CodecTokens = Codes(12, 0),
            PromptCodecTokens = withPrompt ? Codes(6, 17) : null,
            LanguageId = language,
        };
    }

    private static double[] Snapshot(IEnumerable<LayerBase<double>> layers) =>
        layers.SelectMany(l => l.GetParameters().ToArray()).ToArray();

    [Fact(Timeout = 60000)]
    public async Task Vocabulary_HoldsBothLanguagesOnce()
    {
        await Task.Yield();
        using var model = Create(Tiny());
        int expected = 3 + LibriTtsPhonemeTable.Symbols.Concat(MandarinG2P.Symbols).Distinct(StringComparer.Ordinal).Count();
        Assert.Equal(expected, ((VALLEXOptions)model.GetOptions()).VocabSize);
        // Both languages' phonemes are in it; a symbol outside the table is skipped, as the reference's tokenizer skips it.
        Assert.Equal(3, model.EncodePhonemes(new[] { "aɪ", "↓", "_", "%" }).Length);
    }

    [Fact(Timeout = 60000)]
    public async Task TokenLanguages_FollowTheScript()
    {
        await Task.Yield();
        using var model = Create(Tiny());
        // English, then Mandarin: every token of each run takes its run's language, shared symbols included.
        var ids = model.TextToTokens("hello 你好").ToArray().Select(v => (int)v).ToArray();
        var languages = model.TokenLanguages(ids);
        var english = model.EncodePhonemes(EnglishG2P.Default.Phonemize("hello")).Length;
        Assert.All(languages.Skip(1).Take(english), l => Assert.Equal((int)VallEXLanguage.English, l));
        Assert.All(languages.Skip(1 + english + 1).Take(ids.Length - english - 3), l => Assert.Equal((int)VallEXLanguage.Mandarin, l));
    }

    [Fact(Timeout = 180000)]
    public async Task EachStage_TrainsItsOwnModel_AndTheNarNeedsAPromptSentence()
    {
        await Task.Yield();
        using var model = Create(Tiny());
        var sample = Sample(model, "你好，世界。", (int)VallEXLanguage.Mandarin);
        foreach (var stage in new[] { VallETrainingStage.AutoRegressive, VallETrainingStage.NonAutoRegressive })
        {
            model.CurrentStage = stage;
            var own = stage == VallETrainingStage.AutoRegressive ? model.AutoRegressiveLayers : model.NonAutoRegressiveLayers;
            var other = stage == VallETrainingStage.AutoRegressive ? model.NonAutoRegressiveLayers : model.AutoRegressiveLayers;
            var ownBefore = Snapshot(own);
            var otherBefore = Snapshot(other);
            model.Train(sample);
            Assert.NotEqual(ownBefore, Snapshot(own));
            Assert.True(otherBefore.SequenceEqual(Snapshot(other)), $"{stage} changed the other model.");
        }
        Assert.Throws<ArgumentException>(() => model.Train(Sample(model, "你好", 1, withPrompt: false)));
        Assert.Throws<ArgumentException>(() => model.Train(new TtsTrainingSample<double>
        {
            Tokens = model.TextToTokens("hello"), CodecTokens = Codes(12, 0), PromptCodecTokens = Codes(6, 17),
        }));
    }

    [Theory(Timeout = 120000)]
    [InlineData(VallELanguagePlacement.AcousticTokens)]
    [InlineData(VallELanguagePlacement.TextTokens)]
    public async Task LanguageId_ReachesTheObjective(VallELanguagePlacement placement)
    {
        await Task.Yield();
        using var model = Create(Tiny(placement));
        // The same utterance labelled English and Mandarin: the AR objective reads the language embedding.
        double english = model.EvaluateTrainingObjective(Sample(model, "hello there", (int)VallEXLanguage.English));
        double mandarin = model.EvaluateTrainingObjective(Sample(model, "hello there", (int)VallEXLanguage.Mandarin));
        Assert.NotEqual(english, mandarin);
    }

    [Theory(Timeout = 300000)]
    [InlineData(VallETrainingStage.AutoRegressive)]
    [InlineData(VallETrainingStage.NonAutoRegressive)]
    public async Task Training_LowersTheStageObjective(VallETrainingStage stage)
    {
        await Task.Yield();
        using var model = Create(Tiny());
        model.CurrentStage = stage;
        var sample = Sample(model, "我们去银行。", (int)VallEXLanguage.Mandarin);
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 30; i++) model.Train(sample);
        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < 0.8 * before, $"{stage}: the objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 180000)]
    public async Task CrossLingualSynthesis_IsDeterministic_AndReadsTheText()
    {
        await Task.Yield();
        using var model = Create(Tiny());
        var sample = Sample(model, "你好，世界。", (int)VallEXLanguage.Mandarin);
        for (int i = 0; i < 20; i++) model.Train(sample);

        // An English prompt speaking Mandarin.
        model.Voice = model.CreateVoice(Tone(24000 / 3), "a prompt");
        var first = model.Synthesize("你好，世界。");
        var second = model.Synthesize("你好，世界。");
        Assert.Equal(first.ToArray(), second.ToArray());
        Assert.All(first.ToArray(), v => Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(v)));
        Assert.True(first.Length > 0 && first.Length % 320 == 0, $"{first.Length} samples.");
        var other = model.Synthesize("今天天气很好，我们去公园散步吧。");
        Assert.True(other.Length != first.Length || Enumerable.Range(0, first.Length).Any(i => other[i] != first[i]),
            "Different text synthesized identical audio.");
    }
}
