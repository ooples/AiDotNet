using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.MultiModal;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// SpiritLM's tasks on the tiny fixture: the release's tokenizer layout, prompts and forbidden ids, the arbitrary-output
/// helpers against the release, the interleaved training sequences, each training objective, the vocoder's and the pitch
/// VQ-VAE's training, and generation in every modality.
/// </summary>
public class SpiritLmTaskTests
{
    [Fact(Timeout = 60000)]
    public async Task Tokenizer_EncodesAsTheReleaseDoes()
    {
        await Task.Yield();
        var model = SpiritLmFixture.Tts<double>();
        foreach (var e in SpiritLmFixture.TokenizerReference["tiny"]!)
        {
            Assert.Equal(e["ids"]!.Select(v => (int)v).ToArray(), model.Encode((string)e["text"]!));
            Assert.Equal((string)e["decoded"]!, model.Tokenizer.Decode(model.Encode((string)e["text"]!).Skip(1)));
        }
        // Gated on the released tokenizer.model (spirit-lm-expressive-7b).
        if (Environment.GetEnvironmentVariable("SPIRITLM_TOKENIZER") is { Length: > 0 } path && File.Exists(path))
        {
            var real = new SentencePieceModel(File.ReadAllBytes(path));
            Assert.Equal((int)SpiritLmFixture.TokenizerReference["real_pieces"]!, real.Count);
            foreach (var e in SpiritLmFixture.TokenizerReference["real"]!)
            {
                var ids = new[] { real.BosId }.Concat(real.Encode((string)e["text"]!)).ToArray();
                Assert.Equal(e["ids"]!.Select(v => (int)v).ToArray(), ids);
            }
        }
    }

    [Fact(Timeout = 60000)]
    public async Task ArbitraryOutputHelpers_MatchTheRelease()
    {
        await Task.Yield();
        foreach (var c in SpiritLmFixture.TokenizerReference["helper_cases"]!)
        {
            string text = (string)c["text"]!;
            Assert.Equal((bool)c["starts"]!, SpiritLmTokens.StartsWithSpeechToken(text));
            Assert.Equal((bool)c["ends"]!, SpiritLmTokens.EndsWithSpeechToken(text));
            Assert.Equal(c["last_start"]!.Type == Newtonsoft.Json.Linq.JTokenType.Null ? null : (int?)c["last_start"], SpiritLmTokens.LastSpeechStart(text));
            var split = SpiritLmTokens.SplitSpeechAndText(text);
            var expected = c["split"]!.Select(p => ((string)p[0]!, (string)p[1]! == "s")).ToList();
            Assert.Equal(expected, split.Select(s => (s.Content, s.IsSpeech)).ToList());
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Prompts_AndForbiddenIds_AreTheRelease()
    {
        await Task.Yield();
        var model = SpiritLmFixture.Tts<double>();
        var speech = SpiritLmContent<double>.FromSpeech(SpiritLmFixture.Tones<double>(0.5));
        string speechTokens = model.SpeechString(speech.Audio!);
        Assert.StartsWith("[Hu", speechTokens);
        Assert.Equal("[Text]hello[Speech]" + speechTokens + "[Text]", model.Prompt(new[] { SpiritLmContent<double>.FromText("hello"), speech }, SpiritLmModality.Text));
        Assert.Equal("[Speech]" + speechTokens, model.Prompt(new[] { speech }, SpiritLmModality.Speech));
        Assert.Equal("[Text]a b", model.Prompt(new[] { SpiritLmContent<double>.FromText("a"), SpiritLmContent<double>.FromText(" b") }, SpiritLmModality.Arbitrary));
        int text = (int)SpiritLmFixture.TokenizerReference["tiny_text_marker"]!;
        var forSpeech = model.Forbidden(SpiritLmModality.Speech);
        Assert.Equal(Enumerable.Range(0, text + 2).ToHashSet(), forSpeech.ToHashSet());
        var forText = model.Forbidden(SpiritLmModality.Text);
        Assert.Equal(new[] { text, text + 1 }.Concat(Enumerable.Range(text + 2, SpiritLmFixture.Units + 1)).ToHashSet(), forText.ToHashSet());
        Assert.Empty(model.Forbidden(SpiritLmModality.Arbitrary));
    }

    [Fact(Timeout = 120000)]
    public async Task InterleavedSequence_SwitchesAtWordBoundaries()
    {
        await Task.Yield();
        var model = SpiritLmFixture.Tts<double>();
        var audio = SpiritLmFixture.Tones<double>(1.2);
        var words = SpiritLmFixture.Words("the cat sat on the mat today", 1.2);
        string sequence = model.InterleavedSequence(audio, words, new Random(3));
        var runs = SpiritLmTokens.SplitSpeechAndText(sequence);
        Assert.Contains(runs, r => r.IsSpeech);
        Assert.Contains(runs, r => !r.IsSpeech);
        // Every word reaches the sequence exactly once, in order, as text or as the speech of its span.
        string text = string.Join(" ", runs.Where(r => !r.IsSpeech).Select(r => r.Content.Trim()));
        Assert.All(text.Split(' ', StringSplitOptions.RemoveEmptyEntries), w => Assert.Contains(w, words.Select(x => x.Text)));
        Assert.True(sequence.StartsWith("[Text]", StringComparison.Ordinal) || sequence.StartsWith("[Speech]", StringComparison.Ordinal));
    }

    [Fact(Timeout = 600000)]
    public async Task EachObjective_FallsWithTraining()
    {
        await Task.Yield();
        var audio = SpiritLmFixture.Tones<double>(0.8);
        var words = SpiritLmFixture.Words("the cat sat on the mat", 0.8);
        var model = SpiritLmFixture.Tts<double>();
        foreach (var sequence in new[] { model.TextSequence("the cat sat"), model.SpeechSequence(audio), model.InterleavedSequence(audio, words, new Random(5)) })
        {
            double first = Convert.ToDouble(model.TrainSequence(sequence)), last = first;
            for (int i = 0; i < 8; i++) last = Convert.ToDouble(model.TrainSequence(sequence));
            Assert.True(last < first, $"{sequence}: {first} then {last}");
        }
        var sample = new TtsTrainingSample<double>
        {
            Tokens = AiDotNet.TextToSpeech.Latest.IndexTtsOps.Ids<double>(model.TokenizeText("hello there")),
            Audio = audio,
        };
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 6; i++) model.Train(sample);
        Assert.True(model.EvaluateTrainingObjective(sample) < before);
    }

    [Theory(Timeout = 600000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task VocoderTraining_LowersTheGeneratorLoss(bool expressive)
    {
        await Task.Yield();
        var options = SpiritLmFixture.Options(expressive);
        options.IncludeVocoderDiscriminators = true;
        // One fixed segment of four units (the release trains on segments of whole units): each step sees the same example.
        options.VocoderSegmentSize = 4 * 640;
        var model = SpiritLmFixture.Tts<double>(options);
        var (units, pitch, styles) = model.SpeechUnits(SpiritLmFixture.Tones<double>(0.6));
        var full = SpiritLmFixture.Tones<double>(0.6);
        var speech = new Tensor<double>(new[] { 4 * 640 }, new Vector<double>(full.ToArray().Take(4 * 640).ToArray()));
        units = units.Take(4).ToArray();
        pitch = pitch?.Take(2).ToArray();
        styles = styles?.Take(1).ToArray();
        double first = Convert.ToDouble(model.TrainVocoder(speech, units, pitch, styles, speaker: 2, style: expressive ? null : 3)), last = first;
        for (int i = 0; i < 3; i++) last = Convert.ToDouble(model.TrainVocoder(speech, units, pitch, styles, speaker: 2, style: expressive ? null : 3));
        Assert.True(last < first, $"{first} then {last}");
    }

    [Fact(Timeout = 600000)]
    public async Task PitchQuantizerTraining_LowersItsLoss_AndClonesKeepTheCodebook()
    {
        await Task.Yield();
        var model = SpiritLmFixture.Tts<double>(SpiritLmFixture.Options(expressive: true));
        var speech = SpiritLmFixture.Tones<double>(1.0);
        double first = Convert.ToDouble(model.TrainPitchQuantizer(speech)), last = first;
        for (int i = 0; i < 8; i++) last = Convert.ToDouble(model.TrainPitchQuantizer(speech));
        Assert.True(last < first, $"{first} then {last}");
        var clone = (SpiritLM<double>)model.Clone();
        Assert.Equal(model.SpeechString(speech), clone.SpeechString(speech));
    }

    [Fact(Timeout = 600000)]
    public async Task GenerationInEveryModality_Runs()
    {
        await Task.Yield();
        var model = SpiritLmFixture.Tts<double>();
        var speech = SpiritLmContent<double>.FromSpeech(SpiritLmFixture.Tones<double>(0.6));
        var text = model.Generate(new[] { speech }, SpiritLmModality.Text);
        Assert.Single(text);
        Assert.NotNull(text[0].Text);
        var spoken = model.Generate(new[] { SpiritLmContent<double>.FromText("hello there") }, SpiritLmModality.Speech);
        Assert.Single(spoken);
        Assert.All(spoken[0].Audio!.ToArray(), v => Assert.False(double.IsNaN(v)));
        var any = model.Generate(new[] { speech }, SpiritLmModality.Arbitrary);
        Assert.All(any, c => Assert.True(c.IsSpeech || c.Text is not null));
        var synthesized = model.Synthesize("hello there");
        Assert.Equal(synthesized.ToArray(), model.Synthesize("hello there").ToArray());
    }

    [Fact(Timeout = 600000)]
    public async Task SonarExpressive_PretrainAndFinetuneLowerTheirLoss()
    {
        await Task.Yield();
        var model = SpiritLmFixture.SpeechPropModel<double>();
        var audio = SpiritLmFixture.Tones<double>(1.0);
        var semantic = new Tensor<double>(new[] { 16 }, new Vector<double>(Enumerable.Range(0, 16).Select(i => Math.Sin(i)).ToArray()));
        foreach (bool finetune in new[] { false, true })
        {
            double first = Convert.ToDouble(model.TrainSonarExpressive(audio, semantic, finetune)), last = first;
            for (int i = 0; i < 10; i++) last = Convert.ToDouble(model.TrainSonarExpressive(audio, semantic, finetune));
            Assert.True(last < first, $"{(finetune ? "fine-tuning" : "pre-training")}: {first} then {last}");
        }
        var speech = model.SonarExpressiveSpeech(semantic, audio, maxFrames: 8);
        Assert.All(speech.ToArray(), v => Assert.False(double.IsNaN(v)));
    }

    [Fact(Timeout = 600000)]
    public async Task SpeechPropStyles_TrainFitAndTokenize()
    {
        await Task.Yield();
        var model = SpiritLmFixture.SpeechPropModel<double>();
        var recordings = new[] { SpiritLmFixture.Tones<double>(2.0, 150, 400), SpiritLmFixture.Tones<double>(2.0, 260, 700) };
        double first = Convert.ToDouble(model.TrainSpeechPropStyle(recordings[0], 2)), last = first;
        for (int i = 0; i < 8; i++)
        {
            model.TrainSpeechPropStyle(recordings[1], 4);
            last = Convert.ToDouble(model.TrainSpeechPropStyle(recordings[0], 2));
        }
        Assert.True(last < first, $"{first} then {last}");
        model.FitStyleCentroids(recordings.Concat(new[] { SpiritLmFixture.Tones<double>(2.0, 200, 520) }));
        var (_, _, styles) = model.SpeechUnits(recordings[0]);
        Assert.NotNull(styles);
        Assert.Equal(2, styles!.Length);
        Assert.All(styles, s => Assert.InRange(s, 0, 3));
        // The fitted centroids are model state: a clone tokenizes alike.
        var clone = (SpiritLM<double>)model.Clone();
        Assert.Equal(model.SpeechString(recordings[1]), clone.SpeechString(recordings[1]));
    }

    [Fact(Timeout = 600000)]
    public async Task SonarTextConditioning_EmbedsAndTrains()
    {
        await Task.Yield();
        var model = SpiritLmFixture.SpeechPropModel<double>();
        var a = model.SonarEmbedding("the cat sat", "eng_Latn");
        Assert.Equal(16, a.Length);
        Assert.Equal(a.ToArray(), model.SonarEmbedding("the cat sat", "eng_Latn").ToArray());
        Assert.NotEqual(a.ToArray(), model.SonarEmbedding("hello there", "eng_Latn").ToArray());
        Assert.NotEqual(a.ToArray(), model.SonarEmbedding("the cat sat", "fra_Latn").ToArray());
        var audio = SpiritLmFixture.Tones<double>(1.0);
        double first = Convert.ToDouble(model.TrainSonarExpressive(audio, "the cat sat", "eng_Latn", false)), last = first;
        for (int i = 0; i < 10; i++) last = Convert.ToDouble(model.TrainSonarExpressive(audio, "the cat sat", "eng_Latn", false));
        Assert.True(last < first, $"{first} then {last}");
    }

    /// <summary>Expressive speech tokens carry pitch and style; voiced back, every frame has its conditioning. (The release's
    /// decoder fails on an Expressive string without pitch tokens, and so does this one.)</summary>
    [Fact(Timeout = 600000)]
    public async Task ExpressiveSpeech_RoundTripsThroughTheVocoder()
    {
        await Task.Yield();
        var model = SpiritLmFixture.Tts<double>(SpiritLmFixture.Options(expressive: true));
        var audio = SpiritLmFixture.Tones<double>(1.2);
        string tokens = model.SpeechString(audio);
        Assert.Contains("[Pi", tokens);
        Assert.Contains("[St", tokens);
        var voiced = model.Speak(tokens);
        Assert.True(voiced.Length > 0);
        Assert.All(voiced.ToArray(), v => Assert.False(double.IsNaN(v)));
        Assert.Throws<ArgumentException>(() => model.Speak("[Hu1][Hu2]"));
        var text = model.Generate(new[] { SpiritLmContent<double>.FromSpeech(audio) }, SpiritLmModality.Text);
        Assert.NotNull(text[0].Text);
    }
}
