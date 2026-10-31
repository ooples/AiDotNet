using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.MultiModal;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Moshi's tasks on the tiny fixture: App. C's TTS and ASR, the dialogue, the inner monologue's text stream
/// (Eq. 5), the language model's stages and Mimi's adversarial training.</summary>
public class MoshiTaskTests
{
    // double.IsFinite is not on .NET Framework.
    private static bool Finite(double x) => !double.IsNaN(x) && !double.IsInfinity(x);

    [Fact(Timeout = 120000)]
    public async Task TextStream_FollowsEquation5()
    {
        await Task.Yield();
        using var model = MoshiFixture.Tts<double>(MoshiFixture.Options());
        var core = model.Core;
        int pad = core.Options.TextPadding, epad = core.Options.TextEndPadding;
        // Frames at 100 Hz: "hello" starts at 0.03 s (frame 3), "world" at 0.10 s (frame 10).
        var words = new[] { new MoshiWord("hello", 0.03, 0.08), new MoshiWord("world", 0.10, 0.15) };
        var stream = core.TextStream(words, 20, 0.0);
        var hello = core.WordTokens("hello");
        var world = core.WordTokens("world");
        Assert.Equal(epad, stream[2]);
        Assert.Equal(hello, stream.Skip(3).Take(hello.Length).ToArray());
        Assert.Equal(epad, stream[9]);
        Assert.Equal(world, stream.Skip(10).Take(world.Length).ToArray());
        Assert.Equal(pad, stream[19]);
        // ASR: the text 0.05 s behind the audio, its first 5 frames the zero token.
        var delayed = core.TextStream(words, 20, -0.05);
        Assert.All(delayed.Take(5), t => Assert.Equal(-1, t));
        Assert.Equal(hello, delayed.Skip(8).Take(hello.Length).ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task Speak_ConsumesEveryWordAndIsDeterministic()
    {
        await Task.Yield();
        using var model = MoshiFixture.Tts<double>();
        var (audio, words) = model.Speak("the cat sat");
        Assert.True(audio.Length > 0);
        Assert.Equal(new[] { "the", "cat", "sat" }, words.Select(w => w.Text).ToArray());
        Assert.True(words.Zip(words.Skip(1), (a, b) => a.Time <= b.Time).All(x => x));
        var again = model.Synthesize("the cat sat");
        Assert.Equal(audio.ToArray(), again.ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task Chat_AnswersOverTheUsersSpan()
    {
        await Task.Yield();
        using var model = MoshiFixture.Tts<double>(MoshiFixture.Options());
        var user = MoshiFixture.Tones<double>(0.5);
        var (audio, _) = model.Chat(user);
        int spf = model.Core.Mimi.Configuration.SamplesPerFrame;
        Assert.Equal(model.Core.AudioCodes(user).GetLength(1) * spf, audio.Length);
    }

    [Fact(Timeout = 300000)]
    public async Task Release_DepthLayoutTakesTheUserAsInput()
    {
        await Task.Yield();
        var options = MoshiFixture.Options();
        options.DepthLayout = MoshiDepthLayout.MoshiStreams;
        using var model = MoshiFixture.Tts<double>(options);
        Assert.Equal(options.Codebooks, model.Core.LanguageModel.Configuration.DepthStreams);
        var (audio, _) = model.Chat(MoshiFixture.Tones<double>(0.3));
        Assert.True(audio.Length > 0);
    }

    [Fact(Timeout = 300000)]
    public async Task Transcribe_ReturnsTimedWords()
    {
        await Task.Yield();
        using var model = MoshiFixture.Asr<double>();
        var result = model.Transcribe(MoshiFixture.Tones<double>(0.5), includeTimestamps: true);
        string Squeeze(string x) => string.Concat(x.Where(c => !char.IsWhiteSpace(c)));
        Assert.Equal(Squeeze(result.Text), Squeeze(string.Concat(result.Segments.Select(s => s.Text))));
        Assert.All(result.Segments, s => Assert.InRange(s.StartTime, 0.0, result.DurationSeconds));
    }

    [Fact(Timeout = 600000)]
    public async Task Stages_TrainAndLowerTheLoss()
    {
        await Task.Yield();
        foreach (var stage in new[] { MoshiTrainingStage.PreTraining, MoshiTrainingStage.PostTraining, MoshiTrainingStage.FisherFineTuning,
                     MoshiTrainingStage.InstructFineTuning })
        {
            var options = MoshiFixture.Options();
            options.Stage = stage;
            options.TextMaskProbability = 0.0;
            options.TextDelaySeconds = 0.0;
            options.LearningRate = 1e-2;
            options.DepthLearningRate = 1e-2;
            using var model = MoshiFixture.Tts<double>(options);
            var conversation = new MoshiConversation<double>
            {
                Moshi = MoshiFixture.Tones<double>(0.4), User = MoshiFixture.Tones<double>(0.4, 140, 260),
                Words = MoshiFixture.Words("the cat sat", 0.4),
            };
            double first = model.TrainConversation(conversation);
            double text = model.TrainText("the cat sat on the mat");
            Assert.True(Finite(text));
            double last = first;
            for (int i = 0; i < 4; i++) last = model.TrainConversation(conversation);
            Assert.True(Finite(first) && Finite(last), $"{stage}: {first} → {last}");
            if (stage == MoshiTrainingStage.PreTraining) Assert.True(last < first, $"{stage}: {first} → {last}");
        }
    }

    [Fact(Timeout = 300000)]
    public async Task InstructAugmentation_ChangesOnlyTheUser()
    {
        await Task.Yield();
        using var model = MoshiFixture.Tts<double>(MoshiFixture.Options());
        var user = Enumerable.Range(0, 1600).Select(i => Math.Sin(i * 0.1)).ToArray();
        var moshi = Enumerable.Range(0, 1600).Select(i => Math.Sin(i * 0.37)).ToArray();
        var noise = new[] { Enumerable.Range(0, 400).Select(i => 0.1 * Math.Cos(i * 1.3)).ToArray() };
        int changed = 0;
        for (int seed = 0; seed < 20; seed++)
        {
            var augmented = model.Core.AugmentUser(user, moshi, noise, new Random(seed));
            Assert.Equal(user.Length, augmented.Length);
            Assert.All(augmented, v => Assert.True(Finite(v)));
            if (!augmented.SequenceEqual(user)) changed++;
        }
        // P(no augmentation) = 0.5 · 0.7 · 0.7 ≈ 0.25.
        Assert.InRange(changed, 8, 20);
    }

    [Fact(Timeout = 600000)]
    public async Task TrainMimi_StepsTheCodecAndTheDiscriminator()
    {
        await Task.Yield();
        var options = MoshiFixture.Options();
        options.IncludeMimiTraining = true;
        using var model = MoshiFixture.Tts<double>(options);
        var audio = MoshiFixture.Tones<double>(1.0);
        var before = model.EncodeToTokens(audio).ToArray();
        double loss = model.TrainMimi(audio);
        Assert.True(Finite(loss));
        loss = model.TrainMimi(audio);
        Assert.True(Finite(loss));
        model.ApplyMimiWeightAverage();
        Assert.Equal(before.Length, model.EncodeToTokens(audio).Length);
    }

    [Fact(Timeout = 300000)]
    public async Task Codec_RoundTripsTokens()
    {
        await Task.Yield();
        using var model = MoshiFixture.Tts<double>();
        var tokens = model.EncodeToTokens(MoshiFixture.Tones<double>(0.2));
        Assert.Equal(model.NumCodebooks, tokens.Shape[1]);
        var audio = model.DecodeFromTokens(tokens);
        Assert.Equal(tokens.Shape[0] * model.Core.Mimi.Configuration.SamplesPerFrame, audio.Length);
    }
}
