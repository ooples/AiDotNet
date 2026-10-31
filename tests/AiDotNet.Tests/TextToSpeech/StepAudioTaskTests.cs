using System;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.MultiModal;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Step-Audio's tasks and training stages on the tiny fixture: tokenization, TTS (and the music route), chat,
/// pre-training, the two fine-tunings, the reward model and PPO.</summary>
public class StepAudioTaskTests
{
    // double.IsFinite is not on .NET Framework.
    private static bool Finite(double x) => !double.IsNaN(x) && !double.IsInfinity(x);

    [Fact(Timeout = 300000)]
    public async Task AudioTokens_InterleaveTwoLinguisticThenThreeSemantic()
    {
        await Task.Yield();
        using var model = StepAudioFixture.Tts<double>();
        var text = model.AudioTokenString(StepAudioFixture.Tones<double>(1.0), StepAudioFixture.SampleRate);
        var codes = Regex.Matches(text, "<audio_(\\d+)>").Cast<Match>().Select(m => int.Parse(m.Groups[1].Value)).ToArray();
        Assert.True(codes.Length >= 5 && codes.Length % 5 == 0, text);
        for (int g = 0; g < codes.Length / 5; g++)
        {
            Assert.All(codes.Skip(5 * g).Take(2), c => Assert.InRange(c, 0, 1023));
            Assert.All(codes.Skip(5 * g + 2).Take(3), c => Assert.InRange(c, 1024, 1024 + 15));
        }
    }

    [Fact(Timeout = 600000)]
    public async Task Speak_IsDeterministic_AndRapUsesTheMusicDecoder()
    {
        await Task.Yield();
        using var model = StepAudioFixture.Tts<double>();
        var audio = model.Speak("今天天气很好");
        Assert.Equal(audio.ToArray(), model.Synthesize("今天天气很好").ToArray());
        Assert.True(StepAudioTokens.IsMusic("(RAP)我们一起唱"));
        Assert.NotNull(model.Core.Music);
        Assert.NotNull(model.Speak("(RAP)我们一起唱"));
    }

    [Fact(Timeout = 300000)]
    public async Task Answer_WritesText()
    {
        await Task.Yield();
        using var model = StepAudioFixture.Tts<double>();
        Assert.NotNull(model.Answer("你好"));
        Assert.NotNull(model.Answer(StepAudioFixture.Tones<double>(0.5), StepAudioFixture.SampleRate));
    }

    [Fact(Timeout = 900000)]
    public async Task Stages_TrainAndLowerTheLoss()
    {
        await Task.Yield();
        var recording = StepAudioFixture.Tones<double>(0.6, 200, 330);
        foreach (var stage in new[] { StepAudioTrainingStage.PreTraining1, StepAudioTrainingStage.PreTraining3, StepAudioTrainingStage.TtsFineTuning,
                     StepAudioTrainingStage.ChatFineTuning })
        {
            var options = StepAudioFixture.Options();
            options.Stage = stage;
            using var model = StepAudioFixture.Tts<double>(options);
            Func<double> step = stage switch
            {
                StepAudioTrainingStage.TtsFineTuning => () => model.TrainTts("今天天气很好", recording, StepAudioFixture.SampleRate, model.SpeakerVoice!),
                StepAudioTrainingStage.ChatFineTuning => () => model.TrainChat(recording, StepAudioFixture.SampleRate, "好的，没问题。"),
                _ => () => model.TrainContinuation(recording, StepAudioFixture.SampleRate),
            };
            double first = step(), last = first;
            for (int i = 0; i < 4; i++) last = step();
            Assert.True(Finite(first) && Finite(last) && last < first, $"{stage}: {first} → {last}");
        }
    }

    [Fact(Timeout = 900000)]
    public async Task RewardModel_LearnsThePreference_AndPpoUpdatesThePolicy()
    {
        await Task.Yield();
        var question = StepAudioFixture.Tones<double>(0.4);
        var options = StepAudioFixture.Options();
        options.IncludeRlhfModels = true;
        options.Stage = StepAudioTrainingStage.RewardModel;
        options.CriticWarmupSteps = 1;
        using var model = StepAudioFixture.Tts<double>(options);
        model.InitializeRlhfFromChat();
        double before = model.Reward(question, StepAudioFixture.SampleRate, "好的") - model.Reward(question, StepAudioFixture.SampleRate, "不知道");
        for (int i = 0; i < 5; i++) model.TrainReward(question, StepAudioFixture.SampleRate, "好的", "不知道");
        double after = model.Reward(question, StepAudioFixture.SampleRate, "好的") - model.Reward(question, StepAudioFixture.SampleRate, "不知道");
        Assert.True(after > before, $"margin {before} → {after}");

        options.Stage = StepAudioTrainingStage.Ppo;
        model.InitializeCriticFromReward();
        var chatBefore = model.Core.ChatModel.Layers[0].Query.GetParameters().ToArray();
        double warmup = model.TrainPpo(question, StepAudioFixture.SampleRate);
        double policy = model.TrainPpo(question, StepAudioFixture.SampleRate);
        Assert.True(Finite(warmup) && Finite(policy));
        Assert.NotEqual(chatBefore, model.Core.ChatModel.Layers[0].Query.GetParameters().ToArray());
    }
}
