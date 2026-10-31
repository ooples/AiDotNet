using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.MultiModal;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// SpeechGPT's tasks on the tiny fixture: the release's prompts and unit strings, the three language-model stages, the
/// multi-speaker vocoder's GAN training, synthesis, chat and the speech recognizer.
/// </summary>
public class SpeechGptTaskTests
{
    [Fact(Timeout = 120000)]
    public async Task Prompts_AreTheReleaseTemplate()
    {
        await Task.Yield();
        var model = SpeechGptFixture.Tts<double>();
        var options = SpeechGptFixture.Options();
        var tokenizer = model.Core.Tokenizer;
        Assert.Equal(tokenizer.Encode(options.MetaInstruction + "[Human]: Can you please read this sentence out loud? This is input: hi <eoh>. [SpeechGPT]: "),
            model.Core.Prompt(model.TtsQuestion("hi")));
        Assert.Equal("<sosp><3><0><11><eosp>", SpeechGptCore<double>.UnitString(new[] { 3, 0, 11 }));
        var ids = tokenizer.Encode("<sosp><3><0><11><eosp>", addBos: false);
        Assert.Equal(new[] { 3, 0, 11 }, model.Core.UnitsIn(ids));
    }

    [Fact(Timeout = 600000)]
    public async Task EachStage_LowersItsObjective()
    {
        await Task.Yield();
        var speech = SpeechGptFixture.Tones<double>(0.6, 230, 610);
        foreach (var stage in new[] { SpeechGptTrainingStage.ModalityAdaptation, SpeechGptTrainingStage.CrossModalInstruction, SpeechGptTrainingStage.ChainOfModality })
        {
            var options = SpeechGptFixture.Options();
            options.Stage = stage;
            var model = SpeechGptFixture.Tts<double>(options);
            Func<double> step = stage switch
            {
                SpeechGptTrainingStage.ModalityAdaptation => () => Convert.ToDouble(model.TrainContinuation(speech)),
                SpeechGptTrainingStage.CrossModalInstruction => () => Convert.ToDouble(model.TrainTranscription(speech, "hello")),
                _ => () => Convert.ToDouble(model.TrainChainOfModality(speech, "hi there", "hello", speech)),
            };
            double first = step(), last = first;
            for (int i = 0; i < 8; i++) last = step();
            Assert.True(last < first, $"{stage}: {first} then {last}");
        }
    }

    [Fact(Timeout = 600000)]
    public async Task VocoderTraining_LowersTheGeneratorLoss()
    {
        await Task.Yield();
        var options = SpeechGptFixture.Options();
        options.IncludeVocoderDiscriminators = true;
        var model = SpeechGptFixture.Tts<double>(options);
        var speech = SpeechGptFixture.Tones<double>(0.5);
        double first = Convert.ToDouble(model.TrainVocoder(speech, speaker: 2)), last = first;
        for (int i = 0; i < 6; i++) last = Convert.ToDouble(model.TrainVocoder(speech, speaker: 2));
        Assert.True(last < first, $"{first} then {last}");
    }

    [Fact(Timeout = 300000)]
    public async Task SynthesizeChatAndRecognize_Run()
    {
        await Task.Yield();
        var model = SpeechGptFixture.Tts<double>();
        var audio = model.Synthesize("hi there");
        Assert.All(audio.ToArray(), v => Assert.False(double.IsNaN(v)));
        var reply = model.Chat(recording: SpeechGptFixture.Tones<double>(0.4));
        Assert.All(reply.Units, u => Assert.InRange(u, 0, SpeechGptFixture.Units - 1));
        Assert.Equal(reply.Units.Length == 0, reply.Audio.Length == 0);

        var asr = SpeechGptFixture.Asr<double>();
        var result = asr.Transcribe(SpeechGptFixture.Tones<double>(0.4));
        Assert.NotNull(result.Text);
    }

    [Fact(Timeout = 300000)]
    public async Task TtsObjective_FallsWithTraining()
    {
        await Task.Yield();
        var model = SpeechGptFixture.Tts<double>();
        var sample = new TtsTrainingSample<double>
        {
            Tokens = AiDotNet.TextToSpeech.Latest.IndexTtsOps.Ids<double>(model.TokenizeText("hi")),
            Audio = SpeechGptFixture.Tones<double>(0.6),
        };
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 6; i++) model.Train(sample);
        Assert.True(model.EvaluateTrainingObjective(sample) < before);
    }
}
