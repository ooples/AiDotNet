using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.MultiModal;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// LLaMA-Omni's tasks on the tiny fixture: the llama_3 prompt split at the speech, speech in and out, the streaming
/// vocoder chunks, and each training stage.
/// </summary>
public class LlamaOmniTaskTests
{
    [Fact(Timeout = 120000)]
    public async Task Prompt_IsTheLlama3TemplateSplitAtTheSpeech()
    {
        await Task.Yield();
        var model = LlamaOmniFixture.Tts<double>();
        var options = LlamaOmniFixture.Options();
        var (before, after, hasSpeech) = model.PromptIds("<speech>\nPlease directly answer the questions in the user's speech.");
        Assert.True(hasSpeech);
        Assert.Equal(model.TokenizeText($"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{options.SystemPrompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"), before);
        Assert.Equal(model.TokenizeText("\nPlease directly answer the questions in the user's speech.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"), after);
        Assert.Equal(256, before[0]);                                                                          // <|begin_of_text|>, once
    }

    [Fact(Timeout = 300000)]
    public async Task ChatAndSynthesize_ProduceUnitsAndSpeech()
    {
        await Task.Yield();
        var model = LlamaOmniFixture.Tts<double>();
        var response = model.Chat(LlamaOmniFixture.Tones<double>(0.4));
        Assert.All(response.Units, u => Assert.InRange(u, 0, LlamaOmniFixture.Units - 1));
        Assert.Equal(response.Units.Length == 0, response.Audio.Length == 0);
        var audio = model.Synthesize("hi there");
        Assert.All(audio.ToArray(), v => Assert.False(double.IsNaN(v)));

        // Streaming in chunks of Ω units voices the same units, chunk by chunk.
        var options = LlamaOmniFixture.Options();
        options.UnitChunkSize = 2;
        var streaming = LlamaOmniFixture.Tts<double>(options).Chat(LlamaOmniFixture.Tones<double>(0.4));
        Assert.Equal(response.Units, streaming.Units);
    }

    [Fact(Timeout = 600000)]
    public async Task EachStage_LowersItsObjective()
    {
        await Task.Yield();
        var instruction = LlamaOmniFixture.Tones<double>(0.4);
        var speech = LlamaOmniFixture.Tones<double>(0.6, 230, 610);
        foreach (var stage in new[] { LlamaOmniTrainingStage.TextResponse, LlamaOmniTrainingStage.SpeechDecoder, LlamaOmniTrainingStage.Joint })
        {
            var options = LlamaOmniFixture.Options();
            options.Stage = stage;
            options.TotalSteps = 100000;
            var model = LlamaOmniFixture.Tts<double>(options);
            double first = Convert.ToDouble(model.TrainTurn(instruction, "ok then", speech));
            double last = first;
            for (int i = 0; i < 8; i++) last = Convert.ToDouble(model.TrainTurn(instruction, "ok then", speech));
            Assert.True(last < first, $"{stage}: {first} then {last}");
        }
    }

    [Fact(Timeout = 300000)]
    public async Task TtsObjective_FallsWithTraining()
    {
        await Task.Yield();
        var options = LlamaOmniFixture.Options();
        options.TotalSteps = 100000;
        var model = LlamaOmniFixture.Tts<double>(options);
        var sample = new TtsTrainingSample<double>
        {
            Tokens = AiDotNet.TextToSpeech.Latest.IndexTtsOps.Ids<double>(model.TokenizeText("hi")),
            Audio = LlamaOmniFixture.Tones<double>(0.6),
        };
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 8; i++) model.Train(sample);
        Assert.True(model.EvaluateTrainingObjective(sample) < before);
    }
}
