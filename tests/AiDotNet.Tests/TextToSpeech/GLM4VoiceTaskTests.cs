using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.MultiModal;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// GLM-4-Voice's tasks on the tiny fixture: the web demo's prompt as the tokenizer splits it, the streaming-thoughts
/// layout and §4.2.2's split targets, transformers' top-p, synthesis, chat, and each of the three trainings.
/// </summary>
public class GLM4VoiceTaskTests
{
    [Fact(Timeout = 120000)]
    public async Task ChatPrompt_IsTheWebDemoStringTokenized()
    {
        await Task.Yield();
        var model = GLM4VoiceFixture.Tts<double>();
        string system = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens.";
        // web_demo.py: f"<|system|>\n{system_prompt}" + f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n", then the
        // tokenizer's [gMASK]<sop> prefix.
        var expected = new[] { model.TokenizeText("[gMASK]<sop>") }.SelectMany(x => x)
            .Concat(model.TokenizeText($"<|system|>\n{system}<|user|>\nhello there<|assistant|>streaming_transcription\n")).ToArray();
        Assert.Equal(expected, model.ChatPrompt("hello there", null));

        var speech = new[] { 3, 0, 11 };
        string speechSystem = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. ";
        var withSpeech = model.TokenizeText("[gMASK]<sop>")
            .Concat(model.TokenizeText($"<|system|>\n{speechSystem}<|user|>\n<|begin_of_audio|><|audio_3|><|audio_0|><|audio_11|><|end_of_audio|><|assistant|>streaming_transcription\n")).ToArray();
        Assert.Equal(withSpeech, model.ChatPrompt(null, speech));
    }

    [Fact(Timeout = 120000)]
    public async Task StreamingThoughts_Interleave13And26_AndSplitTheTargets()
    {
        await Task.Yield();
        var model = GLM4VoiceFixture.Tts<double>();
        var text = Enumerable.Range(0, 30).Select(i => 97 + i % 26).ToArray();
        var speech = Enumerable.Range(0, 60).Select(i => model.AudioOffset + i % GLM4VoiceFixture.Codes).ToArray();
        var layout = model.Interleave(text, speech);
        int[] runs = Runs(layout.Select(id => id >= model.AudioOffset).ToArray());
        // 13 text, 26 speech, 13 text, 26 speech, the last 4 text, the last 8 speech.
        Assert.Equal(new[] { 13, 26, 13, 26, 4, 8 }, runs);

        var turn = new GlmVoiceDialogueTurn { UserText = "hi", ResponseText = new string('a', 30), ResponseSpeechTokens = Enumerable.Range(0, 60).Select(i => i % GLM4VoiceFixture.Codes).ToArray() };
        var (ids, textTargets, speechTargets) = model.DialogueSequence(turn);
        Assert.Equal(30, textTargets.Count(t => t));
        Assert.Equal(61, speechTargets.Count(t => t));                    // the speech and the <|user|> that ends the turn
        Assert.Equal(model.TokenizeText("<|user|>")[0], ids[ids.Length - 1]);
        Assert.True(speechTargets[ids.Length - 1]);
        Assert.DoesNotContain(Enumerable.Range(0, ids.Length), i => textTargets[i] && speechTargets[i]);
    }

    private static int[] Runs(bool[] flags)
    {
        var runs = new System.Collections.Generic.List<int>();
        for (int i = 0; i < flags.Length;)
        {
            int j = i;
            while (j < flags.Length && flags[j] == flags[i]) j++;
            runs.Add(j - i);
            i = j;
        }
        return runs.ToArray();
    }

    [Fact(Timeout = 120000)]
    public async Task TopP_KeepsTheSmallestSetReachingP()
    {
        await Task.Yield();
        // Probabilities 0.5, 0.3, 0.15, 0.05, ascending cumulative 0.05, 0.2, 0.5, 1: top-p 0.75 drops those at most 0.25
        // (keeping the first two), 0.81 those at most 0.19 (keeping three).
        var logits = new[] { Math.Log(0.5), Math.Log(0.3), Math.Log(0.15), Math.Log(0.05) };
        var random = new Random(3);
        var seen = Enumerable.Range(0, 400).Select(_ => GLM4Voice<double>.Sample(logits, 1.0, 0.75, null, random)).Distinct().OrderBy(i => i).ToArray();
        Assert.Equal(new[] { 0, 1 }, seen);
        seen = Enumerable.Range(0, 400).Select(_ => GLM4Voice<double>.Sample(logits, 1.0, 0.81, null, random)).Distinct().OrderBy(i => i).ToArray();
        Assert.Equal(new[] { 0, 1, 2 }, seen);
        Assert.Equal(2, GLM4Voice<double>.Sample(logits, 1.0, 0.8, id => id >= 2, random));
    }

    [Fact(Timeout = 300000)]
    public async Task SynthesizeAndChat_ProduceSpeech()
    {
        await Task.Yield();
        var model = GLM4VoiceFixture.Tts<double>();
        var audio = model.Synthesize("hi");
        Assert.True(audio.Length > 0);
        var reply = model.Chat("hello");
        Assert.All(reply.SpeechTokens, c => Assert.InRange(c, 0, GLM4VoiceFixture.Codes - 1));
        var spoken = model.Chat(GLM4VoiceFixture.Tones<double>(1.0, 16000));
        Assert.Equal(spoken.SpeechTokens.Length == 0, spoken.Audio.Length == 0);

        var options = GLM4VoiceFixture.Options();
        options.TtsFormat = GlmVoiceTtsFormat.InterleavedBase;
        var interleaved = GLM4VoiceFixture.Tts<double>(options).Synthesize("hi");
        Assert.All(interleaved.ToArray(), v => Assert.False(double.IsNaN(v)));
        options = GLM4VoiceFixture.Options();
        options.Generation = GlmVoiceGeneration.TextFirst;
        var textFirst = GLM4VoiceFixture.Tts<double>(options);
        var textFirstReply = textFirst.Chat("hello");
        Assert.Equal(textFirstReply.SpeechTokens.Length > 0, textFirstReply.Audio.Length > 0);
    }

    [Fact(Timeout = 600000)]
    public async Task EachTraining_LowersItsObjective()
    {
        await Task.Yield();
        var options = GLM4VoiceFixture.Options();
        options.IncludeSpeechTokenizerTraining = true;
        options.LearningRate = 1e-2;
        options.Stage = GlmVoiceTrainingStage.FineTuning;
        options.FineTuningDropout = 0.0;
        var model = GLM4VoiceFixture.Tts<double>(options);

        // Fine-tuning: the split turn's two losses fall over repeated steps.
        var turn = new GlmVoiceDialogueTurn { UserText = "hi", ResponseText = "ok", ResponseSpeechTokens = new[] { 1, 5, 7, 2, 9 } };
        double first = model.TrainDialogue(turn);
        double last = first;
        for (int i = 0; i < 8; i++) last = model.TrainDialogue(turn);
        Assert.True(last < first, $"fine-tuning: {first} then {last}");
        Assert.True(double.IsNaN(model.TrainDialogue(turn, epoch: 25)));

        // The speech tokenizer: the ASR plus commitment loss falls, and the EMA codebook moves.
        var recording = GLM4VoiceFixture.Tones<double>(0.6, 16000);
        var labels = new[] { 5, 9, 3, 7, 2 };
        var before = model.SpeechTokenizer.EmaCount.ToArray();
        double tokenizerFirst = Convert.ToDouble(model.TrainSpeechTokenizer(recording, labels));
        Assert.NotEqual(before, model.SpeechTokenizer.EmaCount);
        double tokenizerLast = tokenizerFirst;
        for (int i = 0; i < 6; i++) tokenizerLast = Convert.ToDouble(model.TrainSpeechTokenizer(recording, labels));
        Assert.True(tokenizerLast < tokenizerFirst, $"speech tokenizer: {tokenizerFirst} then {tokenizerLast}");

        // The decoder's flow: a step on tokens and their audio, the truncated sample included.
        var speech = GLM4VoiceFixture.Tones<double>(1.0, 22050);
        var tokens = Enumerable.Range(0, 12).Select(i => i % GLM4VoiceFixture.Codes).ToArray();
        double loss = Convert.ToDouble(model.TrainSpeechDecoder(tokens, speech));
        Assert.True(loss > 0 && !double.IsNaN(loss));
    }

    [Fact(Timeout = 300000)]
    public async Task TtsObjective_FallsWithTraining()
    {
        await Task.Yield();
        var model = GLM4VoiceFixture.Tts<double>();
        var sample = new TtsTrainingSample<double>
        {
            Tokens = AiDotNet.TextToSpeech.Latest.IndexTtsOps.Ids<double>(model.TokenizeText("hi")),
            Audio = GLM4VoiceFixture.Tones<double>(1.0, 22050),
        };
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 6; i++) model.Train(sample);
        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < before, $"{before} then {after}");
    }
}
