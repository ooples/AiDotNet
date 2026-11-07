using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.MultiModal;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// AudioPaLM's sequence handling on a tiny model (<see cref="AudioPaLMFixture"/>): the joint vocabulary and its zeroed
/// audio rows (§3.2), task prefixes (§3.4), the loss masked to the outputs with PaLM's z-loss (§3.6), training of the
/// TTS task and of others, constrained generation, the detokenizers, and the speech-recognition task variant.
/// </summary>
public class AudioPaLMTaskTests
{
    [Fact(Timeout = 300000)]
    public async Task TaskPrefix_IsStartTagThenInput()
    {
        await Task.Yield();
        var model = AudioPaLMFixture.Tts<double>();
        var tag = model.TokenizeText("[TTS English]");
        var prefix = model.TaskPrefix("[TTS English]", new[] { 5, 6 });
        Assert.Equal(new[] { 0 }.Concat(tag).Concat(new[] { 5, 6 }).ToArray(), prefix);
        Assert.Equal(81, model.AudioOffset);
        var codes = model.AudioTokensOf(AudioPaLMFixture.Tones<double>(0.4));
        Assert.InRange(codes.Length, 9, 10);   // 25 Hz (w2v-BERT's two stride-2 convolutions trim the edges)
        Assert.All(codes, c => Assert.InRange(c, 0, 7));
    }

    [Fact(Timeout = 300000)]
    public async Task Loss_IsTheOutputsCrossEntropyPlusZLoss()
    {
        await Task.Yield();
        var model = AudioPaLMFixture.Tts<double>();
        var decoder = model.Decoder;
        var prefix = model.TaskPrefix("[TTS English]", model.TokenizeText("hi"));
        var output = new[] { 81 + 3, 81 + 1, 81 + 6 };
        double loss;
        using (new NoGradScope<double>()) loss = model.Core.Loss(prefix, output, false, null)[0];

        // Recomputed independently: logits of the whole sequence, softmax over all 89 ids, targets output then </s>.
        var sequence = prefix.Concat(output).Concat(new[] { 2 }).ToArray();
        Tensor<double> logits;
        using (new NoGradScope<double>()) logits = decoder.Logits(decoder.Forward(decoder.Embed(sequence.Take(sequence.Length - 1).ToArray())));
        double expected = 0;
        for (int i = prefix.Length - 1; i < sequence.Length - 1; i++)
        {
            double max = double.NegativeInfinity, sum = 0;
            for (int v = 0; v < 89; v++) max = Math.Max(max, logits[i, v]);
            for (int v = 0; v < 89; v++) sum += Math.Exp(logits[i, v] - max);
            double logZ = max + Math.Log(sum);
            expected += logZ - logits[i, sequence[i + 1]] + 1e-4 * logZ * logZ;
        }
        expected /= output.Length + 1;
        Assert.True(Math.Abs(expected - loss) <= 1e-12 * Math.Max(1, Math.Abs(expected)), $"{expected} vs {loss}");
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheTtsObjective()
    {
        await Task.Yield();
        var model = AudioPaLMFixture.Tts<double>();
        var sample = new TtsTrainingSample<double>
        {
            Tokens = AiDotNet.TextToSpeech.Latest.IndexTtsOps.Ids<double>(model.TokenizeText("hello")),
            Audio = AudioPaLMFixture.Tones<double>(0.4),
        };
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 8; i++) model.Train(sample);
        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < before, $"{before} -> {after}");
    }

    [Fact(Timeout = 300000)]
    public async Task Generation_StaysInTheTasksVocabulary()
    {
        await Task.Yield();
        var model = AudioPaLMFixture.Tts<double>();
        var prefix = model.TaskPrefix("[TTS English]", model.TokenizeText("hello"));
        var audio = model.Generate(prefix, audioOutput: true);
        Assert.NotEmpty(audio);
        Assert.All(audio, id => Assert.InRange(id, 81, 88));
        var text = model.Generate(prefix, audioOutput: false);
        Assert.NotEmpty(text);
        Assert.All(text, id => Assert.True(id < 81 && id != 2 && id != 0, $"{id}"));
    }

    [Theory(Timeout = 300000)]
    [InlineData(AudioPalmDetokenizer.SoundStorm)]
    [InlineData(AudioPalmDetokenizer.AudioLM)]
    public async Task Tasks_RunEndToEnd(AudioPalmDetokenizer detokenizer)
    {
        await Task.Yield();
        var options = AudioPaLMFixture.Options();
        options.Detokenizer = detokenizer;
        var model = AudioPaLMFixture.Tts<double>(options);
        var speech = model.Synthesize("hello");
        Assert.True(speech.Length > 0);
        Assert.All(speech.ToArray(), v => Assert.False(double.IsNaN(v)));
        var recording = AudioPaLMFixture.Tones<double>(0.4);
        Assert.NotNull(model.Transcribe(recording, "English"));
        Assert.NotNull(model.TranslateSpeechToText(recording, "English", "French"));
        Assert.True(model.TranslateSpeech(recording, "English", "French").Length > 0);
    }

    [Fact(Timeout = 300000)]
    public async Task TrainTask_TrainsAnyTask()
    {
        await Task.Yield();
        var model = AudioPaLMFixture.Tts<double>();
        var input = model.AudioTokensOf(AudioPaLMFixture.Tones<double>(0.4)).Select(c => model.AudioOffset + c).ToArray();
        var output = model.TokenizeText("hello");
        var prefix = model.TaskPrefix("[ASR English]", input);
        double before, after;
        using (new NoGradScope<double>()) before = model.Core.Loss(prefix, output, false, null)[0];
        for (int i = 0; i < 8; i++) model.TrainTask("[ASR English]", input, output);
        using (new NoGradScope<double>()) after = model.Core.Loss(prefix, output, false, null)[0];
        Assert.True(after < before, $"{before} -> {after}");
    }

    [Fact(Timeout = 300000)]
    public async Task SpeechRecognitionVariant_TrainsAndTranscribes()
    {
        await Task.Yield();
        var model = AudioPaLMFixture.Asr<double>();
        var recording = AudioPaLMFixture.Tones<double>(0.4);
        var transcript = model.TokenizeText("hello");
        var target = AiDotNet.TextToSpeech.Latest.IndexTtsOps.Ids<double>(transcript);
        var prefix = model.Core.TaskPrefix("[ASR English]", model.Core.AudioIds(model.Core.AudioTokensOf(recording)));
        double before, after;
        using (new NoGradScope<double>()) before = model.Core.Loss(prefix, transcript, false, null)[0];
        for (int i = 0; i < 12; i++) model.Train(recording, target);
        using (new NoGradScope<double>()) after = model.Core.Loss(prefix, transcript, false, null)[0];
        Assert.True(after < before, $"{before} -> {after}");
        var result = model.Transcribe(recording, "en");
        Assert.NotNull(result.Text);
        Assert.InRange(result.Confidence, 0.0, 1.0);
    }
}
