using System.Text.Json;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class MetaVoiceGenerationReferenceTests
{
    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)]
    [InlineData(5)] [InlineData(6)] [InlineData(7)] [InlineData(8)]
    public void UnchangedAuthorGeneration_MatchesEveryProbabilityDrawTokenAndDecoderInput(int index)
    {
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "metavoice_generation_reference.json")));
        var reference = document.RootElement.GetProperty("cases")[index];
        string stage = reference.GetProperty("stage").GetString()!; bool causal = stage == "first";
        var saved = document.RootElement.GetProperty("models").GetProperty(stage);
        var cfg = saved.GetProperty("config");
        static double[] Values(JsonElement x) => x.EnumerateArray().Select(v => v.GetDouble()).ToArray();
        static int[] Integers(JsonElement x) => x.EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var model = new MetaVoiceGpt<double>(AiDotNetEngine.Current, new MetaVoiceGptConfiguration
        {
            Dimension = 8, Layers = 2, Heads = 2, Context = 16, Causal = causal, RmsNorm = causal, SwiGlu = causal,
            SwiGluMultiple = 4, SpeakerDimension = 3, SpeakerDropout = causal ? .1 : 0,
            InputVocabularies = Integers(cfg.GetProperty("vocab_sizes")),
            OutputVocabularies = causal ? null : Integers(cfg.GetProperty("target_vocab_sizes"))
        });
        var parameters = saved.GetProperty("parameters");
        model.LoadTorch((name, shape) =>
        {
            if (!parameters.TryGetProperty(name, out var entry)) entry = parameters.GetProperty(saved.GetProperty("aliases").GetProperty(name).GetString()!);
            Assert.Equal(shape, Integers(entry.GetProperty("shape")));
            return Values(entry.GetProperty("values"));
        });
        var speaker = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new[] { .2, -.4, .7 }));
        int? topK = reference.GetProperty("top_k").ValueKind == JsonValueKind.Null ? null : reference.GetProperty("top_k").GetInt32();
        void Close(double[] actual, JsonElement expected)
        {
            var values = Values(expected); Assert.Equal(values.Length, actual.Length);
            double error = actual.Zip(values, (a, b) => Math.Abs(a - b)).Max();
            Assert.True(error < 1e-10, $"MetaVoice generation probability error {error:R} in case {index}.");
        }
        using var noGrad = new NoGradScope<double>();
        if (causal)
        {
            double? topP = reference.GetProperty("top_p").ValueKind == JsonValueKind.Null ? null : reference.GetProperty("top_p").GetDouble();
            var guidance = reference.GetProperty("guidance");
            (double Speaker, double Prompt)? scales = guidance.ValueKind == JsonValueKind.Null ? null : (guidance[0].GetDouble(), guidance[1].GetDouble());
            int draws = 0;
            var result = MetaVoiceGeneration.FirstStage(Integers(reference.GetProperty("prompt")), 4, 16,
                (prefix, conditioned) => model.Forward(new[] { prefix }, conditioned ? speaker : null, lastOnly: true).Logits[0].ToArray(),
                step => { Assert.Equal(draws++, step); return reference.GetProperty("calls")[step].GetProperty("uniform").GetDouble(); },
                reference.GetProperty("conditioned").GetBoolean(), .8, topK, topP, scales, 16, 25,
                (step, probabilities) => Close(probabilities, reference.GetProperty("calls")[step].GetProperty("probabilities")));
            Assert.Equal(Integers(reference.GetProperty("output")), result);
            Assert.Equal(reference.GetProperty("calls").GetArrayLength(), draws);
        }
        else
        {
            int[][] Rows(string key) => reference.GetProperty(key).EnumerateArray().Select(Integers).ToArray();
            var input = MetaVoiceGeneration.SecondStageInput(Integers(reference.GetProperty("text")), Rows("first_audio"), 16, 8);
            var expectedInput = Rows("input");
            for (int i = 0; i < 2; i++) Assert.Equal(expectedInput[i], input[i]);
            var logits = model.Forward(input, speaker).Logits;
            int draws = 0;
            var generated = MetaVoiceGeneration.NonCausal(6, 16,
                (head, frame) => logits[head].ToArray().Skip(frame * 9).Take(9).ToArray(),
                (head, frame) => { draws++; return reference.GetProperty("calls")[head].GetProperty("uniforms")[frame].GetDouble(); },
                .8, topK, (head, frame, probabilities) => Close(probabilities,
                    reference.GetProperty("calls")[head].GetProperty("probabilities")[frame]));
            Assert.Equal(96, draws);
            var combined = input.Concat(generated).ToArray(); var expected = Rows("output");
            for (int i = 0; i < 8; i++) Assert.Equal(expected[i], combined[i]);
            var decoded = MetaVoiceTokenAdapters.Decode(combined, false, 8);
            Assert.Equal(Integers(reference.GetProperty("extracted_text")), decoded.Text);
            var expectedAudio = Rows("extracted_audio");
            for (int i = 0; i < 8; i++) Assert.Equal(expectedAudio[i], decoded.Audio[i]);
        }
    }
}
