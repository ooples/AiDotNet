using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// KaniTTS 2 against the release's own code (tools/reference-data/kanitts2_reference.py) on a tiny model: the prompt with
/// a language tag, the training sequence, the logits and loss of a training sequence with an inserted speaker embedding at
/// frame-level positions (kani-tts-2-pretrain's model and data collator), the gradients of the learnable rotary scales
/// and the speaker projection, and greedy generation with the speaker at audio steps 1 and 0.5 (KaniTTS2ForCausalLM's
/// prepare_inputs_for_generation).
/// </summary>
public class KaniTTS2ModelTests
{
    private static JObject Reference()
    {
        const string fileName = "kanitts2_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static string TokenizerJson()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", "kanitts_reference.json");
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "kanitts_reference.json");
        return (string)JObject.Parse(File.ReadAllText(path))["tokenizer"]!;
    }

    private static int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();

    private static KaniTTS2<double> Model(JObject f, double audioStep = 1.0)
    {
        var c = f["config"]!;
        var model = new KaniTTS2<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            new KaniTTS2Options
            {
                TextVocabulary = (int)f["text_vocabulary"]!, VocabSize = (int)c["vocab_size"]!, HiddenDim = (int)c["hidden_size"]!,
                AttentionLayers = c["layer_types"]!.Select(t => (string)t! == "full_attention").ToArray(),
                Heads = (int)c["num_attention_heads"]!, KeyValueHeads = (int)c["num_key_value_heads"]!, Intermediate = (int)c["intermediate_size"]!,
                FeedForwardMultipleOf = (int)c["block_multiple_of"]!, NormEpsilon = (double)c["norm_eps"]!, RopeTheta = (double)c["rope_theta"]!,
                SpeakerEmbeddingDim = (int)f["speaker_dim"]!, AudioStep = audioStep,
                // Four codebooks of 8·8 = 64 codes from a tiny NanoCodec; a tiny WavLM-Large-layout embedder (unused here).
                CodecLevels = new[] { 8, 8 }, CodecEncodedDim = 8, CodecEncoderChannels = 2, CodecDownsampleRates = new[] { 2, 3 },
                CodecDecoderChannels = 8, CodecUpsampleRates = new[] { 3, 2 },
                EmbedderConvDims = Enumerable.Repeat(8, 7).ToArray(), EmbedderWidth = 16, EmbedderLayers = 1, EmbedderHeads = 2,
                EmbedderIntermediate = 32, EmbedderPositionalKernel = 16, EmbedderPositionalGroups = 2, EmbedderHeadWidth = 6,
                Temperature = 0, MaxNewTokens = 40, TokenizerJson = TokenizerJson(),
            });
        var state = (JObject)f["state"]!;
        model.LoadLanguageModel(name => state[name]?.Select(v => (double)v).ToArray());
        return model;
    }

    private static Tensor<double> Speaker(JObject f)
    {
        var values = f["speaker"]!.Select(v => (double)v).ToArray();
        return new Tensor<double>(new[] { 1, values.Length }, new Vector<double>(values));
    }

    [Fact(Timeout = 600000)]
    public async Task PromptAndTrainingSequence_MatchTheRelease()
    {
        await Task.Yield();
        var f = Reference();
        var model = Model(f);
        string text = (string)f["text"]!, tag = (string)f["tag"]!;
        Assert.Equal(Ints(f["prompt"]!), model.PromptIds(text, tag));
        var rows = f["codes"]!.Select(Ints).ToArray();
        var codes = new int[rows.Length, rows[0].Length];
        for (int q = 0; q < rows.Length; q++)
            for (int t = 0; t < rows[q].Length; t++) codes[q, t] = rows[q][t];
        Assert.Equal(Ints(f["training_ids"]!), model.ExampleIds(text, tag, codes));
    }

    [Fact(Timeout = 600000)]
    public async Task SpeakerLogitsLossAndGradients_MatchThePretraining()
    {
        await Task.Yield();
        var f = Reference();
        var model = Model(f);
        var ids = Ints(f["training_ids"]!);
        var speaker = Speaker(f);
        var lm = model.LanguageModel;
        var basis = model.FramePositions(ids);
        var positions = new[] { basis[0], basis[0] + 1 }.Concat(basis.Skip(1).Select(p => p + 1)).ToArray();
        var row = new int[1, ids.Length];
        for (int i = 0; i < ids.Length; i++) row[0, i] = ids[i];
        using (new NoGradScope<double>())
        {
            var logits = lm.Forward(lm.EmbedWithSpeaker(row, speaker), null, false, positions).Logits;
            var expected = f["logits"]!.Select(v => (double)v).ToArray();
            Assert.Equal(expected.Length, logits.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - logits[i]) <= 1e-10 * Math.Max(1, Math.Abs(expected[i])), $"logits[{i}]: {expected[i]} vs {logits[i]}");
        }

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(
            lm.Layers.Where(l => l.RopeAlpha is not null).Select(l => (AiDotNet.Interfaces.ILayer<double>)l.RopeAlpha!)
                .Append(lm.SpeakerProjection!).ToList(), -1);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        Tensor<double> loss;
        using (var tape = new GradientTape<double>())
        {
            loss = model.Loss(new[] { new KaniTTSBase<double>.KaniExample(ids, ids, speaker) });
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        double reference = (double)f["loss"]!;
        Assert.True(Math.Abs(reference - loss[0]) <= 1e-10 * reference, $"loss {reference:R} vs {loss[0]:R}");
        var gradients = (JObject)f["gradients"]!;
        for (int i = 0; i < lm.Layers.Count; i++)
        {
            if (lm.Layers[i].RopeAlpha is not { } alpha) continue;
            double e = (double)gradients[$"model.learnable_rope_layers.{i}.alpha_weight"]![0]!, a = grads[alpha.Weights][0];
            Assert.True(Math.Abs(e - a) <= 1e-9 * Math.Max(1e-3, Math.Abs(e)), $"alpha {i}: {e} vs {a}");
        }
        var projection = lm.SpeakerProjection!.Weights;
        var expectedProjection = gradients["model.speaker_emb_projection.weight"]!.Select(v => (double)v).ToArray();
        int inputs = projection.Shape[0], outputs = projection.Shape[1];
        for (int o = 0; o < outputs; o++)
            for (int i = 0; i < inputs; i++)
            {
                double e = expectedProjection[o * inputs + i], a = grads[projection][i * outputs + o];
                Assert.True(Math.Abs(e - a) <= 1e-9 * Math.Max(1e-3, Math.Abs(e)), $"projection[{o},{i}]: {e} vs {a}");
            }
    }

    [Theory(Timeout = 600000)]
    [InlineData("1.0", 1.0)]
    [InlineData("0.5", 0.5)]
    public async Task GreedyGenerationWithSpeaker_MatchesTheRelease(string key, double audioStep)
    {
        await Task.Yield();
        var f = Reference();
        var model = Model(f, audioStep);
        var prompt = Ints(f["prompt"]!);
        Assert.Equal(Ints(f["generated"]![key]!), model.Generate(prompt, new Random(0), Speaker(f)));
    }
}
