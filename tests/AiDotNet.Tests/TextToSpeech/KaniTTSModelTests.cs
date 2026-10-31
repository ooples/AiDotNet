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
/// KaniTTS against the release's own code (tools/reference-data/kanitts_reference.py) on a tiny model with a small
/// byte-level BPE tokenizer in LFM2's form: kani-tts 0.0.4's prompt and speech span, greedy generation through
/// transformers' Lfm2ForCausalLM, the basic example's streaming extraction (with a window it must skip), and the
/// fine-tuning pipeline's training example and its loss, with NeMo's NanoCodec encoding the recording.
/// </summary>
public class KaniTTSModelTests
{
    private static JObject Reference()
    {
        const string fileName = "kanitts_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    private static int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();

    private static KaniTTS<double> Model(JObject f, bool lenient = false)
    {
        var lm = f["lm_config"]!;
        var codec = f["codec_config"]!;
        var model = new KaniTTS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            new KaniTTSOptions
            {
                TextVocabulary = (int)f["text_vocabulary"]!, VocabSize = (int)lm["vocab_size"]!, HiddenDim = (int)lm["hidden_size"]!,
                AttentionLayers = lm["layer_types"]!.Select(t => (string)t! == "full_attention").ToArray(),
                Heads = (int)lm["num_attention_heads"]!, KeyValueHeads = (int)lm["num_key_value_heads"]!, Intermediate = (int)lm["intermediate_size"]!,
                FeedForwardMultipleOf = (int)lm["block_multiple_of"]!, NormEpsilon = (double)lm["norm_eps"]!, RopeTheta = (double)lm["rope_theta"]!,
                CodecEncoderChannels = (int)codec["encoder"]!["base_channels"]!, CodecDownsampleRates = Ints(codec["encoder"]!["down_sample_rates"]!),
                CodecEncodedDim = (int)codec["encoder"]!["encoded_dim"]!, CodecDecoderChannels = (int)codec["decoder"]!["base_channels"]!,
                CodecUpsampleRates = Ints(codec["decoder"]!["up_sample_rates"]!), CodecLevels = Ints(codec["quantizer"]!["num_levels_per_group"]!),
                Temperature = 0, MaxNewTokens = 20, UseLora = false, LenientSpeechSpans = lenient, TokenizerJson = (string)f["tokenizer"]!,
                StreamingChunkFrames = (int)f["chunk"]!, StreamingLookbackFrames = (int)f["lookback"]!,
            });
        var state = (JObject)f["lm_state"]!;
        model.LanguageModel.LoadTorchWeights((name, shape) => state[name]!.Select(v => (double)v).ToArray());
        var codecState = (JObject)f["codec_state"]!;
        model.LoadCodec(name => codecState[name]?.Select(v => (double)v).ToArray(), codecState.Properties().Select(p => p.Name));
        return model;
    }

    private static void Close(JToken expected, Tensor<double> actual, string what)
    {
        var e = expected.Select(v => (double)v).ToArray();
        Assert.Equal(e.Length, actual.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - actual[i]) <= 1e-10 * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {actual[i]}");
    }

    [Fact(Timeout = 600000)]
    public async Task PromptAndGreedyGeneration_MatchTheRelease()
    {
        await Task.Yield();
        var f = Reference();
        var model = Model(f);
        string text = (string)f["text"]!;
        var prompt = model.PromptIds(text, (string)f["speaker"]!);
        Assert.Equal(Ints(f["prompt"]!), prompt);
        Assert.Equal(Ints(f["plain_prompt"]!), model.PromptIds(text, null));
        Assert.Equal(Ints(f["generated"]!), model.Generate(prompt, new Random(0)));
    }

    [Fact(Timeout = 600000)]
    public async Task SpanAndStreamedAudio_MatchTheRelease()
    {
        await Task.Yield();
        var f = Reference();
        using var _ = new NoGradScope<double>();
        var strict = Model(f);
        Close(f["span_audio"]!, strict.AudioOf(Ints(f["sequence"]!)), "span audio");
        // The package rejects a span holding a token that is not a code; the streaming writer skips its windows.
        Assert.Throws<InvalidOperationException>(() => strict.AudioOf(Ints(f["streamed_sequence"]!)));
        Close(f["streamed_audio"]!, Model(f, lenient: true).AudioOf(Ints(f["streamed_sequence"]!)), "streamed audio");
    }

    [Fact(Timeout = 600000)]
    public async Task TrainingExampleAndLoss_MatchThePipeline()
    {
        await Task.Yield();
        var f = Reference();
        var model = Model(f);
        var audio = f["audio"]!.Select(v => (double)v).ToArray();
        var encoded = model.EncodeToTokens(new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio)));
        var expected = f["encoded_codes"]!.Select(Ints).ToArray();
        for (int q = 0; q < expected.Length; q++)
            Assert.Equal(expected[q], Enumerable.Range(0, encoded.Shape[1]).Select(t => (int)encoded[q, t]));

        var rows = f["codes"]!.Select(Ints).ToArray();
        var codes = new int[rows.Length, rows[0].Length];
        for (int q = 0; q < rows.Length; q++)
            for (int t = 0; t < rows[q].Length; t++) codes[q, t] = rows[q][t];
        var ids = model.ExampleIds((string)f["text"]!, (string)f["speaker"]!, codes);
        Assert.Equal(Ints(f["training_ids"]!), ids);
        double loss;
        using (new NoGradScope<double>()) loss = model.Loss(new[] { new KaniTTSBase<double>.KaniExample(ids, ids, null) })[0];
        double reference = (double)f["loss"]!;
        Assert.True(Math.Abs(reference - loss) <= 1e-10 * Math.Abs(reference), $"loss {reference:R} vs {loss:R}");
    }
}
