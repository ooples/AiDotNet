using System;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The generated suite's KaniTTS: kanitts_reference.py's tiny configuration (its small tokenizer, a width-16 LFM2 of
/// conv, attention, conv, attention layers, and a NanoCodec hopping 6 samples with four codebooks of 9 codes) with
/// kanitts_speaking_fixture.py's language model, which answers every prompt with a speech span, so synthesis runs the
/// release's own extraction. Greedy, one sample per optimizer step; LoRA as the release trains.
/// </summary>
internal static class KaniTTSFixture
{
    private static JObject Load(string fileName)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static readonly Lazy<(JObject Reference, JObject Speaking)> Data =
        new(() => (Load("kanitts_reference.json"), Load("kanitts_speaking_lm.json")));

    public static KaniTTS<T> Create<T>()
    {
        var (f, speaking) = Data.Value;
        var lm = f["lm_config"]!;
        var codec = f["codec_config"]!;
        int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();
        var model = new KaniTTS<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            new KaniTTSOptions
            {
                TextVocabulary = (int)f["text_vocabulary"]!, VocabSize = (int)lm["vocab_size"]!, HiddenDim = (int)lm["hidden_size"]!,
                AttentionLayers = lm["layer_types"]!.Select(t => (string)t! == "full_attention").ToArray(),
                Heads = (int)lm["num_attention_heads"]!, KeyValueHeads = (int)lm["num_key_value_heads"]!, Intermediate = (int)lm["intermediate_size"]!,
                FeedForwardMultipleOf = (int)lm["block_multiple_of"]!, NormEpsilon = (double)lm["norm_eps"]!, RopeTheta = (double)lm["rope_theta"]!,
                CodecEncoderChannels = (int)codec["encoder"]!["base_channels"]!, CodecDownsampleRates = Ints(codec["encoder"]!["down_sample_rates"]!),
                CodecEncodedDim = (int)codec["encoder"]!["encoded_dim"]!, CodecDecoderChannels = (int)codec["decoder"]!["base_channels"]!,
                CodecUpsampleRates = Ints(codec["decoder"]!["up_sample_rates"]!), CodecLevels = Ints(codec["quantizer"]!["num_levels_per_group"]!),
                LoraRank = 4, LoraAlpha = 4, Temperature = 0, MaxNewTokens = 40, GradientAccumulationSteps = 1, TokenizerJson = (string)f["tokenizer"]!,
            });
        var state = (JObject)speaking["lm_state"]!;
        model.LoadLanguageModel(name => state[name]?.Select(v => (double)v).ToArray());
        return model;
    }
}
