using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// A tiny LLaMA-Omni ready to run, with every part shrunk:
/// <list type="bullet">
/// <item>Tokenizer: a byte-level BPE <c>tokenizer.json</c> of the 256 bytes (no merges), plus Llama 3's six header and
/// end tokens.</item>
/// <item>Language model: 16 wide, 2 layers.</item>
/// <item>Speech encoder: an 80-band Whisper encoder over a half-second window, 1 layer.</item>
/// <item>Speech decoder: 1 layer, 8 wide, 12 units, λ = 3.</item>
/// <item>Vocoder: a 16-wide unit HiFi-GAN.</item>
/// <item>Unit extractor: a one-layer, 16-wide HuBERT whose centroids are spread over its own features.</item>
/// </list>
/// </summary>
internal static class LlamaOmniFixture
{
    public const int Units = 12;

    private static readonly string[] Specials = { "<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|eom_id|>" };

    public static int VocabularySize => 256 + Specials.Length;

    // GPT-2's bytes_to_unicode: printable bytes map to themselves, the rest to 256 + n.
    private static Dictionary<int, char> ByteAlphabet()
    {
        var printable = Enumerable.Range('!', '~' - '!' + 1).Concat(Enumerable.Range('¡', '¬' - '¡' + 1)).Concat(Enumerable.Range('®', 'ÿ' - '®' + 1)).ToList();
        var map = new Dictionary<int, char>();
        int extra = 0;
        for (int b = 0; b < 256; b++) map[b] = printable.Contains(b) ? (char)b : (char)(256 + extra++);
        return map;
    }

    private static readonly Lazy<string> TokenizerPath = new(() =>
    {
        var alphabet = ByteAlphabet();
        var vocab = new JObject();
        for (int b = 0; b < 256; b++) vocab[alphabet[b].ToString()] = b;
        var added = new JArray(Specials.Select((s, i) => new JObject
        {
            ["id"] = 256 + i, ["content"] = s, ["single_word"] = false, ["lstrip"] = false, ["rstrip"] = false, ["normalized"] = false, ["special"] = true,
        }));
        var json = new JObject
        {
            ["added_tokens"] = added,
            ["normalizer"] = null,
            ["pre_tokenizer"] = new JObject { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["trim_offsets"] = true, ["use_regex"] = true },
            ["post_processor"] = new JObject { ["type"] = "ByteLevel", ["add_prefix_space"] = true, ["trim_offsets"] = false, ["use_regex"] = true },
            ["model"] = new JObject { ["type"] = "BPE", ["vocab"] = vocab, ["merges"] = new JArray(), ["ignore_merges"] = true },
        };
        string path = Path.Combine(Path.GetTempPath(), $"aidotnet_llamaomni_tokenizer_{System.Diagnostics.Process.GetCurrentProcess().Id}.json");
        File.WriteAllText(path, json.ToString());
        return path;
    });

    public static LlamaOmniOptions Options() => new()
    {
        TokenizerPath = TokenizerPath.Value,
        HiddenDim = 16, Layers = 2, NumHeads = 2, KeyValueHeads = 1, Intermediate = 32, VocabSize = VocabularySize, MaxTextLength = 4096,
        RopeOriginalMaxPositions = 32,
        EncoderMels = 80, EncoderContext = 25, EncoderWidth = 16, EncoderHeads = 2, EncoderLayers = 1, AdaptorHidden = 16,
        SpeechDecoderLayers = 1, SpeechDecoderWidth = 8, SpeechDecoderHeads = 2, SpeechDecoderIntermediate = 16, UpsampleFactor = 3, Units = Units,
        VocoderEmbeddingDim = 8, VocoderChannels = 16, VocoderUpsampleRates = new[] { 5, 4 }, VocoderUpsampleKernels = new[] { 11, 8 },
        HubertWidth = 16, HubertLayer = 1, HubertHeads = 2, HubertIntermediate = 32, HubertConvChannels = 16,
        MaxNewTokens = 12, SpeechDecoderLearningRate = 1e-2, LearningRate = 1e-2, TotalSteps = 1000, WarmupFraction = 0,
    };

    /// <summary>Two tones at 16 kHz, <paramref name="seconds"/> long.</summary>
    public static Tensor<T> Tones<T>(double seconds, double low = 180.0, double high = 470.0)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int samples = (int)(16000 * seconds);
        var audio = new T[samples];
        for (int i = 0; i < samples; i++)
            audio[i] = ops.FromDouble(0.4 * Math.Sin(2 * Math.PI * low * i / 16000) + 0.2 * Math.Sin(2 * Math.PI * high * i / 16000));
        return new Tensor<T>(new[] { samples }, new Vector<T>(audio));
    }

    public static LlamaOmni<T> Tts<T>(LlamaOmniOptions? options = null)
    {
        var model = new LlamaOmni<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            options ?? Options());
        SpreadCentroids(model);
        return model;
    }

    /// <summary>The k-means centroids as a fitted model's are: features of the extractor's own layer, at evenly spaced
    /// frames of a second of noisy tones (all zero, every frame would be one unit).</summary>
    private static void SpreadCentroids<T>(LlamaOmni<T> model)
    {
        var extractor = model.UnitExtractor;
        var random = new Random(7);
        var audio = new Tensor<T>(new[] { 16000 }, new Vector<T>(Enumerable.Range(0, 16000)
            .Select(i => MathHelper.GetNumericOperations<T>().FromDouble(0.3 * (2 * random.NextDouble() - 1) + 0.3 * Math.Sin(2 * Math.PI * (150 + i / 40.0) * i / 16000))).ToArray()));
        Tensor<T> features;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>()) features = extractor.Network.Forward(audio, extractor.Layer)[extractor.Layer];
        int frames = features.Shape[0], width = features.Shape[1];
        var centroids = new double[Units, width];
        var ops = MathHelper.GetNumericOperations<T>();
        for (int k = 0; k < Units; k++)
            for (int i = 0; i < width; i++) centroids[k, i] = ops.ToDouble(features[k * frames / Units, i]);
        extractor.Centroids = centroids;
    }
}
