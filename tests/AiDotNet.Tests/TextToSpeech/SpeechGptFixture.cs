using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// A tiny SpeechGPT ready to run, every part shrunk:
/// <list type="bullet">
/// <item>Tokenizer: SpeechT5's 81-character SentencePiece model (the library's embedded <c>spm_char.model</c>) with 12
/// unit tokens and the release's six special tokens added after it.</item>
/// <item>Language model: a 16-wide, 2-layer LLaMA.</item>
/// <item>Unit extractor: a one-layer, 16-wide HuBERT, its centroids spread over its own features.</item>
/// <item>Vocoder: a 32-wide multi-speaker unit HiFi-GAN with 4 speakers and the paper's 320-sample hop, so that even two
/// units are a recording mHuBERT can read.</item>
/// </list>
/// </summary>
internal static class SpeechGptFixture
{
    public const int Units = 12;
    private const int Pieces = 81;
    private static readonly string[] Specials = { "<sosp>", "<eosp>", "[Human]", "[SpeechGPT]", "<eoh>", "<eos>" };

    public static int VocabularySize => Pieces + Units + Specials.Length;

    private static readonly Lazy<string> TokenizerDirectory = new(() =>
    {
        string directory = Path.Combine(Path.GetTempPath(), $"aidotnet_speechgpt_tokenizer_{System.Diagnostics.Process.GetCurrentProcess().Id}");
        Directory.CreateDirectory(directory);
        const string name = "AiDotNet.TextToSpeech.MultiModal.speecht5_spm_char.model.gz";
        using (var stream = typeof(SpeechT5<>).Assembly.GetManifestResourceStream(name) ?? throw new InvalidDataException(name))
        using (var gzip = new GZipStream(stream, CompressionMode.Decompress))
        using (var file = File.Create(Path.Combine(directory, "tokenizer.model")))
            gzip.CopyTo(file);
        var added = new JObject();
        for (int u = 0; u < Units; u++) added[$"<{u}>"] = Pieces + u;
        for (int i = 0; i < Specials.Length; i++) added[Specials[i]] = Pieces + Units + i;
        File.WriteAllText(Path.Combine(directory, "added_tokens.json"), added.ToString());
        return directory;
    });

    public static SpeechGPTOptions Options() => new()
    {
        TokenizerDirectory = TokenizerDirectory.Value,
        HiddenDim = 16, Layers = 2, NumHeads = 2, Intermediate = 32, VocabSize = VocabularySize, MaxTextLength = 4096,
        Units = Units, HubertWidth = 16, HubertLayer = 1, HubertHeads = 2, HubertIntermediate = 32, HubertConvChannels = 16,
        VocoderSpeakers = 4, VocoderEmbeddingDim = 8, VocoderChannels = 32, VocoderUpsampleRates = new[] { 5, 4, 4, 4 },
        VocoderUpsampleKernels = new[] { 11, 8, 8, 8 }, VocoderSegmentSize = 1280,
        MaxNewTokens = 24, MinNewTokens = 2, LearningRate = 1e-2, TotalSteps = 1000, WarmupFraction = 0,
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

    private static NeuralNetworkArchitecture<T> Architecture<T>() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);

    public static SpeechGPT<T> Tts<T>(SpeechGPTOptions? options = null)
    {
        var model = new SpeechGPT<T>(Architecture<T>(), options ?? Options());
        SpreadCentroids(model.Core.Units);
        return model;
    }

    public static AiDotNet.SpeechRecognition.LLMIntegrated.SpeechGPT<T> Asr<T>(SpeechGPTOptions? options = null)
    {
        var model = new AiDotNet.SpeechRecognition.LLMIntegrated.SpeechGPT<T>(Architecture<T>(), options ?? Options());
        SpreadCentroids(model.Core.Units);
        return model;
    }

    /// <summary>The k-means centroids as a fitted model's are: the extractor's own features at evenly spaced frames of a
    /// second of noisy tones (all zero, every frame would be one unit).</summary>
    private static void SpreadCentroids<T>(AiDotNet.Audio.Foundations.HubertUnitExtractor<T> extractor)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var random = new Random(7);
        var audio = new Tensor<T>(new[] { 16000 }, new Vector<T>(Enumerable.Range(0, 16000)
            .Select(i => ops.FromDouble(0.3 * (2 * random.NextDouble() - 1) + 0.3 * Math.Sin(2 * Math.PI * (150 + i / 40.0) * i / 16000))).ToArray()));
        Tensor<T> features;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>()) features = extractor.Network.Forward(audio, extractor.Layer)[extractor.Layer];
        int frames = features.Shape[0], width = features.Shape[1];
        var centroids = new double[Units, width];
        for (int k = 0; k < Units; k++)
            for (int i = 0; i < width; i++) centroids[k, i] = ops.ToDouble(features[k * frames / Units, i]);
        extractor.Centroids = centroids;
    }
}
