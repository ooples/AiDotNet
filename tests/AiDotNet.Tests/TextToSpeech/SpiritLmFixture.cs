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
/// A tiny SpiritLM ready to run, every part shrunk:
/// <list type="bullet">
/// <item>Tokenizer: a 96-piece BPE SentencePiece model laid out as the release's (text pieces, then <c>[Text]</c>,
/// <c>[Speech]</c>, 13 HuBERT tokens, 4 pitch and 3 style tokens as user-defined pieces;
/// <c>tools/reference-data/spiritlm_tokenizer_reference.py</c>).</item>
/// <item>Language model: a 16-wide, 2-layer LLaMA.</item>
/// <item>Speech tokenizer: a one-layer, 16-wide 25 Hz mHuBERT and its linear quantizer over 12 units; for Expressive a
/// one-layer FCPE, an 8-wide pitch VQ-VAE and a one-layer wav2vec 2.0 style classifier.</item>
/// <item>Vocoder: a 32-wide unit HiFi-GAN with the paper's 640-sample hop, 4 speakers and 5 styles (Base) or the pitch and
/// style tokens (Expressive).</item>
/// </list>
/// </summary>
internal static class SpiritLmFixture
{
    public const int Units = 12;

    private static readonly Lazy<JObject> Reference = new(() =>
    {
        const string name = "spiritlm_tokenizer_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", name);
        return JObject.Parse(File.ReadAllText(path));
    });

    public static JObject TokenizerReference => Reference.Value;

    private static readonly Lazy<string> TokenizerPath = new(() =>
    {
        string path = Path.Combine(Path.GetTempPath(), $"aidotnet_spiritlm_tokenizer_{System.Diagnostics.Process.GetCurrentProcess().Id}.model");
        File.WriteAllBytes(path, Convert.FromBase64String((string)Reference.Value["tiny_model_base64"]!));
        return path;
    });

    public static int VocabularySize => (int)Reference.Value["tiny_pieces"]!;

    /// <summary>An NLLB-layout tokenizer directory for the tiny SONAR text encoder: the fixture's SentencePiece model and
    /// two language codes (ids pieces + 1 and + 2).</summary>
    private static readonly Lazy<string> SonarTokenizerDirectory = new(() =>
    {
        string directory = Path.Combine(Path.GetTempPath(), $"aidotnet_sonar_tokenizer_{System.Diagnostics.Process.GetCurrentProcess().Id}");
        Directory.CreateDirectory(directory);
        File.WriteAllBytes(Path.Combine(directory, "sentencepiece.bpe.model"), Convert.FromBase64String((string)Reference.Value["tiny_model_base64"]!));
        File.WriteAllText(Path.Combine(directory, "special_tokens_map.json"), "{\"additional_special_tokens\": [\"eng_Latn\", \"fra_Latn\"]}");
        return directory;
    });

    public static SpiritLMOptions Options(bool expressive = false)
    {
        var options = new SpiritLMOptions
        {
            TokenizerPath = TokenizerPath.Value,
            HiddenDim = 16, Layers = 2, NumHeads = 2, Intermediate = 32, VocabSize = VocabularySize, MaxTextLength = 4096,
            Units = Units, HubertWidth = 16, HubertLayer = 1, HubertHeads = 2, HubertIntermediate = 32, HubertConvChannels = 16,
            VocoderSpeakers = 4, Speaker = 1, VocoderStyles = 5, VocoderStyle = 1, VocoderEmbeddingDim = 8, VocoderChannels = 32,
            VocoderUpsampleRates = new[] { 5, 4, 4, 4, 2 }, VocoderUpsampleKernels = new[] { 11, 8, 8, 8, 4 }, VocoderSegmentSize = 2560,
            MaxNewTextTokens = 8, MaxNewSpeechTokensFromText = 12, MaxNewSpeechTokensFromSpeech = 12,
            MinTextSpanWords = 2, MaxTextSpanWords = 3, MinSpeechSpanWords = 1, MaxSpeechSpanWords = 2,
            LearningRate = 1e-2, WeightDecay = 0.0,
        };
        if (expressive)
        {
            options.Variant = SpiritLmVariant.Expressive;
            options.VocoderStyles = 0;
            options.PitchTokens = (int)Reference.Value["pitch"]!;
            options.StyleTokens = (int)Reference.Value["style"]!;
            options.FcpeMels = 16;
            options.FcpeWidth = 16;
            options.FcpeLayers = 1;
            options.FcpeBins = 30;
            options.PitchQuantizerWidth = 8;
            options.PitchQuantizerLatent = 8;
            options.StyleEncoderWidth = 16;
            options.StyleEncoderLayers = 1;
            options.StyleEncoderHeads = 2;
            options.StyleEncoderIntermediate = 32;
            options.StyleEncoderConvChannels = 16;
            options.StyleEncoderProjection = 8;
        }
        return options;
    }

    /// <summary>SpiritLM Expressive with the paper's SpeechProp style tokens: a tiny 25 Hz EnCodec (2 codebooks of 16), a
    /// one-layer 16-wide SpeechProp encoder and decoders, 6 Expresso styles, 4 style centroids and a tiny SONAR text
    /// encoder.</summary>
    public static SpiritLMOptions SpeechPropOptions()
    {
        var options = Options(expressive: true);
        options.StyleTokenizer = SpiritLmStyleTokenizer.SpeechProp;
        options.SpeechPropCodec = new AiDotNet.Audio.Generation.EnCodecOptions
        {
            SampleRate = 16000, Ratios = new[] { 8, 5, 4, 4 }, Filters = 4, Dimension = 8, LstmLayers = 1, NumQuantizers = 2, CodebookSize = 16,
            TargetBandwidths = new[] { 0.2 }, TargetBandwidthKbps = 0.2, SegmentSize = 16000, IncludeDiscriminators = false,
        };
        options.SpeechPropWidth = 16;
        options.SpeechPropHeads = 2;
        options.SpeechPropFeedForward = 32;
        options.SpeechPropEncoderLayers = 1;
        options.SpeechPropDecoderLayers = 1;
        options.SpeechPropDim = 8;
        options.ExpressoStyles = 6;
        options.StyleTokens = 4;
        options.SpeechPropLearningRate = 1e-2;
        options.IncludeSonarTextEncoder = true;
        options.SonarVocabSize = VocabularySize + 3;
        options.SonarTokenizerDirectory = SonarTokenizerDirectory.Value;
        options.SonarWidth = 16;
        options.SonarLayers = 1;
        options.SonarHeads = 2;
        options.SonarFeedForward = 32;
        return options;
    }

    /// <summary>A SpeechProp SpiritLM whose codec's codebooks are fitted on a second of tones (an unfitted codec encodes
    /// everything alike).</summary>
    public static SpiritLM<T> SpeechPropModel<T>()
    {
        var model = Tts<T>(SpeechPropOptions());
        model.SpeechPropCodec!.InitializeCodebooks(Tones<T>(1.0));
        return model;
    }

    private static NeuralNetworkArchitecture<T> Architecture<T>() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);

    public static SpiritLM<T> Tts<T>(SpiritLMOptions? options = null)
    {
        var model = new SpiritLM<T>(Architecture<T>(), options ?? Options());
        SpreadUnits(model);
        return model;
    }

    /// <summary>
    /// Makes the untrained CTC quantizer a nearest-centroid classifier, as a trained one is: centroids are the extractor's
    /// own features at evenly spaced frames of a second of noisy chirp, and the three layers compute
    /// <c>2c·x − ‖c‖²</c> on the first 15 feature dimensions (the leaky ReLUs kept in their linear range by an offset;
    /// the blank never wins). An untrained quantizer labels every frame alike, which leaves no speech to interleave.
    /// </summary>
    private static void SpreadUnits<T>(SpiritLM<T> model)
    {
        var extractor = model.UnitExtractor;
        var quantizer = extractor.Quantizer!;
        var ops = MathHelper.GetNumericOperations<T>();
        var random = new Random(7);
        var audio = new Tensor<T>(new[] { 16000 }, new Vector<T>(Enumerable.Range(0, 16000)
            .Select(i => ops.FromDouble(0.3 * (2 * random.NextDouble() - 1) + 0.3 * Math.Sin(2 * Math.PI * (150 + i / 40.0) * i / 16000))).ToArray()));
        Tensor<T> features;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>()) features = extractor.Network.Forward(audio, extractor.Layer)[extractor.Layer];
        int frames = features.Shape[0], width = quantizer.Dim, first = quantizer.First, second = quantizer.Second, units = quantizer.Units;
        const double offset = 100;
        var w1 = new double[first * width];
        var b1 = new double[first];
        for (int i = 0; i < first; i++) { if (i < width) w1[i * width + i] = 1; b1[i] = offset; }
        var w2 = new double[second * first];
        var b2 = new double[second];
        for (int i = 0; i < second; i++) { w2[i * first + i] = 1; b2[i] = 0; }
        // h2 = x[0..second) + offset; logits = 2c·(h2 − offset) − ‖c‖².
        var w3 = new double[(units + 1) * second];
        var b3 = new double[units + 1];
        for (int k = 0; k < units; k++)
        {
            int frame = k * frames / units;
            double norm = 0, shift = 0;
            for (int i = 0; i < second; i++)
            {
                double c = ops.ToDouble(features[frame, i]);
                w3[k * second + i] = 2 * c;
                norm += c * c;
                shift += 2 * c * offset;
            }
            b3[k] = -norm - shift;
        }
        b3[units] = -1e6;
        var weights = new Dictionary<string, double[]>
        {
            ["0.weight"] = w1, ["0.bias"] = b1, ["2.weight"] = w2, ["2.bias"] = b2, ["4.weight"] = w3, ["4.bias"] = b3,
        };
        quantizer.LoadTorchWeights(name => weights.TryGetValue(name, out var v) ? v : null);
    }

    /// <summary>Two tones at 16 kHz, <paramref name="seconds"/> long.</summary>
    public static Tensor<T> Tones<T>(double seconds, double low = 180.0, double high = 470.0)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int samples = (int)(16000 * seconds);
        var audio = new T[samples];
        for (int i = 0; i < samples; i++)
        {
            double t = (double)i / 16000;
            double f = low + 40 * Math.Sin(2 * Math.PI * 1.5 * t);
            audio[i] = ops.FromDouble(0.4 * Math.Sin(2 * Math.PI * f * t) + 0.2 * Math.Sin(2 * Math.PI * high * t));
        }
        return new Tensor<T>(new[] { samples }, new Vector<T>(audio));
    }

    /// <summary>A word list spread evenly over <paramref name="seconds"/>.</summary>
    public static SpiritLmWord[] Words(string text, double seconds)
    {
        var words = text.Split(' ');
        double each = seconds / words.Length;
        return words.Select((w, i) => new SpiritLmWord(w, i * each, (i + 1) * each)).ToArray();
    }
}
