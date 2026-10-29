using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Vevo against the release's code on tiny sizes (tools/reference-data/vevo_reference.py): the mel, the 16 kHz resampling,
/// HuBERT's layer output, both tokenizers, the Vevo-TTS transformer's loss, logits and greedy decoding, the Vevo-Voice
/// transformer's decoding of its content-token input, the flow's guided Euler solution and training loss, and Vocos.
/// </summary>
public class VevoReferenceTests
{
    private static readonly int[] Phones = { 3, 7, 11, 4, 9, 2 };
    private static readonly int[] StylePhones = { 5, 1, 8 };

    private static JObject Fixture()
    {
        const string fileName = "vevo_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();

    private static int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();

    private static Tensor<double> Tensor(double[] values, params int[] shape) => new(shape, new Vector<double>(values));

    private static void Close(JToken expected, Tensor<double> actual, string what, double tolerance = 1e-9)
    {
        var e = Doubles(expected);
        var a = actual.ToArray();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= tolerance * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {a[i]}");
    }

    /// <summary>The fixture's sizes.</summary>
    internal static VevoOptions Options() => new()
    {
        MelChannels = 8, FftSize = 64, HopSize = 30, HiddenDim = 16, PhoneVocabulary = 20, MelMaxFrequency = 12000,
        HubertConvDims = new[] { 8, 8, 8 }, HubertConvKernels = new[] { 10, 3, 2 }, HubertConvStrides = new[] { 5, 2, 2 },
        HubertHidden = 16, HubertHeads = 2, HubertIntermediate = 32, HubertPositionKernel = 8, HubertPositionGroups = 4, HubertLayer = 2,
        ContentStyleCodes = 16, ContentStyleCodeDim = 4, ContentStyleVocosDim = 8, ContentStyleVocosIntermediate = 16, ContentStyleVocosLayers = 1,
        ContentCodes = 4, ContentCodecChannels = 16, TextArHidden = 16, TextArIntermediate = 32, TextArLayers = 2, TextArHeads = 2,
        VoiceArHidden = 16, VoiceArIntermediate = 24, VoiceArLayers = 2, VoiceArHeads = 2, FlowLayers = 2, FlowHeads = 2,
        VocoderDim = 16, VocoderIntermediate = 32, VocoderLayers = 2,
    };

    // torchaudio's names for the transformers names HubertNetwork reads.
    private static string TorchaudioName(string name)
    {
        if (name.StartsWith("feature_projection.", StringComparison.Ordinal)) return "encoder." + name;
        if (name.StartsWith("encoder.", StringComparison.Ordinal)) return "encoder.transformer." + name.Substring("encoder.".Length);
        return name;
    }

    [Fact(Timeout = 600000)]
    public async Task Model_MatchesTheRelease()
    {
        await Task.Yield();
        var f = Fixture();
        var shapes = (JObject)f["shapes"]!;
        var used = new HashSet<string>();
        double[]? Take(string key)
        {
            if (shapes[key] is not JArray shape) return null;
            used.Add(key);
            return MaskGctReferenceTests.ClosedForm(key.Substring(key.IndexOf('.') + 1), shape.Select(v => (int)v).ToArray());
        }
        Func<string, int[], double[]> Reader(string prefix) => (name, _) => Take(prefix + name) ?? throw new InvalidDataException(prefix + name);

        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new Vevo<double>(architecture, Options());
        model.LoadHubert(name => Take("hubert." + TorchaudioName(name)));
        model.ContentStyleTokenizer.LoadStatistics(Doubles(f["hubert_mean"]!), Doubles(f["hubert_std"]!).Select(s => s * s).ToArray());
        model.ContentStyleTokenizer.Load((name, _) => Take("content_style." + name));
        model.ContentTokenizer.Load(Reader("content."));
        model.TextToContentStyle.Load(Reader("ar_tts."));
        model.ContentToContentStyle.Load(Reader("ar_vc."));
        model.Flow.Load(Reader("fmt."));
        var o = Options();
        model.Vocoder.LoadTorchWeights(Reader("vocos."), o.MelChannels, o.VocoderIntermediate);
        // HuBERT's third layer is never read; the tokenizers' decoders and the RepCodec's codebook usage counters only trained them.
        var unused = shapes.Properties().Select(p => p.Name).Where(n => !used.Contains(n)).ToList();
        Assert.All(unused, n => Assert.True(n.StartsWith("hubert.encoder.transformer.layers.2.", StringComparison.Ordinal)
            || n.StartsWith("content.decoder.", StringComparison.Ordinal) || n.StartsWith("content_style.decoder.", StringComparison.Ordinal), n));

        using var _ = new NoGradScope<double>();
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var wave = Doubles(f["wave24k"]!);
        var timbre = Doubles(f["timbre24k"]!);
        var wave24 = Tensor(wave, wave.Length);
        var timbre24 = Tensor(timbre, timbre.Length);
        Close(f["mel"]!, model.Mel(wave24), "mel");
        var wave16 = model.AtContentRate(wave24);
        Close(f["wave16k"]!, wave16, "16 kHz");
        wave16 = Tensor(Doubles(f["wave16k"]!), Doubles(f["wave16k"]!).Length);
        var features = model.HubertFeatures(wave16);
        Close(f["hubert"]!, features, "HuBERT");
        var contentStyle = Ints(f["content_style_codes"]!);
        Assert.Equal(contentStyle, model.ContentStyleCodes(wave16));
        Assert.Equal(Ints(f["content_codes"]!), model.ContentTokenizer.Codes(features));
        Assert.Equal(Ints(f["content_codes_reduced"]!), model.ContentCodes(wave16));

        var tts = model.TextToContentStyle;
        var targets = contentStyle.Take(10).ToArray();
        // transformers computes the causal-LM loss in float32 (ForCausalLMLoss upcasts the logits with .float()), so the
        // recorded loss carries float32 rounding; the logits are float64.
        Close(f["ar_loss"]!, tts.Loss(Phones, targets), "AR loss", 1e-7);
        Close(f["ar_logits"]!, tts.Logits(Phones, targets), "AR logits");
        var input = StylePhones.Concat(Phones).ToArray();
        var greedy = new VevoDecoding(false, 0.8, 50, 0.9, 1.0, 5, input.Length + 2 + 1 + 6 + 14);
        Assert.Equal(Ints(f["ar_tts_codes"]!), tts.Generate(input, contentStyle.Take(6).ToArray(), greedy, new Random(1)));

        var timbre16 = model.AtContentRate(timbre24);
        var joined = model.ContentCodes(engine.TensorConcatenate(new[] { timbre16, wave16 }, 0));
        Assert.Equal(Ints(f["voice_input"]!), joined);
        var timbreCodes = model.ContentStyleCodes(timbre16);
        var voiceDecoding = new VevoDecoding(false, 0.8, 50, 0.9, 1.0, 3, joined.Length + 2 + 1 + 39 + 10);
        Assert.Equal(Ints(f["ar_vc_codes"]!), model.ContentToContentStyle.Generate(joined.Select(c => c + 20).ToArray(), timbreCodes, voiceDecoding, new Random(1)));

        var fmCodes = Ints(f["fm_codes"]!);
        Assert.Equal(fmCodes, timbreCodes.Concat(contentStyle).ToArray());
        var prompt = model.Mel(timbre24);
        int target = fmCodes.Length - prompt.Shape[0];
        var noise = Tensor(Doubles(f["fm_noise"]!), target, 8);
        var mel = model.Flow.Solve(model.Flow.Embed(fmCodes), prompt, noise, 4, 1.0, 0.75);
        Close(f["fm_mel"]!, mel, "flow mel");
        Close(f["vocoder_wave"]!, model.Vocoder.Forward(engine.Reshape(engine.TensorTranspose(mel), new[] { 1, 8, target })), "Vocos");

        int frames = contentStyle.Length;
        var x = engine.TensorNarrow(model.Mel(wave24), 0, 0, frames);
        var lossNoise = Tensor(Doubles(f["loss_noise"]!), frames, 8);
        Close(f["fm_loss"]!, model.Flow.Loss(x, contentStyle, 0.3, dropCondition: 0.9 <= 0.2, 7, lossNoise), "flow loss");
    }
}
