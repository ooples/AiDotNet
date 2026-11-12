using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// SpeechT5 against Microsoft's own code on tiny sizes (tools/reference-data/speecht5_reference.py, the official fairseq
/// modules): text to speech (teacher-forced outputs, cross-attentions, the fine-tuning loss with guided attention, and
/// generation), and both pre-training objectives (HuBERT's masked prediction with the feature penalty, the codebook and the
/// reconstruction; text infilling with the codebook).
/// </summary>
public class SpeechT5ReferenceTests
{
    internal static SpeechT5Configuration TinyConfiguration() => new()
    {
        Hidden = 16, FeedForward = 32, Heads = 2, EncoderLayers = 2, DecoderLayers = 2, MaxRelativePosition = 4, TextVocabulary = 15,
        PadId = 1, EosId = 2, MaxTextPositions = 40, MaxSpeechPositions = 200, MelChannels = 10, ReductionFactor = 2, SpeakerDim = 6,
        PrenetUnits = 8, PrenetLayers = 2, PrenetDropout = 0, PostnetChannels = 8, PostnetLayers = 5, PostnetKernel = 5, PostnetDropout = 0,
        FeatureLayers = new[] { (8, 10, 5), (8, 3, 2), (8, 3, 2) }, ConvPositionKernel = 8, ConvPositionGroups = 2, FinalDim = 6,
        HubertClasses = new[] { 16 }, LatentVars = 5, LatentGroups = 2, Dropout = 0, AttentionDropout = 0, ActivationDropout = 0,
        PositionalDropout = 0,
    };

    private static JObject Fixture()
    {
        const string fileName = "speecht5_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
    private static int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();
    private static Tensor<double> Tensor(JToken t, params int[] shape) => new(shape, new Vector<double>(Doubles(t)));

    private static void Close(JToken expected, Tensor<double> actual, string what, double tolerance = 1e-9)
    {
        var e = Doubles(expected);
        Assert.Equal(e.Length, actual.Length);
        for (int i = 0; i < e.Length; i++)
        {
            if (double.IsNegativeInfinity(e[i])) { Assert.True(double.IsNegativeInfinity(actual.GetFlat(i)), $"{what}[{i}]"); continue; }
            Assert.True(Math.Abs(e[i] - actual.GetFlat(i)) <= tolerance * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {actual.GetFlat(i)}");
        }
    }

    private static int[] Permutation(int[] given, int n) =>
        given.Where(p => p < n).Concat(Enumerable.Range(0, n).Where(i => !given.Contains(i))).ToArray();

    [Fact(Timeout = 300000)]
    public async Task Model_MatchesOfficialCode()
    {
        await Task.Yield();
        var f = Fixture();
        var state = (JObject)f["state"]!;
        var buffers = (JObject)f["buffers"]!;
        var used = new HashSet<string>();
        double[] Read(string name, int[] _)
        {
            used.Add(name);
            return Doubles((state[name] ?? buffers[name]) ?? throw new InvalidDataException(name));
        }
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var c = TinyConfiguration();
        var core = new SpeechT5Core<double>(engine, new List<LayerBase<double>>(), new List<LayerBase<double>>(), c, new Random(3));
        core.Load(Read, pretraining: true);
        foreach (var bn in core.SpeechPostnet.Postnet) bn.Norm.SetTrainingMode(false);
        // Built but unused by these tasks: the CTC projection, the decoder's relative-position table and the layers'
        // key norms (applied only with pre-norm layers), and the text decoder pre-net's shared table under its own name.
        var unused = state.Properties().Select(p => p.Name).Where(n => !used.Contains(n)).ToList();
        Assert.All(unused, n => Assert.True(n.StartsWith("encoder.proj", StringComparison.Ordinal) || n.StartsWith("decoder.pos_emb", StringComparison.Ordinal)
            || n.Contains(".norm_k.") || n.StartsWith("text_decoder_prenet.", StringComparison.Ordinal) || n.StartsWith("text_decoder_postnet.", StringComparison.Ordinal), n));

        using var _ = new NoGradScope<double>();
        var tokens = Ints(f["tokens"]!);
        var speaker = Tensor(f["spk"]!, 6);
        var mel = Tensor(f["mel"]!, 13, 10);
        var memory = core.EncodeText(tokens);
        var outputs = core.DecodeSpeech(memory, core.DecoderInputs(mel), speaker);
        Close(f["before"]!, outputs.Before, "before");
        Close(f["after"]!, outputs.After, "after");
        Close(f["logits"]!, outputs.Logits, "logits");
        Close(f["attn"]!, engine.TensorConcatenate(outputs.CrossWeights.Select(w => engine.Reshape(w, new[] { 1, w.Shape[0], w.Shape[1], w.Shape[2] })).ToArray(), 0), "attn");
        Close(f["tts_loss"]!, core.SpeechLoss(outputs, mel, tokens.Length, 2), "tts loss");

        var (generated, probabilities) = core.Generate(tokens, speaker, new Random(1));
        Close(f["generated"]!, generated, "generated");
        Close(f["generated_probs"]!, new Tensor<double>(new[] { probabilities.Length }, new Vector<double>(probabilities)), "generated probabilities");

        var wave = Tensor(f["wave"]!, 1600);
        var mask = f["mask"]!.Select(v => (bool)v).ToArray();
        int frames = mask.Length;
        var terms = core.SpeechPretrainLoss(wave, new[] { Ints(f["hubert"]!) }, Tensor(f["pt_mel"]!, 10, 10), speaker, mask,
            Permutation(Ints(f["permutation"]!), frames), 0.5, 50, 16000, false, null);
        Close(f["features_pen"]!, terms.FeaturePenalty, "feature penalty");
        Close(f["prob_perplexity"]!, terms.Perplexity, "perplexity");
        Close(f["speech_loss"]!, terms.Total, "speech pre-training loss");
        var (inputs, _) = core.SpeechEncoderPrenet.Forward(wave, mask, false, null);
        var hubert = core.Hubert.MaskedLogits(core.Encoder.Forward(inputs), mask, new[] { Ints(f["hubert"]!).Select((_, t) => Ints(f["hubert"]!)[(int)(t * 0.0625)]).ToArray() });
        Close(f["hubert_logits"]!, hubert[0], "hubert logits");

        var noised = Ints(f["noised"]!);
        var target = Ints(f["target"]!);
        Close(f["text_loss"]!, core.TextPretrainLoss(noised, target, Permutation(Ints(f["permutation"]!), noised.Length), 0.5, false, null), "text pre-training loss");
        var (mixed, _) = core.MixCodebook(core.EncodeText(noised), Permutation(Ints(f["permutation"]!), noised.Length), 0.5, false, null);
        var previous = new[] { 2 }.Concat(target.Take(target.Length - 1)).ToArray();
        Close(f["text_logits"]!, core.TextLogits(mixed, previous), "text logits");
    }
}
