using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// IndexTTS (v1) against the release's code on tiny sizes (tools/reference-data/indextts_reference.py): the log mel, the
/// DVAE's encoding and codes, the speaker latents, UnifiedVoice's text and mel cross-entropies and logits on a padded
/// item, its latents, beam-search generation (also of a text holding the start and stop ids), the ECAPA-TDNN speaker
/// embedding, BigVGAN's waveform, remove_long_silence and fade_out_tail.
/// </summary>
public class IndexTtsReferenceTests
{
    private static readonly int[] Text = { 3, 7, 11, 4, 9 };
    private static readonly int[] TextWithSpecials = { 12, 0, 2, 1, 19, 6 };
    private static readonly int[] Codes = { 5, 17, 2, 2, 9, 13, 8, 6 };
    private static readonly int[] SilenceCodes = new[] { 4 }.Concat(Enumerable.Repeat(52, 33)).Concat(new[] { 7, 52, 52, 9 }).ToArray();

    private static JObject Fixture()
    {
        const string fileName = "indextts_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();

    private static Tensor<double> Tensor(JToken t, params int[] shape) => new(shape, new Vector<double>(Doubles(t)));

    private static void Close(JToken expected, Tensor<double> actual, string what, double tolerance = 1e-9) =>
        Close(Doubles(expected), actual.ToArray(), what, tolerance);

    private static void Close(double[] e, double[] actual, string what, double tolerance = 1e-9)
    {
        Assert.Equal(e.Length, actual.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {actual[i]}");
    }

    /// <summary>The fixture's sizes: the release's 100-band mel and ×1024 BigVGAN, everything else narrow.</summary>
    internal static IndexTTSOptions Options() => new()
    {
        HiddenDim = 32, NumHeads = 2, Layers = 2, MaxTextLength = 20, MaxMelTokens = 40, VocabSize = 20, MelCodes = 24,
        ConditionLatents = 32, ConditionDim = 16, ConditionLinearUnits = 32, ConditionHeads = 2, ConditionBlocks = 2, PerceiverMultiple = 2,
        DvaeHidden = 8, DvaeLayers = 2, DvaeResBlocks = 1, DvaeCodebookDim = 6, DvaeKernel = 3,
        VocoderInitialChannels = 64, VocoderResBlockKernels = new[] { 3 }, VocoderResBlockDilations = new[] { new[] { 1, 3, 5 } },
        SpeakerEmbeddingDim = 6, SpeakerEncoderChannels = new[] { 16, 16, 16, 16, 48 }, SpeakerEncoderAttentionChannels = 8,
        SpeakerEncoderSeChannels = 8, SpeakerEncoderScale = 8,
    };

    /// <summary>
    /// The released IndexTTS 1.0 against the release's code in float64 (tools/reference-data/indextts_realweights_reference.py)
    /// where the checkpoint is present (<c>INDEXTTS_DIR</c>: IndexTeam/Index-TTS with gpt.pth, bigvgan_generator.pth,
    /// dvae.pth and bpe.model): the tokens and ids, the DVAE's codes of the prompt, the speaker latents, generation without
    /// sampling, the latents of the codes, the ECAPA-TDNN embedding and BigVGAN's waveform.
    /// </summary>
    [Fact(Timeout = 3600000)]
    public async Task ReleasedCheckpoint_MatchesTheRelease_WhenPresent()
    {
        await Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("INDEXTTS_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "gpt.pth"))) return;
        string fixturePath = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", "indextts_realweights_reference.json");
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(fixturePath) && d is not null; d = d.Parent)
            fixturePath = Path.Combine(d.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "indextts_realweights_reference.json");
        var f = JObject.Parse(File.ReadAllText(fixturePath));
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new IndexTTS<double>(architecture, new IndexTTSOptions());
        model.DisableAutoStreaming();
        model.LoadReferenceWeights(dir);
        GC.Collect();

        var pieces = model.Tokenizer.Tokenize("Hello world, this is IndexTTS speaking.");
        Assert.Equal(f["tokens"]!.Select(v => (string)v!).ToArray(), pieces.ToArray());
        var ids = model.TokenizeText("Hello world, this is IndexTTS speaking.");
        Assert.Equal(f["ids"]!.Select(v => (int)v).ToArray(), ids);

        using var _ = new NoGradScope<double>();
        var prompt = Doubles(f["prompt"]!);
        var mel = model.MelSpectrogram(new Tensor<double>(new[] { prompt.Length }, new Vector<double>(prompt)));
        Assert.Equal(f["dvae_codes"]!.Select(v => (int)v).ToArray(), model.Dvae.Codes(mel));
        var gpt = model.TextToMel;
        var conditions = gpt.SpeakerConditioning(mel);
        Close(Doubles(f["speaker"]!), conditions.ToArray().Take(256).ToArray(), "speaker latents", 1e-7);
        var codes = gpt.Generate(conditions, ids, new IndexTtsDecoding(false, 3, 30, 0.8, 1.0, 10.0, 40), new Random(1));
        Assert.Equal(f["codes"]!.Select(v => (int)v).ToArray(), codes);
        Close(Doubles(f["latents"]!), gpt.Latents(conditions, ids, codes).ToArray().Take(256).ToArray(), "latents", 1e-7);
        var speaker = model.SpeakerEmbedding(mel);
        Close(f["speaker_embedding"]!, speaker, "speaker embedding", 1e-7);
        var wave = model.Decode(conditions, ids, codes, speaker);
        Assert.Equal((int)f["samples"]!, wave.Length);
        Close(Doubles(f["waveform"]!), wave.Take(4096).ToArray(), "waveform", 1e-6);
    }

    [Fact(Timeout = 600000)]
    public async Task Model_MatchesTheRelease()
    {
        await Task.Yield();
        var f = Fixture();
        var state = (JObject)f["state"]!;
        var used = new HashSet<string>();
        Func<string, double[]?> Reader(string prefix) => name =>
        {
            var token = state[prefix + name];
            if (token is null) return null;
            used.Add(prefix + name);
            return Doubles(token);
        };
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new IndexTTS<double>(architecture, Options());
        model.LoadState(Reader("gpt."), Reader("bigvgan."), Reader("dvae."));
        // The DVAE's decoder and its codebook's running averages only train the DVAE.
        var unused = state.Properties().Select(p => p.Name).Where(n => !used.Contains(n)).ToList();
        Assert.All(unused, n => Assert.True(n.StartsWith("dvae.decoder.", StringComparison.Ordinal)
            || n is "dvae.codebook.cluster_size" or "dvae.codebook.embed_avg", n));

        using var _ = new NoGradScope<double>();
        int frames = (int)f["frames"]!;
        var mel = model.MelSpectrogram(Tensor(f["wave"]!, 4000));
        Close(f["mel"]!, mel, "mel");
        mel = Tensor(f["mel"]!, frames, 100);

        Close(f["dvae_latents"]!, model.Dvae.Encode(mel), "DVAE encoding");
        Assert.Equal(f["dvae_codes"]!.Select(v => (int)v).ToArray(), model.Dvae.Codes(mel));

        var gpt = model.TextToMel;
        var conditions = gpt.SpeakerConditioning(mel);
        Close(f["speaker"]!, conditions, "speaker latents");

        var (textLoss, melLoss) = model.Losses(mel, Text, Codes, 5 * 1024);
        Close(f["loss_text"]!, textLoss, "text loss");
        Close(f["loss_mel"]!, melLoss, "mel loss");
        Close(f["mel_logits"]!, model.Logits(mel, Text, Codes, 5 * 1024).Mel, "mel logits");

        var latents = gpt.Latents(conditions, Text, Codes);
        Close(f["latents"]!, latents, "latents");

        var decoding = new IndexTtsDecoding(false, 3, 30, 0.8, 1.0, 10.0, 12);
        Assert.Equal(f["codes"]!.Select(v => (int)v).ToArray(), gpt.Generate(conditions, Text, decoding, new Random(1)));
        Assert.Equal(f["codes_specials"]!.Select(v => (int)v).ToArray(), gpt.Generate(conditions, TextWithSpecials, decoding, new Random(1)));

        var speaker = model.SpeakerEmbedding(mel);
        Close(f["speaker_embedding"]!, speaker, "speaker embedding");
        Close(Doubles(f["waveform"]!), model.Decode(conditions, Text, Codes, speaker), "waveform");

        Assert.Equal(f["silence_shrunk"]!.Select(v => (int)v).ToArray(), model.RemoveLongSilence(SilenceCodes));
        Close(Doubles(f["faded"]!), IndexTTS<double>.FadeOutTail(Doubles(f["fade_input"]!), 1000), "faded");
    }
}
