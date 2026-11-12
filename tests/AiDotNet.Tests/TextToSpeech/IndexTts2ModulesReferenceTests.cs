using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// IndexTTS2's modules against the release's code on tiny sizes (tools/reference-data/indextts2_modules_reference.py):
/// the speaker and emotion conditioners, the semantic latents under the release's speed tokens and the paper's duration
/// token, beam-search generation, the length regulator, the GPT latent fusion, the DiT velocity, the guided Euler solution
/// and BigVGAN-v2.
/// </summary>
public class IndexTts2ModulesReferenceTests
{
    internal const int Features = 24, D = 32, Heads = 2, Layers = 2, MaxText = 20, MaxMel = 40, TextTokens = 20, MelCodes = 24, StartMel = 22, StopMel = 23;

    internal static IndexTtsUnifiedVoiceConfiguration VoiceConfiguration(IndexTts2DurationConditioning duration) => new(
        D, Heads, Layers, MaxText, MaxMel, TextTokens, MelCodes, 0, 1, StartMel, StopMel, 32,
        new IndexTtsConformerConfiguration(Features, 16, 32, 2, 2), 2, new IndexTtsConformerConfiguration(Features, 16, 24, 2, 1), 2, Features,
        duration, 0.1);

    internal static IndexTtsSemanticToMelConfiguration MelConfiguration() =>
        new(5, 16, 2, 5, 6, 12, 8, 2, 16, 3, 5, 0.2, 0.1, D, 10, 7);

    private static JObject Fixture()
    {
        const string fileName = "indextts2_modules_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();

    private static Tensor<double> Tensor(JToken t, params int[] shape) => new(shape, new Vector<double>(Doubles(t)));

    private static void Close(JToken expected, Tensor<double> actual, string what, double tolerance = 1e-9)
    {
        var e = Doubles(expected);
        Assert.Equal(e.Length, actual.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {actual[i]}");
    }

    [Fact(Timeout = 300000)]
    public async Task Modules_MatchRelease()
    {
        await Task.Yield();
        var f = Fixture();
        var state = (JObject)f["state"]!;
        var used = new HashSet<string>();
        Func<string, int[], double[]> Reader(string prefix) => (name, _) =>
        {
            used.Add(prefix + name);
            return Doubles(state[prefix + name] ?? throw new InvalidDataException(prefix + name));
        };
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var random = new Random(5);
        var layers = new List<LayerBase<double>>();
        var release = new IndexTtsUnifiedVoice<double>(engine, layers, VoiceConfiguration(IndexTts2DurationConditioning.ReleaseSpeedTokens), random);
        var paper = new IndexTtsUnifiedVoice<double>(engine, layers, VoiceConfiguration(IndexTts2DurationConditioning.Paper), random);
        release.Load(Reader("gpt."));
        paper.Load((name, shape) => name == "speed_emb.weight" ? throw new InvalidOperationException() : Doubles(state["gpt." + name]!));
        var s2mel = new IndexTtsSemanticToMel<double>(engine, layers, MelConfiguration(), random);
        s2mel.Regulator.Load(engine, Reader("regulator."), "");
        s2mel.Estimator.Load(Reader("cfm.estimator."), name => state["cfm.estimator." + name] is not null, "");
        var gptLayer = Reader("gpt_layer.");
        IndexTtsOps.LoadLinear(engine, gptLayer, s2mel.Gpt1, "0", D, 10);
        IndexTtsOps.LoadLinear(engine, gptLayer, s2mel.Gpt2, "1", 10, 7);
        IndexTtsOps.LoadLinear(engine, gptLayer, s2mel.Gpt3, "2", 7, 12);
        var vocoder = new BigVganGenerator<double>(engine, random, 5, 8, new[] { 2, 2 }, new[] { 4, 4 }, new[] { 3 }, new[] { new[] { 1, 3 } },
            snakeBeta: true, tanhAtFinal: false, biasAtFinal: false);
        var readVocoder = Reader("vocoder.");
        vocoder.LoadTorch(name => state["vocoder." + name] is null ? null : readVocoder(name, Array.Empty<int>()));
        // Weights the release builds but never reads in inference: the text head, the DiT's unused embedders, the
        // regulator's discrete-content embedding, and the skip projections of blocks that emit rather than receive.
        var unused = state.Properties().Select(p => p.Name).Where(n => !used.Contains(n)).ToList();
        Assert.All(unused, n => Assert.True(n.StartsWith("gpt.text_head", StringComparison.Ordinal) || n.Contains("x_embedder") ||
            n.Contains("cond_embedder") || n.Contains("content_mask_embedder") || n == "regulator.embedding.weight" || n == "regulator.mask_token" ||
            (n.Contains("skip_in_linear") && (n.Contains("layers.0.") || n.Contains("layers.1.") || n.Contains("layers.2."))), n));

        using var _ = new NoGradScope<double>();
        var features = Tensor(f["features"]!, 13, Features);
        var speaker = release.SpeakerConditioning(features);
        Close(f["speaker"]!, speaker, "speaker");
        var emotion = release.EmotionVector(Tensor(f["emo_features"]!, 9, Features));
        Close(f["emotion"]!, emotion, "emotion");

        int[] text = { 3, 7, 11, 4, 9 };
        int[] codes = { 5, 17, 2, 2, 9, 13 };
        var releaseLatents = release.Latents(release.Conditions(speaker, emotion, null), text, codes);
        Close(f["release_latents"]!, engine.TensorNarrow(releaseLatents, 0, 0, codes.Length), "release latents");
        var paperLatents = paper.Latents(paper.Conditions(speaker, emotion, codes.Length + 1), text, codes);
        Close(f["paper_latents"]!, paperLatents, "paper latents");
        Close(f["paper_logits"]!, paper.Logits(paperLatents), "paper logits");
        Close(f["paper_free_latents"]!, paper.Latents(paper.Conditions(speaker, emotion, null), text, codes), "paper free latents");

        var generated = release.Generate(release.Conditions(speaker, emotion, null), text,
            new IndexTtsDecoding(false, 3, 30, 0.8, 0.8, 10.0, 10), new Random(1));
        Assert.Equal(f["codes"]!.Select(v => (int)v).ToArray(), generated);
        // Generation drops the start and stop text ids (prepare_gpt_inputs); this text's codes differ without the strip.
        var stripped = release.Generate(release.Conditions(speaker, emotion, null), f["strip_text"]!.Select(v => (int)v).ToArray(),
            new IndexTtsDecoding(false, 3, 30, 0.8, 0.8, 10.0, 10), new Random(1));
        Assert.Equal(f["stripped_codes"]!.Select(v => (int)v).ToArray(), stripped);

        var content = Tensor(f["content"]!, 6, 12);
        var regulated = s2mel.Regulator.Forward(content, 11);
        Close(f["regulated"]!, regulated, "regulated");
        Close(f["fused"]!, s2mel.Fuse(content, Tensor(f["gpt_latents"]!, 6, D)), "fused");
        var promptMel = Tensor(f["prompt_mel"]!, 4, 5);
        var promptX = engine.TensorConcatenate(new[] { promptMel, new Tensor<double>(new[] { 7, 5 }) }, 0);
        var style = Tensor(f["style"]!, 6);
        Close(f["velocity"]!, s2mel.Estimator.Forward(Tensor(f["x"]!, 11, 5), promptX, 0.37, style, regulated, false), "velocity");
        Close(f["solved"]!, s2mel.Solve(regulated, promptMel, style, Tensor(f["noise"]!, 11, 5), 4, 0.7), "solved");

        Close(f["waveform"]!, vocoder.Forward(Tensor(f["vocoder_mel"]!, 1, 5, 6)), "waveform");
    }
}
