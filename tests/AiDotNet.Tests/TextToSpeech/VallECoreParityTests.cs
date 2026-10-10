using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// VALL-E's codec language models reproduce the reference reproduction (lifeiteng/vall-e <c>VALLE</c>, the code the
/// paper's gaps follow): AR logits and loss, NAR logits and loss for the stage and prompt segment the reference drew,
/// and greedy prompted inference of every codebook.
/// </summary>
/// <remarks><c>ReferenceData/valle_reference.json</c> holds a tiny seeded model's float32 weights and its float64
/// results (<c>tools/reference-data/valle_reference.py</c>).</remarks>
public class VallECoreParityTests
{
    private static JObject Reference()
    {
        const string fileName = "valle_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    private static (VallECore<double> Core, JObject Fixture) Load()
    {
        var fixture = Reference();
        var config = fixture["config"]!;
        int d = (int)config["d_model"]!;
        var core = new VallECore<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), new List<LayerBase<double>>(),
            new VallEConfiguration(TextTokens: 512, AudioTokens: 1024, Codebooks: (int)config["num_quantizers"]!, ModelDim: d,
                Heads: (int)config["nhead"]!, Layers: (int)config["num_layers"]!, FeedForwardDim: 4 * d, Dropout: 0.1));
        string path = Path.Combine(Path.GetTempPath(), $"valle_reference_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)fixture["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            core.LoadTorchWeights((name, shape) =>
            {
                var tensor = weights[name];
                Assert.Equal(shape, tensor.Shape.ToArray());
                used.Add(name);
                return tensor.ToVector().Select(v => (double)v).ToArray();
            });
            Assert.Equal(weights.Keys.OrderBy(k => k, StringComparer.Ordinal), used.OrderBy(k => k, StringComparer.Ordinal));
        }
        finally
        {
            File.Delete(path);
        }
        return (core, fixture);
    }

    private static int[,] Matrix(JToken rows)
    {
        var list = rows.Select(r => r.Select(v => (int)v).ToArray()).ToArray();
        var m = new int[list.Length, list[0].Length];
        for (int i = 0; i < list.Length; i++)
            for (int j = 0; j < list[0].Length; j++) m[i, j] = list[i][j];
        return m;
    }

    private static void Close(double[] expected, Tensor<double> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-8 * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: reference {expected[i]:R}, port {actual[i]:R}.");
    }

    [Fact(Timeout = 120000)]
    public async Task TeacherForcedLogitsAndLosses_MatchTheReference()
    {
        await Task.Yield();
        var (core, fixture) = Load();
        var random = new Random(0);
        var text = fixture["text"]!.Select(v => (int)v).ToArray();
        var codes = Matrix(fixture["codes"]!);
        int frames = codes.GetLength(0);
        var first = Enumerable.Range(0, frames).Select(t => codes[t, 0]).ToArray();

        Close(fixture["ar_logits"]!.Select(v => (double)v).ToArray(), core.ArLogits(text, first, false, random), "AR logits");
        Close(new[] { (double)fixture["ar_loss"]! }, core.ArLoss(text, first, false, random), "AR loss");

        int stage = (int)fixture["nar_stage"]!, start = (int)fixture["nar_prompt_start"]!, length = Math.Min(225, frames / 4);
        var prompt = new int[length, codes.GetLength(1)];
        for (int t = 0; t < length; t++)
            for (int j = 0; j < codes.GetLength(1); j++) prompt[t, j] = codes[start + t, j];
        Close(fixture["nar_logits"]!.Select(v => (double)v).ToArray(), core.NarLogits(text, prompt, codes, stage, false, random), "NAR logits");
        // The reference computes its scale total / (total − prefix) from float32 lengths (12 / 9 → 1.3333334f); the port's
        // is exact, so the reference's loss is compared after replacing that one rounding.
        double exact = (double)frames / (frames - length), single = (float)frames / (float)(frames - length);
        Close(new[] { (double)fixture["nar_loss"]! / single * exact }, core.NarLossAt(text, codes, stage, start, length, false, random), "NAR loss");
    }

    [Fact(Timeout = 120000)]
    public async Task GreedyPromptedInference_MatchesTheReference()
    {
        await Task.Yield();
        var (core, fixture) = Load();
        var random = new Random(0);
        var text = fixture["text"]!.Select(v => (int)v).ToArray();
        var prompt = Matrix(fixture["prompt"]!);
        var expected = Matrix(fixture["generated"]!);
        int enrolled = (int)fixture["enrolled"]!;

        var promptFirst = Enumerable.Range(0, prompt.GetLength(0)).Select(t => prompt[t, 0]).ToArray();
        // The reference stops after 16 new codes per phoneme token (x_lens · 16, then one more).
        var generated = core.ArGenerate(text, promptFirst, temperature: 1.0, topK: 1, maxNewCodes: 16 * text.Length + 1, random);
        Assert.Equal(Enumerable.Range(0, expected.GetLength(0)).Select(t => expected[t, 0]), generated);

        // prefix_mode 2 drops the enrolled phonemes from the NAR's text: <bos> + text[enrolled − 1 :].
        var narText = new[] { text[0] }.Concat(text.Skip(enrolled - 1)).ToArray();
        var codes = core.NarGenerate(narText, prompt, generated.ToArray(), random);
        Assert.Equal(expected, codes);
    }
}
