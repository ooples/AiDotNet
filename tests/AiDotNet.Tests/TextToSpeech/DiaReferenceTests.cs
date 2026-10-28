using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Dia against nari-labs/dia's DiaModel and Dia.generate at a tiny size and transformers' DiaForConditionalGeneration on
/// the same weights (tools/reference-data/dia_reference.py): every release tensor has a place, the script tokenizes to the
/// release's bytes, the encoder's output matches, greedy generation with classifier-free guidance and an audio prompt
/// decodes the same logits and returns the same codes, and the training loss matches transformers'.
/// </summary>
public class DiaReferenceTests
{
    private static JsonElement Load(string name)
    {
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static double[] ClosedForm(string name, int[] shape)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        int fanIn = name.EndsWith("o_proj.weight", StringComparison.Ordinal) ? shape[0] * shape[1] : shape[0];
        var values = new double[count];
        for (int i = 0; i < count; i++)
        {
            long k = i;
            double s = Math.Sin(0.37 * i + 0.11 * name.Length + 0.0013 * (k * k % 9973));
            values[i] = name.EndsWith("norm.weight", StringComparison.Ordinal) ? 1 + 0.2 * s
                : name.Contains("embedding") ? 0.5 * s
                : 0.5 * s / Math.Sqrt(fanIn);
        }
        return values;
    }

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();
    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static void Close(double[] expected, double[] actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = Math.Max(1e-300, expected.Max(Math.Abs)), worst = 0;
        for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
        Assert.True(worst <= tolerance * scale, $"{what} differs by {worst} (scale {scale})");
    }

    internal static DiaOptions TinyOptions() => new()
    {
        VocabSize = 256, HiddenDim = 16, EncoderLayers = 2, EncoderHeads = 2, EncoderKeyValueHeads = 2, EncoderHeadDim = 8, EncoderIntermediate = 24,
        DecoderHidden = 16, DecoderLayers = 2, DecoderHeads = 4, DecoderKeyValueHeads = 2, DecoderHeadDim = 8, CrossHeads = 2, CrossKeyValueHeads = 2,
        CrossHeadDim = 8, DecoderIntermediate = 24, AudioVocabSize = 20, Channels = 3, DelayPattern = new[] { 0, 2, 3 }, EndToken = 16, PadToken = 17,
        BeginToken = 18, MaxAudioTokens = 16, Temperature = 0, CfgScale = 3, CfgFilterTopK = 5, TopP = 0.95, SampleRate = 16000, HopSize = 8,
        Codec = new AiDotNet.Audio.Effects.DACOptions
        {
            SampleRate = 16000, EncoderDim = 4, EncoderRates = [2, 4], DecoderDim = 16, DecoderRates = [4, 2], NumQuantizers = 3, CodebookSize = 16,
            CodebookDim = 2, TargetBandwidthKbps = 24.0,
        },
    };

    private static Dia<double> Tiny(JsonElement reference)
    {
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new Dia<double>(architecture, TinyOptions());
        var shapes = reference.GetProperty("shapes").EnumerateObject().ToDictionary(p => p.Name, p => Ints(p.Value), StringComparer.Ordinal);
        var used = model.LoadReleaseWeights(name => shapes.TryGetValue(name, out var shape) ? ClosedForm(name, shape) : null);
        Assert.Empty(shapes.Keys.Where(n => !used.Contains(n)));
        return model;
    }

    private static int[,] Matrix(JsonElement rows)
    {
        var values = rows.EnumerateArray().Select(Ints).ToArray();
        var matrix = new int[values.Length, values.Length == 0 ? 0 : values[0].Length];
        for (int i = 0; i < values.Length; i++)
            for (int j = 0; j < values[i].Length; j++) matrix[i, j] = values[i][j];
        return matrix;
    }

    [Fact(Timeout = 600000)]
    public async Task EncoderGenerationAndLoss_MatchTheReleaseAndTransformers()
    {
        await Task.Yield();
        var reference = Load("dia_reference.json");
        var model = Tiny(reference);
        var text = Dia<double>.Tokenize(reference.GetProperty("text").GetString()!);
        Assert.Equal(Ints(reference.GetProperty("text_ids")), text);

        var ids = new int[1, text.Length];
        for (int j = 0; j < text.Length; j++) ids[0, j] = text[j];
        Close(Doubles(reference.GetProperty("encoded")), model.Network.Encode(ids).ToArray(), 1e-12, "encoder output");

        var steps = reference.GetProperty("steps").EnumerateArray().ToArray();
        var observed = new List<(double[] Uncond, double[] Cond)>();
        model.StepLogits = (u, c) => observed.Add((u, c));
        var codes = model.GenerateCodes(text, Matrix(reference.GetProperty("prompt")), TinyOptions().MaxAudioTokens);
        model.StepLogits = null;
        Assert.Equal(steps.Length, observed.Count);
        for (int i = 0; i < steps.Length; i++)
        {
            Close(Doubles(steps[i].GetProperty("uncond")), observed[i].Uncond, 1e-12, $"step {i}'s unconditional logits");
            Close(Doubles(steps[i].GetProperty("cond")), observed[i].Cond, 1e-12, $"step {i}'s conditional logits");
        }
        var expected = Matrix(reference.GetProperty("generated"));
        Assert.Equal(expected.GetLength(0), codes.GetLength(0));
        Assert.Equal(expected, codes);

        // The processor's layout and labels, then transformers' loss (computed from float32 logits).
        var frames = Matrix(reference.GetProperty("frames"));
        Assert.Equal(Matrix(reference.GetProperty("decoder_input_ids")), model.DelayedSequence(frames));
        double loss = model.Loss(text, frames).ToArray()[0];
        double target = reference.GetProperty("loss").GetDouble();
        Assert.True(Math.Abs(loss - target) <= 1e-6 * target, $"loss {loss} vs {target}");
    }

    // The fixture's conversion to transformers' names (its to_hf).
    private static Dictionary<string, double[]> TransformersState(Dictionary<string, int[]> shapes, int channels)
    {
        static double[] Transposed(double[] kernel, int rows, int columns)
        {
            var result = new double[kernel.Length];
            for (int r = 0; r < rows; r++)
                for (int k = 0; k < columns; k++) result[k * rows + r] = kernel[r * columns + k];
            return result;
        }
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var (name, shape) in shapes)
        {
            if (name.StartsWith("decoder.embeddings.", StringComparison.Ordinal)) continue;
            var w = ClosedForm(name, shape);
            if (name == "decoder.logits_dense.weight")
            {
                state["logits_dense.weight"] = Transposed(w, shape[0], w.Length / shape[0]);
                continue;
            }
            int rows = name.EndsWith("o_proj.weight", StringComparison.Ordinal) ? shape[0] * shape[1] : shape[0];
            bool kernel = name.EndsWith("proj.weight", StringComparison.Ordinal) || name.EndsWith("wi_fused.weight", StringComparison.Ordinal)
                || name.EndsWith("wo.weight", StringComparison.Ordinal);
            if (kernel) w = Transposed(w, rows, w.Length / rows);
            state["model." + name.Replace("mlp.wi_fused", "mlp.gate_up_proj").Replace("mlp.wo", "mlp.down_proj")] = w;
        }
        state["model.decoder.embeddings.embed.weight"] = Enumerable.Range(0, channels)
            .SelectMany(c => ClosedForm($"decoder.embeddings.{c}.weight", shapes[$"decoder.embeddings.{c}.weight"])).ToArray();
        return state;
    }

    [Fact(Timeout = 600000)]
    public async Task TransformersNames_LoadTheSameModel()
    {
        await Task.Yield();
        var reference = Load("dia_reference.json");
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new Dia<double>(architecture, TinyOptions());
        var shapes = reference.GetProperty("shapes").EnumerateObject().ToDictionary(p => p.Name, p => Ints(p.Value), StringComparer.Ordinal);
        var state = TransformersState(shapes, TinyOptions().Channels);
        var used = model.Network.LoadTorchWeights((name, shape) =>
        {
            var values = state[name];
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), values.Length);
            return values;
        });
        Assert.Empty(state.Keys.Where(n => !used.Contains(n)));
        var text = Ints(reference.GetProperty("text_ids"));
        double loss = model.Loss(text, Matrix(reference.GetProperty("frames"))).ToArray()[0];
        double target = reference.GetProperty("loss").GetDouble();
        Assert.True(Math.Abs(loss - target) <= 1e-6 * target, $"loss {loss} vs {target}");
    }

    /// <summary>With the release's checkpoint (<c>DIA_DIR</c>: nari-labs/Dia-1.6B-0626's dia-v1.pth, or a directory holding
    /// it or transformers' shards), generating the README's script as tools/reference-data/dia_realweights_reference.py does:
    /// the script tokenizes to the release's bytes, the encoder's output and the first steps' logits match, and greedy
    /// generation with guidance returns the same codes.</summary>
    [Fact(Timeout = 3600000)]
    public async Task ReleasedCheckpoint_MatchesTheRelease_WhenPresent()
    {
        await Task.Yield();
        string? path = Environment.GetEnvironmentVariable("DIA_DIR");
        if (path is null || !(File.Exists(path) || Directory.Exists(path))) return;
        var reference = Load("dia_realweights_reference.json");
        var options = new DiaOptions { Temperature = 0, CfgScale = 3, CfgFilterTopK = 45 };
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new Dia<double>(architecture, options);
        // Above 500M parameters the model would stream its weights, paging and compressing them on every step; the test
        // machine holds the 13 GB of double weights resident.
        model.DisableAutoStreaming();
        GC.Collect();
        model.LoadReferenceWeights(path);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var text = Dia<double>.Tokenize(reference.GetProperty("script").GetString()!);
        Assert.Equal(Ints(reference.GetProperty("text_ids")), text);
        var ids = new int[1, text.Length];
        for (int j = 0; j < text.Length; j++) ids[0, j] = text[j];
        Close(Doubles(reference.GetProperty("encoded")), model.Network.Encode(ids).ToArray(), 1e-9, "encoder output");

        var steps = reference.GetProperty("steps").EnumerateArray().ToArray();
        var observed = new List<(double[] Uncond, double[] Cond)>();
        model.StepLogits = (u, c) => { if (observed.Count < steps.Length) observed.Add((u, c)); };
        var codes = model.GenerateCodes(text, null, reference.GetProperty("max_tokens").GetInt32());
        model.StepLogits = null;
        for (int i = 0; i < steps.Length; i++)
        {
            Close(Doubles(steps[i].GetProperty("uncond")), observed[i].Uncond, 1e-9, $"step {i}'s unconditional logits");
            Close(Doubles(steps[i].GetProperty("cond")), observed[i].Cond, 1e-9, $"step {i}'s conditional logits");
        }
        Assert.Equal(Matrix(reference.GetProperty("generated")), codes);
        Assert.False(model.IsWeightStreamingActive);
    }
}
