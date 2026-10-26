using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Audio.Codecs;
using AiDotNet.Audio.Foundations;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// X-codec2, Llasa's codec, and its Wav2Vec2-BERT semantic encoder against the xcodec2 0.1.5 package and transformers'
/// Wav2Vec2BertModel and SeamlessM4TFeatureExtractor, built small (tools/reference-data/xcodec2_reference.py): every
/// parameter has a place, the features match the extractor's steps in float64, the 16th hidden state and the
/// decoded audio match to 1e-9, and the codes match exactly.
/// </summary>
public class XCodec2ReferenceTests
{
    private static JsonElement Load(string name)
    {
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    internal static double[] ClosedForm(string name, int[] shape)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        double fanIn = shape.Length >= 2 ? (double)count / shape[0] : 1;
        return Enumerable.Range(0, count).Select(i =>
        {
            long k = i;
            double s = Math.Sin(0.37 * i + 0.11 * name.Length + 0.0013 * (k * k % 9973));
            if (name.EndsWith("norm.weight", StringComparison.Ordinal) || name.EndsWith("norm1.weight", StringComparison.Ordinal)
                || name.EndsWith("norm2.weight", StringComparison.Ordinal)) return 1 + 0.2 * s;
            if (name.EndsWith("weight_g", StringComparison.Ordinal) || name.EndsWith("original0", StringComparison.Ordinal)) return 0.5 + 0.2 * s;
            if (name.EndsWith("act.alpha", StringComparison.Ordinal) || name.EndsWith("act.bias", StringComparison.Ordinal)) return 0.1 * s;
            if (name.EndsWith("project_in.weight", StringComparison.Ordinal)) return 8 * s / Math.Sqrt(fanIn);
            if (shape.Length >= 2) return 0.6 * s / Math.Sqrt(fanIn);
            return 0.05 * s;
        }).ToArray();
    }

    internal static XCodec2Configuration Configuration(JsonElement config)
    {
        var semantic = config.GetProperty("semantic");
        return new XCodec2Configuration
        {
            EncoderDim = config.GetProperty("encoder_dim").GetInt32(),
            EncoderRates = config.GetProperty("encoder_rates").EnumerateArray().Select(v => v.GetInt32()).ToArray(),
            AcousticDim = config.GetProperty("acoustic_dim").GetInt32(),
            SemanticDim = config.GetProperty("semantic_dim").GetInt32(),
            SemanticLayer = config.GetProperty("semantic_layer").GetInt32(),
            DecoderDim = config.GetProperty("decoder_dim").GetInt32(),
            DecoderDepth = config.GetProperty("decoder_depth").GetInt32(),
            DecoderHeads = config.GetProperty("decoder_heads").GetInt32(),
            Semantic = new Wav2Vec2BertConfiguration
            {
                HiddenSize = semantic.GetProperty("hidden_size").GetInt32(),
                Layers = semantic.GetProperty("num_hidden_layers").GetInt32(),
                Heads = semantic.GetProperty("num_attention_heads").GetInt32(),
                IntermediateSize = semantic.GetProperty("intermediate_size").GetInt32(),
            },
        };
    }

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    private static double[,] Matrix(JsonElement e)
    {
        var rows = e.EnumerateArray().Select(Doubles).ToArray();
        var m = new double[rows.Length, rows[0].Length];
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < rows[0].Length; j++) m[i, j] = rows[i][j];
        return m;
    }

    private static Tensor<double> Tensor(double[,] m)
    {
        var t = new Tensor<double>(new[] { m.GetLength(0), m.GetLength(1) });
        for (int i = 0; i < m.GetLength(0); i++)
            for (int j = 0; j < m.GetLength(1); j++) t[i, j] = m[i, j];
        return t;
    }

    [Fact(Timeout = 300000)]
    public async Task EncodeAndDecode_MatchXCodec2()
    {
        await Task.Yield();
        var reference = Load("xcodec2_reference.json");
        var shapes = reference.GetProperty("shapes").EnumerateObject()
            .ToDictionary(p => p.Name, p => p.Value.EnumerateArray().Select(v => v.GetInt32()).ToArray(), StringComparer.Ordinal);
        var codec = new XCodec2<double>(new CpuEngine(), Configuration(reference.GetProperty("config")), new List<LayerBase<double>>());
        var buffers = reference.GetProperty("buffers").EnumerateObject().ToDictionary(p => p.Name, p => Doubles(p.Value), StringComparer.Ordinal);
        var used = codec.Load(name => shapes.TryGetValue(name, out var shape) ? ClosedForm(name, shape) : buffers.TryGetValue(name, out var values) ? values : null);
        Assert.Contains("generator.head.istft.window", used);
        // The reference's semantic model has a 17th layer that X-codec2 never reads; everything else has a place.
        Assert.DoesNotContain(shapes.Keys, n => !used.Contains(n) && !n.StartsWith("semantic_model.encoder.layers.16.", StringComparison.Ordinal));

        foreach (var c in reference.GetProperty("cases").EnumerateArray())
        {
            var audio = Doubles(c.GetProperty("audio"));
            var padded = codec.PadForEncoding(audio);
            Assert.Equal(0, padded.Length % 320);

            // The extractor's own output was cast to float32 on the way; its steps in float64 match up to the rounding of the
            // filterbank, which folds DC removal, pre-emphasis, windowing and the DFT into one matrix (about 1e-8 after the
            // per-bin normalization).
            var features = codec.Features(padded);
            var expected32 = Matrix(c.GetProperty("features"));
            var expectedFeatures = Matrix(c.GetProperty("features64"));
            Assert.Equal(new[] { expectedFeatures.GetLength(0), expectedFeatures.GetLength(1) }, features.Shape.ToArray());
            for (int i = 0; i < expectedFeatures.GetLength(0); i++)
                for (int j = 0; j < expectedFeatures.GetLength(1); j++)
                {
                    Assert.True(Math.Abs(expectedFeatures[i, j] - features[i, j]) <= 1e-6, $"feature [{i}, {j}]: {expectedFeatures[i, j]} vs {features[i, j]}");
                    Assert.True(Math.Abs(expected32[i, j] - features[i, j]) <= 0.02, $"float32 feature [{i}, {j}]: {expected32[i, j]} vs {features[i, j]}");
                }

            var hidden = codec.Semantic.HiddenStates(Tensor(expectedFeatures))[16];
            var expectedHidden = Matrix(c.GetProperty("hidden16"));
            for (int i = 0; i < expectedHidden.GetLength(0); i++)
                for (int j = 0; j < expectedHidden.GetLength(1); j++)
                    Assert.True(Math.Abs(expectedHidden[i, j] - hidden[i, j]) <= 1e-9, $"hidden [{i}, {j}]: {expectedHidden[i, j]} vs {hidden[i, j]}");

            var expectedCodes = c.GetProperty("codes").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            Assert.Equal(expectedCodes, codec.Encode(padded, Tensor(Matrix(c.GetProperty("features")))));
            Assert.Equal(c.GetProperty("codes64").EnumerateArray().Select(v => v.GetInt32()).ToArray(), codec.Encode(audio));

            var expected = Doubles(c.GetProperty("decoded"));
            var decoded = codec.Decode(expectedCodes);
            Assert.Equal(expected.Length, decoded.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - decoded[i]) <= 1e-9, $"sample {i}: {expected[i]} vs {decoded[i]}");
        }
    }
}
