using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ComputerVision;

/// <summary>
/// EfficientTAM against the official yformer/EfficientTAM code (tools/reference-data/efficienttam_reference.py): the ViTDet
/// encoder and neck, the mask decoder without high-resolution features, and SAM 2's memory with the full cross-attention and
/// with both forms of the efficient 2×2 pooled one, tracking five frames from a click and a mask. The weights load from the
/// official state-dict names through the port of Hugging Face's renaming.
/// </summary>
public class EfficientTamReferenceTests
{
    private const int Size = 128;

    private static readonly Lazy<JsonElement> Data = new(() =>
    {
        const string name = "efficienttam_reference.json";
        string relative = Path.Combine("ComputerVision", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    private static Sam2Configuration Configuration(MemoryCrossAttention attention) => Sam2Configuration.Vit(small: false) with
    {
        ImageSize = Size, EmbedDims = new[] { 48 }, VitDepth = 4, VitHeads = 3, VitWindow = 3, VitGlobalBlocks = new[] { 1, 3 },
        VitPositionGrid = 4, Dim = 128, DecoderHeads = 2, DecoderMlp = 64, IouHidden = 16,
        Memory = new Sam2MemoryConfiguration
        {
            MemoryDim = 64, MaskMemories = 3, AttentionLayers = 2, AttentionFeedForward = 64, FuserHidden = 512, MaxObjectPointers = 4,
            Dropout = 0.0, CrossAttention = attention,
        },
    };

    [Theory(Timeout = 600000)]
    [InlineData("full", MemoryCrossAttention.Full)]
    [InlineData("scores", MemoryCrossAttention.PooledScores)]
    [InlineData("keys", MemoryCrossAttention.PooledKeys)]
    public async Task Tracking_MatchesTheOfficialCode(string variant, MemoryCrossAttention attention)
    {
        await Task.Yield();
        var k = Data.Value;
        var layers = new List<LayerBase<double>>();
        var net = new Sam2Network<double>(AiDotNetEngine.Current, layers, Configuration(attention));
        var state = k.GetProperty("state").EnumerateObject().ToDictionary(p => Sam2Network<double>.HuggingFaceName(p.Name), p => Values(p.Value));
        net.LoadTorchWeights((name, shape) =>
        {
            var v = name == "prompt_encoder.point_embed.weight"
                ? Enumerable.Range(0, shape[0]).SelectMany(i => state[$"prompt_encoder.point_embed.{i}.weight"]).ToArray()
                : state[name];
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        var frames = Values(k.GetProperty("frames"));
        Tensor<double> Frame(int t) => new(new[] { 1, 3, Size, Size }, new Vector<double>(frames.Skip(t * 3 * Size * Size).Take(3 * Size * Size).ToArray()));
        var mask = new Tensor<double>(new[] { 1, 1, Size, Size }, new Vector<double>(Values(k.GetProperty("mask"))));
        var prompts = new Dictionary<int, Sam2Prompt<double>>
        {
            [0] = new(new Tensor<double>(new[] { 1, 1, 2 }, new Vector<double>(new[] { 60.0, 50.0 })), new Tensor<double>(new[] { 1, 1 }, new Vector<double>(new[] { 1.0 })), null),
            [2] = new(null, null, mask),
        };
        var tracker = new Sam2VideoTracker<double>(AiDotNetEngine.Current, net);
        Sam2FrameOutput<double>[] outputs;
        using (new NoGradScope<double>())
            outputs = tracker.Propagate(5, t => net.EncodeFeatures(Frame(t)), prompts);
        var expected = k.GetProperty("variants").GetProperty(variant).GetProperty("frames").EnumerateArray().ToArray();
        for (int t = 0; t < 5; t++)
        {
            var logits = Values(expected[t].GetProperty("logits"));
            Assert.Equal(logits.Length, outputs[t].LowRes.Length);
            for (int i = 0; i < logits.Length; i++)
                Assert.True(Math.Abs(logits[i] - outputs[t].LowRes[i]) < 1e-9, $"{variant} frame {t} logits[{i}]: {logits[i]} vs {outputs[t].LowRes[i]}");
            double score = Values(expected[t].GetProperty("object_score"))[0];
            Assert.True(Math.Abs(score - outputs[t].ObjectScore) < 1e-9, $"{variant} frame {t} object score: {score} vs {outputs[t].ObjectScore}");
        }
    }
}
