using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Mega-TTS's building blocks against the code they come from (tools/reference-data/megatts_modules_reference.py):
/// NATSpeech's FastSpeech content encoder, duration predictor, convolution stacks and causal decoder layers (the prosody
/// language model's), and SyntaSpeech's multi-window discriminator.
/// </summary>
public class MegaTtsModulesReferenceTests
{
    [Fact(Timeout = 300000)]
    public async Task Modules_MatchNatSpeechAndSyntaSpeech()
    {
        await Task.Yield();
        const string fileName = "megatts_modules_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        var f = JObject.Parse(File.ReadAllText(path));
        double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
        var state = (JObject)f["state"]!;
        var used = new HashSet<string>();
        double[] Read(string name)
        {
            used.Add(name);
            return Doubles(state[name] ?? throw new InvalidDataException(name));
        }
        var engine = AiDotNetEngine.Current;
        var layers = new List<LayerBase<double>>();
        var encoder = new MegaTtsContentEncoder<double>(engine, layers, 20, 16, 2, 2, 5);
        encoder.LoadTorch(engine, Read, "encoder");
        var duration = new MegaTtsDurationPredictor<double>(engine, layers, 16, 16, 3, 3);
        duration.LoadTorch(engine, Read, "duration");
        var blocks = new MegaTtsConvBlocks<double>(engine, layers, 16, 12, 5, 2);
        blocks.LoadTorch(engine, Read, "blocks");
        var decoder = Enumerable.Range(0, 2).Select(_ => new MegaTtsTransformerLayer<double>(engine, layers, 16, 2, 5, decoder: true)).ToList();
        for (int i = 0; i < 2; i++) decoder[i].LoadTorch(engine, Read, $"decoder.{i}.op");
        var disc = new MegaTtsMelDiscriminator<double>(engine, layers, new[] { 8, 16 }, 20, 6);
        disc.LoadTorch(Read, "disc");
        foreach (var layer in layers) layer.SetTrainingMode(false);
        // The decoder layers' unused encoder attention (the prosody language model is decoder-only).
        Assert.Empty(state.Properties().Select(p => p.Name).Where(n => !used.Contains(n) && !n.Contains("encoder_attn") && !n.Contains(".op.layer_norm2.")
            || n.StartsWith("encoder.", StringComparison.Ordinal) && !used.Contains(n)));

        void Close(JToken expected, Tensor<double> actual, string what)
        {
            var e = Doubles(expected);
            Assert.Equal(e.Length, actual.Length);
            for (int i = 0; i < e.Length; i++)
                Assert.True(Math.Abs(e[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {actual[i]}");
        }
        using var _ = new NoGradScope<double>();
        var tokenList = f["tokens"]!.Select(v => (int)v).ToArray();
        var tokens = new int[1, tokenList.Length];
        var padding = new bool[1, tokenList.Length];
        for (int j = 0; j < tokenList.Length; j++) { tokens[0, j] = tokenList[j]; padding[0, j] = tokenList[j] == 0; }
        var encoded = encoder.Forward(tokens);
        Close(f["encoder"]!, encoded, "encoder");
        Close(f["duration"]!, duration.Forward(encoded, padding), "duration");
        Close(f["blocks"]!, blocks.Forward(new Tensor<double>(new[] { 1, 10, 16 }, new Vector<double>(Doubles(f["x16"]!)))), "blocks");
        var seq = new Tensor<double>(new[] { 1, 7, 16 }, new Vector<double>(Doubles(f["seq"]!)));
        var h = engine.TensorAdd(seq, MegaTtsPositions.Of<double>(new bool[1, 7], 16));
        foreach (var layer in decoder) h = layer.Forward(h, null, null);
        Close(f["decoder"]!, h, "decoder");
        var mel = new Tensor<double>(new[] { 1, 20, 20 }, new Vector<double>(Doubles(f["mel"]!)));
        Close(f["validity"]!, disc.Forward(mel, f["starts"]!.Select(v => (int)v).ToArray())!, "validity");
    }
}
