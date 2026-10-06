using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The SoundStorm Conformer behind Pheme's acoustic stage reproduces Pheme's own <c>modules/conformer.py</c>
/// (after lucidrains/soundstorm-pytorch): macaron feed-forwards, rotary self-attention with an inner width independent
/// of the model width, the GLU / depthwise / channel-norm convolution module, and the post-norm.
/// </summary>
/// <remarks><c>ReferenceData/soundstorm_conformer_reference.json</c> holds a tiny Conformer's float32 weights, an input
/// and the float64 output (<c>tools/reference-data/conformer_reference.py</c>).</remarks>
public class SoundStormConformerParityTests
{
    private static JObject Reference()
    {
        const string fileName = "soundstorm_conformer_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        if (!File.Exists(output))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(output))
            {
                output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
                dir = dir.Parent;
            }
        }
        return JObject.Parse(File.ReadAllText(output));
    }

    [Fact(Timeout = 120000)]
    public async Task Output_MatchesPhemesConformer()
    {
        await Task.Yield();
        var fixture = Reference();
        var c = fixture["config"]!;
        var configuration = new SoundStormConformerConfiguration(
            Dim: (int)c["dim"]!, Layers: (int)c["num_layers"]!, Heads: (int)c["heads"]!, HeadDim: (int)c["dim_head"]!,
            FeedForwardMultiplier: (int)c["ff_mult"]!, ConvExpansion: (int)c["conv_expansion_factor"]!,
            ConvKernel: (int)c["conv_kernel_size"]!);
        var engine = AiDotNetEngine.Current;
        var conformer = new SoundStormConformer<double>(engine, new List<LayerBase<double>>(), configuration);

        string path = Path.Combine(Path.GetTempPath(), $"conformer_reference_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)fixture["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            conformer.LoadTorchWeights(engine, "", (name, shape) =>
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

        int time = (int)fixture["time"]!;
        var values = fixture["input"]!.Select(v => (double)v).ToArray();
        var input = new Tensor<double>(new[] { time, configuration.Dim }, new Vector<double>(values));
        var output = conformer.Forward(input, training: false, new Random(0));
        var expected = fixture["output"]!.Select(v => (double)v).ToArray();
        Assert.Equal(expected.Length, output.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - output[i]) <= 1e-9 * Math.Max(1, Math.Abs(expected[i])),
                $"output[{i}]: reference {expected[i]:R}, port {output[i]:R}.");
    }
}
