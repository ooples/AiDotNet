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
/// The speaker encoder behind Pheme reproduces pyannote.audio's <c>XVectorSincNet</c> (the <c>pyannote/embedding</c>
/// architecture): SincNet front end, TDNN layers with batch norm, statistics pooling and the embedding projection.
/// </summary>
/// <remarks><c>ReferenceData/pyannote_xvector_reference.json</c> holds a seeded model's float32 weights (batch norms
/// with non-trivial running statistics), an 8000-sample waveform and its float64 embedding
/// (<c>tools/reference-data/xvector_reference.py</c>).</remarks>
public class PyannoteXVectorParityTests
{
    private static JObject Reference()
    {
        const string fileName = "pyannote_xvector_reference.json";
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
    public async Task Embedding_MatchesPyannote()
    {
        await Task.Yield();
        var fixture = Reference();
        var engine = AiDotNetEngine.Current;
        var layers = new List<LayerBase<double>>();
        var encoder = new PyannoteXVector<double>(engine, layers);
        foreach (var layer in layers) layer.SetTrainingMode(false);

        string path = Path.Combine(Path.GetTempPath(), $"xvector_reference_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)fixture["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            encoder.LoadTorchWeights((name, shape) =>
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

        var samples = fixture["waveform"]!.Select(v => (double)v).ToArray();
        var waveform = new Tensor<double>(new[] { 1, 1, samples.Length }, new Vector<double>(samples));
        var embedding = encoder.Forward(waveform);
        var expected = fixture["embedding"]!.Select(v => (double)v).ToArray();
        Assert.Equal(expected.Length, embedding.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - embedding[i]) <= 1e-8 * Math.Max(1, Math.Abs(expected[i])),
                $"embedding[{i}]: reference {expected[i]:R}, port {embedding[i]:R}.");
    }
}
