using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Foundations;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// mHuBERT's discrete units against the release's pipeline (<c>tools/reference-data/hubert_units_reference.py</c>): the
/// joblib k-means reader on a file the installed joblib writes (aligned) and on the 2021 release file, and layer-11
/// features and k-means labels of the released mHuBERT (gated on LLAMAOMNI_HUBERT_DIR, holding
/// <c>mhubert_base_vp_en_es_fr_it3.pt</c> and its <c>_L11_km1000.bin</c>).
/// </summary>
public class HubertUnitsReferenceTests
{
    private static JObject Fixture()
    {
        const string fileName = "hubert_units_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "Audio", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "Audio", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    [Fact]
    public void JoblibReader_ReadsAnAlignedFile()
    {
        var f = Fixture();
        string path = Path.Combine(Path.GetTempPath(), $"kmeans_{Guid.NewGuid():N}.bin");
        File.WriteAllBytes(path, Convert.FromBase64String((string)f["tiny_joblib_base64"]!));
        try
        {
            var centers = JoblibKMeans.ReadClusterCenters(path);
            var expected = f["tiny_centers"]!.Select(v => (double)v).ToArray();
            Assert.Equal(5, centers.GetLength(0));
            Assert.Equal(3, centers.GetLength(1));
            for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], centers[i / 3, i % 3]);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact(Timeout = 900000)]
    public async Task ReleasedMHubert_UnitsMatchTheRelease()
    {
        await Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("LLAMAOMNI_HUBERT_DIR");
        string checkpoint = dir is null ? "" : Path.Combine(dir, "mhubert_base_vp_en_es_fr_it3.pt");
        if (!File.Exists(checkpoint)) return;
        var f = Fixture();
        var centers = JoblibKMeans.ReadClusterCenters(Path.Combine(dir!, "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin"));
        var head = f["centroids_head"]!.Select(v => (double)v).ToArray();
        for (int i = 0; i < head.Length; i++) Assert.Equal(head[i], centers[i / 768, i % 768]);

        var extractor = new HubertUnitExtractor<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), HubertUnitExtractor<double>.MHubertBase(), 11, 1000)
        {
            Centroids = centers,
        };
        var weights = new WeightLoader().LoadWeights(checkpoint);
        // The fairseq checkpoint keeps the state under "model".
        extractor.LoadFairseqWeights(name => weights.TryGetValue("model." + name, out var t) ? t.ToVector().Select(v => (double)v).ToArray() : null);
        var audio = f["audio"]!.Select(v => (double)v).ToArray();
        var features = extractor.Network.Forward(new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio)), 11)[11];
        var expected = f["features"]!.Select(v => (double)v).ToArray();
        Assert.Equal(expected.Length, features.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - features[i]) <= 1e-8 * Math.Max(1, Math.Abs(expected[i])), $"features[{i}]: {expected[i]:R} vs {features[i]:R}");
        Assert.Equal(f["labels"]!.Select(v => (int)v).ToArray(), extractor.Units(audio, deduplicate: false));
        Assert.Equal(f["merged"]!.Select(v => (int)v).ToArray(), extractor.Units(audio, deduplicate: true));
    }
}
