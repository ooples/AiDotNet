using System;
using System.IO;
using System.Linq;
using AiDotNet.ComputerVision.Weights;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.ComputerVision;

/// <summary>
/// WeightLoader on a checkpoint in torch.save's legacy (non-ZIP) format (tools/reference-data/legacy_torch_reference.py):
/// nested dictionaries flattened with dots, views sharing a storage (a transpose and a slice) read through their strides,
/// and float64 and float16 storages, as torch reads them.
/// </summary>
public class WeightLoaderLegacyTorchTests
{
    private static string Find(string fileName)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "ComputerVision", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "ComputerVision", "ReferenceData", fileName);
        return path;
    }

    [Fact]
    public void LegacyCheckpoint_LoadsAsTorchReadsIt()
    {
        var expected = JObject.Parse(File.ReadAllText(Find("legacy_torch.json")));
        var weights = new WeightLoader().LoadWeights(Find("legacy_torch.pt"));
        Assert.Equal(expected.Properties().Select(p => p.Name).OrderBy(n => n, StringComparer.Ordinal), weights.Keys.OrderBy(n => n, StringComparer.Ordinal));
        foreach (var p in expected.Properties())
        {
            var tensor = weights[p.Name];
            Assert.Equal(p.Value["shape"]!.Select(v => (int)v), tensor.Shape.ToArray());
            var values = p.Value["values"]!.Select(v => (float)(double)v).ToArray();
            Assert.Equal(values, tensor.ToArray());
        }
    }
}
