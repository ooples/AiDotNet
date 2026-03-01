using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Graph;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data;

public class GraphBenchmarkTests
{
    [Fact]
    public void Qm9Options_DefaultValues()
    {
        var options = new Qm9DataLoaderOptions();
        Assert.True(options.AutoDownload);
        Assert.Equal(0, options.TargetProperty);
        Assert.Null(options.MaxSamples);
    }

    [Fact]
    public void ZincOptions_DefaultValues()
    {
        var options = new ZincDataLoaderOptions();
        Assert.True(options.AutoDownload);
        Assert.True(options.UseSubset);
    }

    [Fact]
    public void ProteinOptions_DefaultValues()
    {
        var options = new ProteinDataLoaderOptions();
        Assert.Equal(8.0, options.ContactThreshold);
        Assert.Equal(20, options.FeatureDimension);
        Assert.Equal(384, options.NumClasses);
    }

    [Fact]
    public void Wikidata5mOptions_DefaultValues()
    {
        var options = new Wikidata5mDataLoaderOptions();
        Assert.Equal(DatasetSplit.Train, options.Split);
        Assert.Equal(128, options.EmbeddingDimension);
    }

    [Fact]
    public void TemporalGraphOptions_DefaultValues()
    {
        var options = new TemporalGraphDataLoaderOptions();
        Assert.Equal(DatasetSplit.Train, options.Split);
        Assert.Equal(172, options.NodeFeatureDimension);
        Assert.Equal(172, options.EdgeFeatureDimension);
    }

    [Fact]
    public async Task ProteinDataLoader_LoadsCsvData()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "protein_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            string proteinDir = Path.Combine(tempDir, "proteins");
            Directory.CreateDirectory(proteinDir);

            // Create synthetic protein CSV files (residue features)
            for (int i = 0; i < 5; i++)
            {
                var lines = new List<string>();
                for (int r = 0; r < 10; r++) // 10 residues per protein
                {
                    var feats = Enumerable.Range(0, 20).Select(f => (f * 0.1 + r * 0.01).ToString("F3"));
                    lines.Add(string.Join(",", feats));
                }
                File.WriteAllLines(Path.Combine(proteinDir, $"protein_{i}.csv"), lines);
            }

            // Create labels file
            var labelLines = Enumerable.Range(0, 5).Select(i => $"protein_{i},{i % 3}");
            File.WriteAllLines(Path.Combine(tempDir, "labels.csv"), labelLines);

            var options = new ProteinDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                FeatureDimension = 20,
                NumClasses = 3
            };

            var loader = new ProteinDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(5, loader.TotalCount);
            Assert.Equal(20, loader.FeatureCount);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public async Task Wikidata5mDataLoader_LoadsTriplets()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "wikidata_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            Directory.CreateDirectory(tempDir);

            // Create synthetic triplet file
            var lines = new List<string>();
            for (int i = 0; i < 20; i++)
            {
                lines.Add($"Q{i}\tP{i % 5}\tQ{i + 100}");
            }
            File.WriteAllLines(Path.Combine(tempDir, "train.tsv"), lines);

            var options = new Wikidata5mDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                EmbeddingDimension = 16,
                MaxSamples = 20
            };

            var loader = new Wikidata5mDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(20, loader.TotalCount);
            Assert.Equal(32, loader.FeatureCount); // 2 * 16
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public async Task TemporalGraphDataLoader_LoadsInteractions()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "temporal_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            Directory.CreateDirectory(tempDir);

            // Create synthetic interactions CSV
            var lines = new List<string>();
            for (int i = 0; i < 15; i++)
            {
                // src, dst, timestamp, label, feature1, feature2
                lines.Add($"{i},{i + 10},{i * 1.5},{i % 2},0.{i},0.{i + 1}");
            }
            File.WriteAllLines(Path.Combine(tempDir, "interactions.csv"), lines);

            var options = new TemporalGraphDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                NodeFeatureDimension = 4,
                EdgeFeatureDimension = 2,
                MaxSamples = 15
            };

            var loader = new TemporalGraphDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(15, loader.TotalCount);
            Assert.Equal(6, loader.FeatureCount); // 4 + 2
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }
}
