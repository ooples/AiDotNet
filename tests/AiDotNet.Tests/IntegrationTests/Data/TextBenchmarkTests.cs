using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Data.Sampling;
using AiDotNet.Data.Text;
using AiDotNet.Data.Text.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data;

public class TextBenchmarkTests
{
    [Fact]
    public void GlueOptions_DefaultValues()
    {
        var options = new GlueDataLoaderOptions();
        Assert.Equal(GlueTask.SST2, options.Task);
        Assert.Equal(128, options.MaxSequenceLength);
        Assert.Equal(30000, options.VocabularySize);
        Assert.True(options.AutoDownload);
    }

    [Fact]
    public void SuperGlueOptions_DefaultValues()
    {
        var options = new SuperGlueDataLoaderOptions();
        Assert.Equal(SuperGlueTask.BoolQ, options.Task);
        Assert.Equal(256, options.MaxSequenceLength);
        Assert.Equal(30000, options.VocabularySize);
    }

    [Fact]
    public void SquadOptions_DefaultValues()
    {
        var options = new SquadDataLoaderOptions();
        Assert.Equal(384, options.MaxContextLength);
        Assert.Equal(64, options.MaxQuestionLength);
        Assert.False(options.Version2);
    }

    [Fact]
    public void WikiText103Options_DefaultValues()
    {
        var options = new WikiText103DataLoaderOptions();
        Assert.Equal(256, options.SequenceLength);
        Assert.Equal(30000, options.VocabularySize);
    }

    [Fact]
    public async Task GlueDataLoader_LoadsTsvData()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "glue_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            // CoLA directory name and TSV format: index, label, star, sentence
            string trainDir = Path.Combine(tempDir, "CoLA");
            Directory.CreateDirectory(trainDir);

            var lines = new[]
            {
                "header_idx\theader_label\theader_star\theader_sentence",
                "gj04\t1\t*\tThe sailors rode the breeze clear of the rocks.",
                "gj04\t0\t*\tThe weights made the rope stretched.",
                "gj04\t1\t*\tHe danced the night away."
            };
            File.WriteAllLines(Path.Combine(trainDir, "train.tsv"), lines);

            var options = new GlueDataLoaderOptions
            {
                Task = GlueTask.CoLA,
                DataPath = tempDir,
                AutoDownload = false,
                MaxSamples = 3
            };

            var loader = new GlueDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(3, loader.TotalCount);
            Assert.True(loader.FeatureCount > 0);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public async Task WikiText103DataLoader_LoadsTextData()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "wikitext_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            // WikiText103 expects {DataPath}/wikitext-103/wiki.train.tokens or {DataPath}/wiki.train.tokens
            Directory.CreateDirectory(tempDir);

            // Create text file with enough tokens directly at DataPath level
            string text = string.Join(" ", Enumerable.Range(0, 600).Select(i => $"word{i}"));
            File.WriteAllText(Path.Combine(tempDir, "wiki.train.tokens"), text);

            var options = new WikiText103DataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                SequenceLength = 32,
                VocabularySize = 1000
            };

            var loader = new WikiText103DataLoader<double>(options);
            await loader.LoadAsync();

            Assert.True(loader.TotalCount > 0);
            Assert.Equal(32, loader.FeatureCount);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public async Task StreamingTextDataset_LoadsTextFiles()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "streaming_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            Directory.CreateDirectory(tempDir);

            // Create text files with content
            for (int i = 0; i < 3; i++)
            {
                string text = string.Join(" ", Enumerable.Range(0, 200).Select(j => $"token{j}file{i}"));
                File.WriteAllText(Path.Combine(tempDir, $"corpus_{i}.txt"), text);
            }

            var options = new StreamingTextDatasetOptions
            {
                DataPath = tempDir,
                SequenceLength = 32,
                VocabularySize = 1000,
                Seed = 42,
                MaxSamples = 10
            };

            var loader = new StreamingTextDataset<double>(options);
            await loader.LoadAsync();

            Assert.True(loader.TotalCount > 0);
            Assert.Equal(32, loader.FeatureCount);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void DomainMixingSampler_ProducesIndices()
    {
        var domainSizes = new[] { 100, 200, 50 };
        var domainWeights = new[] { 0.5, 0.3, 0.2 };

        var sampler = new DomainMixingSampler(domainSizes, domainWeights, seed: 42);

        Assert.Equal(350, sampler.Length);

        var indices = sampler.GetIndices().ToArray();
        Assert.Equal(350, indices.Length);

        // All indices should be within valid range
        foreach (var idx in indices)
        {
            Assert.InRange(idx, 0, 349);
        }
    }

    [Fact]
    public void DomainMixingSampler_ThrowsOnMismatchedArrays()
    {
        Assert.Throws<ArgumentException>(() =>
            new DomainMixingSampler(new[] { 100, 200 }, new[] { 0.5, 0.3, 0.2 }));
    }
}
