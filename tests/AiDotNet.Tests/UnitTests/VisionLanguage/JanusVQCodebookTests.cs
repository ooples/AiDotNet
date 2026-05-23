using System;
using System.Threading.Tasks;
using AiDotNet.VisionLanguage.Unified;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Unit tests for Janus-Pro's VQ-VAE codebook (Chen et al. DeepSeek 2025, arXiv:2501.17811 §3.1).
/// Covers the IsLoaded contract (Lookup / Quantize / LookupGrid throw until LoadCodebook is called),
/// shape correctness, quantize round-trip on the codebook's own entries, out-of-range clamping, and
/// hot-swap shape validation.
/// </summary>
public class JanusVQCodebookTests
{
    /// <summary>Builds a small deterministic spectral-style codebook for tests that need a populated lookup table.</summary>
    private static JanusVQCodebook<double> BuildLoadedCodebook(int codebookSize, int embeddingDim, double scale = 1.0)
    {
        var codebook = new JanusVQCodebook<double>(codebookSize: codebookSize, embeddingDim: embeddingDim);
        var book = new double[codebookSize, embeddingDim];
        double inv = scale / Math.Sqrt(Math.Max(1, embeddingDim));
        for (int id = 0; id < codebookSize; id++)
        {
            for (int d = 0; d < embeddingDim; d++)
            {
                double angle = (id + 1) * Math.PI * (d + 1) * 0.0001;
                book[id, d] = inv * (Math.Sin(angle) + 0.5 * Math.Cos(angle * 2.7183));
            }
        }
        codebook.LoadCodebook(book);
        return codebook;
    }

    [Fact(Timeout = 30000)]
    public async Task Lookup_BeforeLoad_Throws()
    {
        await Task.Yield();
        var codebook = new JanusVQCodebook<double>(codebookSize: 16, embeddingDim: 4);

        Assert.False(codebook.IsLoaded);
        Assert.Throws<InvalidOperationException>(() => codebook.Lookup(0));
    }

    [Fact(Timeout = 30000)]
    public async Task Quantize_BeforeLoad_Throws()
    {
        await Task.Yield();
        var codebook = new JanusVQCodebook<double>(codebookSize: 16, embeddingDim: 4);
        var dummy = new Tensor<double>([4]);

        Assert.Throws<InvalidOperationException>(() => codebook.Quantize(dummy));
    }

    [Fact(Timeout = 30000)]
    public async Task LookupGrid_BeforeLoad_Throws()
    {
        await Task.Yield();
        var codebook = new JanusVQCodebook<double>(codebookSize: 16, embeddingDim: 4);
        var tokens = new[] { 0, 1, 2, 3 };

        Assert.Throws<InvalidOperationException>(() => codebook.LookupGrid(tokens, 2, 2));
    }

    [Fact(Timeout = 30000)]
    public async Task Lookup_ReturnsTensorOfEmbeddingDim()
    {
        await Task.Yield();
        var codebook = BuildLoadedCodebook(codebookSize: 4096, embeddingDim: 8);

        var embed = codebook.Lookup(tokenId: 1234);

        Assert.Equal(8, embed.Length);
    }

    [Fact(Timeout = 30000)]
    public async Task Quantize_OnLookupOutput_RoundTripsToSameId()
    {
        await Task.Yield();
        var codebook = BuildLoadedCodebook(codebookSize: 512, embeddingDim: 4);

        // Quantizing the codebook's own embedding must return its own ID (nearest-neighbour to itself).
        for (int id = 0; id < 8; id++)
        {
            var embed = codebook.Lookup(id);
            int recovered = codebook.Quantize(embed);
            Assert.Equal(id, recovered);
        }
    }

    [Fact(Timeout = 30000)]
    public async Task Lookup_OutOfRangeTokenId_ClampsToValidEntry()
    {
        await Task.Yield();
        var codebook = BuildLoadedCodebook(codebookSize: 256, embeddingDim: 4);

        var positiveOOR = codebook.Lookup(10_000);
        var negativeOOR = codebook.Lookup(-100);

        Assert.Equal(4, positiveOOR.Length);
        Assert.Equal(4, negativeOOR.Length);
        // Both should equal valid codebook entries (255 and 0 respectively).
        var lastEntry = codebook.Lookup(255);
        var firstEntry = codebook.Lookup(0);
        for (int d = 0; d < 4; d++)
        {
            Assert.Equal(lastEntry[d], positiveOOR[d]);
            Assert.Equal(firstEntry[d], negativeOOR[d]);
        }
    }

    [Fact(Timeout = 30000)]
    public async Task LookupGrid_ReturnsFlattenedHxWxEmbedShape()
    {
        await Task.Yield();
        var codebook = BuildLoadedCodebook(codebookSize: 256, embeddingDim: 4);
        var tokens = new[] { 1, 2, 3, 4, 5, 6 };

        var grid = codebook.LookupGrid(tokens, gridHeight: 2, gridWidth: 3);

        Assert.Equal(2 * 3 * 4, grid.Length);
    }

    [Fact(Timeout = 30000)]
    public async Task LoadCodebook_OverwritesEntries()
    {
        await Task.Yield();
        var codebook = new JanusVQCodebook<double>(codebookSize: 16, embeddingDim: 4);
        var customBook = new double[16, 4];
        for (int id = 0; id < 16; id++)
            for (int d = 0; d < 4; d++)
                customBook[id, d] = id * 100.0 + d;

        codebook.LoadCodebook(customBook);

        Assert.True(codebook.IsLoaded);
        var embed = codebook.Lookup(7);
        Assert.Equal(700.0, embed[0]);
        Assert.Equal(701.0, embed[1]);
        Assert.Equal(702.0, embed[2]);
        Assert.Equal(703.0, embed[3]);
    }

    [Fact(Timeout = 30000)]
    public async Task LoadCodebook_RejectsWrongShape()
    {
        await Task.Yield();
        var codebook = new JanusVQCodebook<double>(codebookSize: 16, embeddingDim: 4);
        var wrongShape = new double[16, 8];

        Assert.Throws<ArgumentException>(() => codebook.LoadCodebook(wrongShape));
    }

    [Fact(Timeout = 30000)]
    public async Task Constructor_RejectsNonPositiveCodebookSize()
    {
        await Task.Yield();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new JanusVQCodebook<double>(codebookSize: 0, embeddingDim: 8));
    }
}
