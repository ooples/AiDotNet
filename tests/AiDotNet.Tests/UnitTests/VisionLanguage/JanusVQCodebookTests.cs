using System;
using System.Threading.Tasks;
using AiDotNet.VisionLanguage.Unified;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Unit tests for Janus-Pro's VQ-VAE codebook (Chen et al. DeepSeek 2025, arXiv:2501.17811 §3.1).
/// Covers shape correctness, quantization round-trip on the codebook's own entries, and
/// out-of-range clamping. Does not depend on trained weights.
/// </summary>
public class JanusVQCodebookTests
{
    [Fact(Timeout = 30000)]
    public async Task Lookup_ReturnsTensorOfEmbeddingDim()
    {
        await Task.Yield();
        var codebook = new JanusVQCodebook<double>(codebookSize: 4096, embeddingDim: 8);

        var embed = codebook.Lookup(tokenId: 1234);

        Assert.Equal(8, embed.Length);
    }

    [Fact(Timeout = 30000)]
    public async Task Quantize_OnLookupOutput_RoundTripsToSameId()
    {
        await Task.Yield();
        var codebook = new JanusVQCodebook<double>(codebookSize: 512, embeddingDim: 4);

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
        var codebook = new JanusVQCodebook<double>(codebookSize: 256, embeddingDim: 4);

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
        var codebook = new JanusVQCodebook<double>(codebookSize: 256, embeddingDim: 4);
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
