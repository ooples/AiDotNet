using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

public class BatchCosineSimilarityTests
{
    [Fact]
    public void Scores_StreamCandidatesOnceAndSnapshotReusedBuffersAcrossBlocks()
    {
        const int count = 10001, width = 769;
        int enumerations = 0;
        var query = new Vector<double>(Enumerable.Repeat(1.0, width).ToArray());
        IEnumerable<Vector<double>> Candidates()
        {
            Assert.Equal(1, ++enumerations);
            var reused = new Vector<double>(width);
            for (int row = 0; row < count; row++)
            {
                reused.AsWritableSpan().Fill(row % 3 - 1);
                yield return reused;
            }
        }
        var scores = BatchCosineSimilarity.Scores(query, Candidates());
        Assert.Equal(count, scores.Length);
        for (int row = 0; row < count; row++) Assert.Equal((double)(row % 3 - 1), scores.GetFlat(row), 12);
    }

    [Fact]
    public void RowScores_BlocksKeepOrderAndMaskZeroAndNaNRows()
    {
        const int count = 2051, width = 129;
        var query = new Vector<double>(Enumerable.Repeat(1.0, width).ToArray());
        var rows = new Tensor<double>(new[] { count, width });
        for (int row = 0; row < count; row++)
            rows.AsWritableSpan().Slice(row * width, width).Fill(row % 3 - 1);
        rows.SetFlat((count - 1) * width, double.NaN);
        var scores = BatchCosineSimilarity.RowScores(query, rows);
        for (int row = 0; row < count - 1; row++) Assert.Equal((double)(row % 3 - 1), scores.GetFlat(row), 12);
        Assert.Equal(0.0, scores.GetFlat(count - 1));
    }

    [Theory]
    [InlineData(100f)]
    [InlineData(1e10f)]
    [InlineData(1e20f)]
    public void FloatScores_PreserveDirectionAcrossScales(float scale)
    {
        var query = new Vector<float>(new[] { scale, scale });
        var scores = BatchCosineSimilarity.Scores(query, new[]
        {
            query,
            new Vector<float>(new[] { -scale, -scale }),
            new Vector<float>(new[] { scale, -scale }),
            new Vector<float>(new[] { 0f, 0f })
        });

        Assert.Equal(new[] { 1f, -1f, 0f, 0f }, scores.ToArray());
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void HalfScores_DoNotOverflowTheNormProduct()
    {
        var query = new Vector<Half>(new[] { (Half)100 });
        var scores = BatchCosineSimilarity.Scores(query, new[] { query });

        Assert.Equal((Half)1, scores.GetFlat(0));
    }
#endif

    [Fact]
    public void DecimalZeroVectors_AreMaskedBeforeDivision()
    {
        var zero = new Vector<decimal>(new[] { 0m, 0m });
        var nonzero = new Vector<decimal>(new[] { 1m, 2m });

        Assert.Equal(new[] { 0m, 1m }, BatchCosineSimilarity.Scores(nonzero, new[] { zero, nonzero }).ToArray());
        Assert.Equal(new[] { 0m, 0m }, BatchCosineSimilarity.Scores(zero, new[] { zero, nonzero }).ToArray());
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void HalfScores_KeepEpsilonInWidePrecision()
    {
        var query = new Vector<Half>(new[] { (Half)0.001 });

        Assert.Equal((Half)0, BatchCosineSimilarity.Scores(query, new[] { query }, 1e-5).GetFlat(0));
        Assert.Equal((Half)1, BatchCosineSimilarity.Scores(query, new[] { query }, 1e-8).GetFlat(0));
    }
#endif

    [Fact]
    public void NaNNorm_ScoresZero()
    {
        var query = new Vector<double>(new[] { 1.0 });
        var candidate = new Vector<double>(new[] { double.NaN });

        Assert.Equal(0.0, BatchCosineSimilarity.Scores(query, new[] { candidate }).GetFlat(0));
    }
}
